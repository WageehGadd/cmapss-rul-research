"""Session 7 inputs and the predeclared experiment matrix.

Reads the frozen definitions and sensor subsets (checking every file against its session
manifest), loads ONLY the training and validation arrays of Session 1, and builds the immutable
experiment plan. A condition's identity depends only on scientific inputs: the model definition,
the ordered sensor columns, the training seed and protocol, the data identity and the timing
protocol. Nothing here ever opens the official test arrays.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from .benchmark_config import resolve_path
from .benchmark_data import array_digest
from .sequence_data import SequenceData

CLASSICAL = ("random_forest", "gradient_boosting")
DEEP = ("lstm", "gru", "tcn", "transformer")
FAMILY = {"random_forest": "classical", "gradient_boosting": "classical", "lstm": "recurrent",
          "gru": "recurrent", "tcn": "convolutional", "transformer": "attention"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_session7_config(config_path):
    config_path = Path(config_path).resolve()
    text = config_path.read_text()
    cfg = yaml.safe_load(text)
    required = {"session1": ["config", "processed_dir"], "session2": ["config"],
                "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
                "frozen_sources": ["selected_models", "subsets_topk", "subsets_randomk", "lstm", "gru", "tcn",
                                   "transformer", "manifests", "historical_results"],
                "matrix": ["models", "full_budget", "budgets", "ranked_strategies", "random_subset_seeds",
                           "training_seeds"],
                "execution": ["device", "classical_n_jobs_train", "classical_n_jobs_predict", "experiment_name",
                              "reproducibility_gate"],
                "analysis": ["close_to_full_tolerance_pct", "per_engine_metric", "statistics"],
                "outputs": ["frozen_dir", "working_dir", "report", "figures_dir"]}
    for section, keys in required.items():
        if section not in cfg:
            raise ValueError(f"Config {config_path} is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config {config_path} is missing '{section}.{key}'")
    if cfg["benchmark"]["evaluation_sets"] != ["validation"]:
        raise ValueError("benchmark.evaluation_sets must be [validation]: the test set is not part of Session 7")
    if cfg["execution"]["device"] != "cpu":
        raise ValueError("execution.device must be 'cpu' (frozen reference device)")
    if cfg["execution"]["reproducibility_gate"] != "exact_predictions":
        raise ValueError("execution.reproducibility_gate must be 'exact_predictions'")
    m = cfg["matrix"]
    if set(m["models"]) - set(FAMILY) or len(set(m["models"])) != len(m["models"]) or not m["models"]:
        raise ValueError(f"matrix.models must be unique entries of {tuple(FAMILY)}")
    if len(m["training_seeds"]) != 1 or m["training_seeds"][0] != cfg["benchmark"]["seed"]:
        raise ValueError("matrix.training_seeds must contain exactly benchmark.seed")
    if any(k >= m["full_budget"] or k < 1 for k in m["budgets"]):
        raise ValueError("matrix.budgets must be smaller than matrix.full_budget and positive")
    return cfg, text, config_path


def load_frozen_sources(cfg):
    """Definitions and subsets, each verified against the manifest of its session."""
    src = cfg["frozen_sources"]
    listed = {}
    for manifest_path in src["manifests"]:
        manifest = json.loads(resolve_path(manifest_path).read_text())
        folder = Path(manifest_path).parent
        for name, digest in manifest["files"].items():
            listed[str(folder / name)] = (digest, manifest_path)
    out = {"hashes": {}, "manifest_hashes": {p: sha256_file(resolve_path(p)) for p in src["manifests"]}}
    for key in ("selected_models", "subsets_topk", "subsets_randomk", *DEEP):
        rel = src[key]
        path = resolve_path(rel)
        digest = sha256_file(path)
        if rel not in listed:
            raise ValueError(f"{rel} is not listed in any frozen manifest")
        if listed[rel][0] != digest:
            raise ValueError(f"{rel} differs from its manifest {listed[rel][1]}")
        out[key] = json.loads(path.read_text())
        out["hashes"][key] = digest

    reference = None
    for key in DEEP:
        training = out[key]["training"]
        reference = reference or training
        if training != reference:
            raise ValueError(f"{key}: frozen training protocol differs from the other deep models")
    if reference["seed"] != cfg["benchmark"]["seed"] or reference["device"] != cfg["execution"]["device"]:
        raise ValueError("training seed or device differs from the frozen deep-learning protocol")
    out["training_protocol"] = reference
    return out


def load_train_validation(cfg):
    """Frozen Session 1 training and validation arrays, checked against the Session 1 manifest.
    The test file is never opened."""
    processed = resolve_path(cfg["session1"]["processed_dir"])
    manifest = json.loads((processed / "manifest.json").read_text())
    s1 = yaml.safe_load(resolve_path(cfg["session1"]["config"]).read_text())
    suffix = "y" if cfg["benchmark"]["target_type"] == "capped" else "y_raw"
    X, y, units, ends = {}, {}, {}, {}
    for split, prefix in (("train", "train"), ("validation", "val")):
        with np.load(processed / f"{prefix}_windows.npz") as z:
            arrays = {k: z[k] for k in z.files}
        for key, arr in arrays.items():
            if array_digest(arr) != manifest[key]:
                raise ValueError(f"{prefix}_windows.npz:{key} does not match the Session 1 manifest")
        X[split], y[split] = arrays[f"{prefix}_X"], arrays[f"{prefix}_{suffix}"]
        units[split], ends[split] = arrays[f"{prefix}_unit_ids"], arrays[f"{prefix}_end_cycles"]
    sensors = list(s1["features"]["selected"])
    if X["train"].shape[2] != len(sensors):
        raise ValueError("array width does not match the Session 1 sensor list")
    if set(units["train"].tolist()) & set(units["validation"].tolist()):
        raise ValueError("training and validation engines overlap")
    data = SequenceData(X=X, y=y, unit_ids=units, end_cycles=ends, sensors=sensors,
                        target_type=cfg["benchmark"]["target_type"], rul_cap=int(s1["rul"]["cap"]),
                        sequence_length=int(s1["windows"]["sequence_length"]))
    identity = {"manifest_sha256": sha256_file(processed / "manifest.json"),
                "split_sha256": sha256_file(processed / "split.json"),
                "config_sha256": sha256_file(resolve_path(cfg["session1"]["config"])),
                "shapes": {k: list(v.shape) for k, v in X.items()},
                "n_engines": {k: int(len(set(v.tolist()))) for k, v in units.items()}}
    return data, identity


def sensor_indices(sensors, canonical):
    """Column indices of `sensors`, which must be distinct canonical sensors in canonical order."""
    if len(set(sensors)) != len(sensors):
        raise ValueError(f"duplicate sensors in subset {sensors}")
    missing = [s for s in sensors if s not in canonical]
    if missing:
        raise ValueError(f"unknown sensors {missing}")
    indices = [canonical.index(s) for s in sensors]
    if indices != sorted(indices):
        raise ValueError(f"subset is not in the Session 1 sensor order: {sensors}")
    return indices


def subset_data(data, indices):
    """The same windows restricted to the given sensor columns (training and validation only).

    The result is C-contiguous like the frozen arrays. Column indexing alone yields another memory
    layout, which changes the order of NumPy reductions: the temporal features then differ in the
    last bits, enough to change the splits of the float64 gradient boosting."""
    if set(data.X) - {"train", "validation"}:
        raise ValueError("Session 7 data must not contain a test split")
    return SequenceData(X={k: np.ascontiguousarray(v[:, :, indices]) for k, v in data.X.items()}, y=data.y, unit_ids=data.unit_ids,
                        end_cycles=data.end_cycles, sensors=[data.sensors[i] for i in indices],
                        target_type=data.target_type, rul_cap=data.rul_cap, sequence_length=data.sequence_length)


def sensor_subsets(cfg, sources, canonical):
    """Every (budget, strategy, subset seed) of the matrix with its sensors in canonical order.
    The full set appears once, as strategy 'full'."""
    topk, randomk = sources["subsets_topk"], sources["subsets_randomk"]
    if topk["sensors"] != canonical or randomk["sensors"] != canonical:
        raise ValueError("subset artifacts were built for a different sensor list")
    m = cfg["matrix"]
    full = int(m["full_budget"])
    if full != len(canonical):
        raise ValueError("matrix.full_budget must equal the number of canonical sensors")
    for method in m["ranked_strategies"]:
        if sorted(topk["subsets"][method][f"top_{full}"]["sensors"]) != sorted(canonical):
            raise ValueError(f"{method} top_{full} is not the full sensor set")
    out = [{"budget": full, "strategy": "full", "subset_seed": None, "sensors": list(canonical),
            "memberships": [*m["ranked_strategies"], *[f"random:{s}" for s in m["random_subset_seeds"]]]}]
    for k in m["budgets"]:
        for method in m["ranked_strategies"]:
            out.append({"budget": k, "strategy": method, "subset_seed": None,
                        "sensors": list(topk["subsets"][method][f"top_{k}"]["sensors"]), "memberships": [method]})
        for seed in m["random_subset_seeds"]:
            out.append({"budget": k, "strategy": "random", "subset_seed": int(seed),
                        "sensors": list(randomk["subsets"][f"random_{k}"]["draws"][str(seed)]),
                        "memberships": [f"random:{seed}"]})
    for item in out:
        item["indices"] = sensor_indices(item["sensors"], canonical)
        if len(item["sensors"]) != item["budget"]:
            raise ValueError(f"subset for k={item['budget']} has {len(item['sensors'])} sensors")
    return out


def model_definition(model, sources):
    """The frozen scientific definition of a model (what is hashed into the condition identity)."""
    if model in CLASSICAL:
        d = sources["selected_models"]["models"][model]
        return {"model": model, "estimator": d["estimator"], "params": d["params"], "seed": d["seed"],
                "temporal_features": d["temporal_features"], "source_sha256": sources["hashes"]["selected_models"]}
    d = sources[model]
    arch = {k: v for k, v in d["architecture"].items() if k not in ("n_sensors", "n_parameters")}
    return {"model": model, "architecture": arch, "training": d["training"],
            "source_sha256": sources["hashes"][model]}


def build_plan(cfg, sources, data, data_identity, timing_cfg):
    """The predeclared matrix. Conditions with an identical sensor set are trained once and keep all
    their strategy memberships."""
    stages = {"full": 1, "correlation": 2, "permutation": 2, "random": 3}
    subsets = sensor_subsets(cfg, sources, data.sensors)
    conditions = []
    seed = int(cfg["matrix"]["training_seeds"][0])
    for model in cfg["matrix"]["models"]:
        definition = model_definition(model, sources)
        by_sensors = {}
        for item in subsets:
            key = tuple(item["indices"])
            if key in by_sensors:                       # identical sensor set: one training run
                existing = by_sensors[key]
                existing["memberships"] = sorted(set(existing["memberships"]) | set(item["memberships"]))
                continue
            identity = {"model_definition": definition, "sensor_indices": item["indices"],
                        "sensors": item["sensors"], "training_seed": seed, "data": data_identity,
                        "timing": {k: timing_cfg[k] for k in ("warmup_runs", "timed_runs", "batch_size", "timing_set")},
                        "evaluation_sets": cfg["benchmark"]["evaluation_sets"],
                        "target_type": cfg["benchmark"]["target_type"]}
            tag = item["strategy"] + (str(item["subset_seed"]) if item["subset_seed"] else "")
            condition = {
                "experiment_id": f"s7_{model}_k{item['budget']:02d}_{tag}_{sha256_json(identity)[:10]}",
                "model": model, "family": FAMILY[model], "budget": item["budget"], "strategy": item["strategy"],
                "subset_seed": item["subset_seed"], "training_seed": seed, "sensors": item["sensors"],
                "sensor_indices": item["indices"], "memberships": list(item["memberships"]),
                "definition_sha256": sha256_json(definition),
                "definition_source_sha256": definition["source_sha256"],
                "subset_source_sha256": sources["hashes"]["subsets_randomk" if item["strategy"] == "random"
                                                         else "subsets_topk"],
                "data_manifest_sha256": data_identity["manifest_sha256"],
                "evaluation_sets": list(cfg["benchmark"]["evaluation_sets"]),
                "planned_metrics": ["rmse", "mae", "r2", "nasa_score", "train_time_s",
                                    "inference_latency_ms_per_batch", "throughput_samples_per_s",
                                    "model_size_bytes", "n_parameters"],
                "stage": stages[item["strategy"]], "identity_sha256": sha256_json(identity)}
            conditions.append(condition)
            by_sensors[key] = condition
    order = {m: i for i, m in enumerate(cfg["matrix"]["models"])}
    conditions.sort(key=lambda c: (c["stage"], order[c["model"]], -c["budget"], c["strategy"], c["subset_seed"] or 0))
    ids = [c["experiment_id"] for c in conditions]
    if len(set(ids)) != len(ids):
        raise ValueError("experiment IDs are not unique")
    return {"schema_version": 1, "session": 7, "n_conditions": len(conditions),
            "per_model": {m: sum(c["model"] == m for c in conditions) for m in cfg["matrix"]["models"]},
            "training_protocol": sources["training_protocol"], "data": data_identity,
            "source_hashes": sources["hashes"], "manifest_hashes": sources["manifest_hashes"],
            "conditions": conditions}
