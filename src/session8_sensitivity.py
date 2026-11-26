"""Training-cap sensitivity: a small, separate study. The seven declared configurations are refitted
with other RUL caps (and once at 125 as a control that must reproduce the canonical predictions).
Scored on FD001 validation only. No FD003 data is used, no cap is selected, and no canonical file,
record or run is touched."""
import copy
import json
import time

import numpy as np

from .benchmark_config import resolve_path
from .evaluation import predictions_digest
from .metrics import compute_metrics
from .sequence_data import SequenceData
from .session7_execution import atomic_write_json
from .session7_integrity import git_state
from .session7_models import build_condition_model
from .session7_plan import DEEP, load_session7_config, load_train_validation, sha256_json, subset_data
from .session8_evaluate import candidate_members
from .session8_guard import require_authorization, require_new_location


def with_cap(data, raw, cap):
    """The same windows with targets min(raw RUL, cap)."""
    y = {k: np.minimum(raw[k], cap).astype("float32") for k in data.y}
    return SequenceData(X=data.X, y=y, unit_ids=data.unit_ids, end_cycles=data.end_cycles, sensors=data.sensors,
                        target_type=data.target_type, rul_cap=int(cap), sequence_length=data.sequence_length)


def sources_with_cap(sources, cap):
    """Frozen definitions with only the deep models' target scale following the cap."""
    out = copy.deepcopy(sources)
    for model in DEEP:
        out[model]["training"]["target_scale"] = type(out[model]["training"]["target_scale"])(cap)
    return out


def raw_targets(processed_dir):
    out = {}
    for split, prefix in (("train", "train"), ("validation", "val")):
        with np.load(resolve_path(processed_dir) / f"{prefix}_windows.npz") as z:
            out[split] = z[f"{prefix}_y_raw"].astype("float64")
    return out


def require_control(key, record, canonical_record):
    if record["predictions_sha256"] != canonical_record["predictions_sha256"]:
        raise RuntimeError(f"{key}: the control refit does not reproduce the canonical predictions; the "
                           "sensitivity study is not comparable")


def run_fits(cfg, canonical, say=print, tracker=None, *, authorization=None):
    """Fit (or reuse) every declared configuration at every cap. Returns the records. Training: needs
    authorization and a location without a completed execution, also when resuming."""
    require_authorization(authorization, "training the cap-sensitivity refits")
    require_new_location(cfg)
    s7, _, _ = load_session7_config(resolve_path(cfg["upstream"]["session7_config"]))
    data, identity = load_train_validation(s7)
    raw = raw_targets(cfg["upstream"]["processed_dir"])
    members = candidate_members(cfg, canonical.conditions)
    sens = cfg["sensitivity"]
    folder = resolve_path(cfg["outputs"]["working_dir"]) / "sensitivity"
    caps = [sens["control_cap"], *sens["caps"]]
    out = []
    for cap in caps:
        capped, sources = with_cap(data, raw, cap), sources_with_cap(canonical.sources, cap)
        for key in sens["configurations"]:
            condition = canonical.conditions[members[key][0]]
            fit_identity = sha256_json({"condition": condition["identity_sha256"], "cap": cap, "data": identity,
                                        "execution": s7["execution"]})
            path = folder / f"{key}_cap{cap}.json"
            record = json.loads(path.read_text()) if path.exists() else None
            if record is not None and record["identity"] != fit_identity:
                raise RuntimeError(f"{path} belongs to another fit; it is not overwritten")
            if record is not None:
                with np.load(path.with_suffix(".npz")) as z:
                    if predictions_digest(z["prediction"]) != record["predictions_sha256"]:
                        raise RuntimeError(f"{path.with_suffix('.npz')} does not match its record")
            else:
                subset = subset_data(capped, condition["sensor_indices"])
                model = build_condition_model(condition["model"], sources, subset, execution=s7["execution"],
                                              device="cpu")
                start = time.perf_counter()
                model.fit(subset.X["train"], subset.y["train"])
                seconds = time.perf_counter() - start
                prediction = np.asarray(model.predict(subset.X["validation"]), dtype=np.float64)
                folder.mkdir(parents=True, exist_ok=True)
                model.save(folder / "models" / f"{key}_cap{cap}")
                np.savez(path.with_suffix(".npz"), prediction=prediction)
                record = {"identity": fit_identity, "candidate": key, "cap": cap, "model": condition["model"],
                          "experiment_id_template": condition["experiment_id"], "train_time_s": seconds,
                          "predictions_sha256": predictions_digest(prediction), "training_seed": condition["training_seed"],
                          "own_cap_metrics": compute_metrics(subset.y["validation"], prediction),
                          "git": git_state(resolve_path("."))}
                if cap == sens["control_cap"]:
                    require_control(key, record, canonical.records[condition["experiment_id"]])
                if tracker is not None:
                    record["mlflow_run_id"] = tracker(key, cap, record)
                atomic_write_json(path, record)
                say(f"  cap {cap:>3} {key:<18} fitted in {seconds:.0f}s")
            if cap == sens["control_cap"]:                     # checked for reused records as well
                require_control(key, record, canonical.records[condition["experiment_id"]])
            out.append(record)
    return out, data, raw


def sensitivity_rows(cfg, canonical, records, data, raw):
    """One row per configuration and cap, scored on identical targets where possible."""
    folder = resolve_path(cfg["outputs"]["working_dir"]) / "sensitivity"
    limit = cfg["sensitivity"]["common_subset_max_raw_rul"]
    if limit > min(cfg["sensitivity"]["caps"]):
        raise ValueError("the common subset must lie below every training cap")
    common = raw["validation"] <= limit
    rows = []
    for r in records:
        with np.load(folder / f"{r['candidate']}_cap{r['cap']}.npz") as z:
            prediction = z["prediction"]
        m = compute_metrics(raw["validation"][common], prediction[common])
        d = prediction[common] - raw["validation"][common]
        rows.append({"candidate": r["candidate"], "model": r["model"], "training_cap": r["cap"],
                     "common_subset_rmse": m["rmse"], "common_subset_mae": m["mae"],
                     "common_subset_mean_signed": float(d.mean()), "common_subset_nasa": m["nasa_score"],
                     "n_common_windows": int(common.sum()), "own_cap_rmse": r["own_cap_metrics"]["rmse"],
                     "control_reproduces_canonical":           # compared, and only for the control cap
                     r["predictions_sha256"] == canonical.records[r["experiment_id_template"]]["predictions_sha256"]
                     if r["cap"] == cfg["sensitivity"]["control_cap"] else ""})
    order = {}
    for cap in sorted({r["training_cap"] for r in rows}):
        group = sorted((r for r in rows if r["training_cap"] == cap and r["candidate"].endswith("_full")),
                       key=lambda r: r["common_subset_rmse"])
        for rank, r in enumerate(group, 1):
            order[cap, r["candidate"]] = rank
    for r in rows:
        r["rank_among_full_models"] = order.get((r["training_cap"], r["candidate"]), "")
    return rows
