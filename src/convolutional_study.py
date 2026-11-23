"""Candidate grids, runs and validation-only selection for the CNN / TCN study.

Nothing in this module can see the official test windows: every function receives a SequenceData
object and uses only its training and validation splits. The test split is touched solely by
`run_experiment` in the entry point, after the final architecture has been selected.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from .convolutional_models import ConvolutionalRegressor
from .metrics import compute_metrics
from .recurrent_models import resolve_device
from .recurrent_study import _model_size_bytes
from .timing import measure_inference, time_training
from .tracking import mlflow


def candidate_grid(cfg, kind):
    """Declared candidates of one family in grid order (channels outermost)."""
    space = cfg["architecture"][kind]
    last = "depth" if kind == "cnn" else "dilations"
    out = []
    for channels, kernel, tail in itertools.product(space["channels"], space["kernel_size"], space[last]):
        arch = {"channels": int(channels), "kernel_size": int(kernel)}
        arch[last] = int(tail) if kind == "cnn" else [int(d) for d in tail]
        out.append(arch)
    return out


def arch_label(kind, arch):
    tail = f"d{arch['depth']}" if kind == "cnn" else "dil" + "-".join(str(d) for d in arch["dilations"])
    return f"{kind}_c{arch['channels']}_k{arch['kernel_size']}_{tail}"


def make_model(cfg, kind, arch, data, name=None, with_validation=True):
    """A ConvolutionalRegressor for `arch` on `data` (validation split attached for early stopping)."""
    return ConvolutionalRegressor(
        name or kind, kind, channels=arch["channels"], kernel_size=arch["kernel_size"],
        depth=arch.get("depth"), dilations=arch.get("dilations"), dropout=float(cfg["architecture"]["dropout"]),
        sequence_length=data.sequence_length, n_sensors=len(data.sensors), training=cfg["training"],
        seed=int(cfg["benchmark"]["seed"]), device=resolve_device(cfg["execution"]["device"]),
        validation=(data.X["validation"], data.y["validation"]) if with_validation else None,
        target_scale=float(cfg["training"]["target_scale"]))


def tiny_batch_gate(cfg, kind, arch, data, n=64, epochs=300, lr=0.01, min_drop=0.5):
    """Stability gate: the model must overfit a small fixed batch of training windows."""
    idx = np.random.default_rng(int(cfg["benchmark"]["seed"])).choice(len(data.X["train"]), n, replace=False)
    X, y = data.X["train"][idx], data.y["train"][idx]
    model = make_model(cfg, kind, arch, data, with_validation=False)
    model.validation = (X, y)
    model.training = {**cfg["training"], "learning_rate": lr, "max_epochs": epochs, "batch_size": n,
                      "early_stopping_patience": epochs, "scheduler": {"type": "none"}}
    model.fit(X, y)
    first, last = model.history_[0]["train_loss"], model.history_[-1]["train_loss"]
    pred = model.predict(X)
    return {"family": kind, "first_loss": first, "last_loss": last, "relative_drop": 1.0 - last / first,
            "predictions_finite": bool(np.isfinite(pred).all()), "stable": bool(model.fit_summary_["stable"]),
            "passed": bool(np.isfinite(last) and last <= first * (1.0 - min_drop) and np.isfinite(pred).all()
                           and model.fit_summary_["stable"])}


def run_candidate(cfg, timing_cfg, kind, arch, data, tracker, stage, environment, candidate, checkpoint_dir):
    """Train one candidate on the training split, score it on VALIDATION, log it to MLflow and keep
    its best checkpoint. Unstable runs are kept and flagged, never hidden."""
    name = arch_label(kind, arch)
    model = make_model(cfg, kind, arch, data, name=name)
    tags = {"stage": stage, "model": kind, "git_commit": str(environment["git"]["git_commit"]),
            "git_dirty": str(environment["git"]["git_dirty"])}
    with tracker.start_run(name, tags=tags) as run_id:
        train_time = time_training(model, data.X["train"], data.y["train"])
        s, h = model.fit_summary_, model.history_
        rec = {"stage": stage, "family": kind, "candidate": candidate, "run_name": name, "mlflow_run_id": run_id,
               **arch, "dilations_label": "-".join(map(str, arch["dilations"])) if "dilations" in arch else "",
               "sequence_length": data.sequence_length, "receptive_field": model.receptive_field,
               "train_time_s": train_time, "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
               "stop_reason": s["stop_reason"], "best_val_loss": s["best_val_loss"],
               "status": "ok" if s["stable"] else "unstable",
               "mean_epoch_time_s": float(np.mean([e["epoch_time_s"] for e in h])) if h else float("nan"),
               "max_grad_norm": float(np.nanmax([e["grad_norm_max"] for e in h])) if h else float("nan"),
               "mean_clip_fraction": float(np.nanmean([e["clip_fraction"] for e in h])) if h else float("nan"),
               "n_parameters": model.n_parameters(), "history": h}
        if s["stable"]:
            pred = model.predict(data.X["validation"])
            rec["predictions"] = pred
            rec["validation"] = compute_metrics(data.y["validation"], pred)
            batch = int(timing_cfg["batch_size"])
            timing = measure_inference(model.predict, data.X["validation"][:batch], int(timing_cfg["warmup_runs"]),
                                       int(timing_cfg["timed_runs"]))
            rec.update(model_size_bytes=_model_size_bytes(model), weights_sha256=model.weights_sha256(),
                       latency_ms_per_batch=timing["latency_ms_per_batch"],
                       latency_ms_per_sample=timing["latency_ms_per_sample"],
                       throughput_samples_per_s=timing["throughput_samples_per_s"])
            model.save(Path(checkpoint_dir) / name)
        else:
            rec["validation"] = {k: float("nan") for k in ("rmse", "mae", "r2", "nasa_score")}

        params = {"model": kind, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in arch.items()},
                  "dropout": cfg["architecture"]["dropout"], "sequence_length": data.sequence_length,
                  "receptive_field": model.receptive_field, "seed": cfg["benchmark"]["seed"],
                  "optimizer": cfg["training"]["optimizer"], "learning_rate": cfg["training"]["learning_rate"],
                  "batch_size": cfg["training"]["batch_size"], "grad_clip": cfg["training"]["grad_clip"],
                  "early_stopping_patience": cfg["training"]["early_stopping_patience"],
                  "max_epochs": cfg["training"]["max_epochs"], "device": model.device,
                  "target_type": data.target_type, "sensor_count": len(data.sensors),
                  "config_ref": "session5_convolutional", "git_commit": environment["git"]["git_commit"],
                  "git_dirty": environment["git"]["git_dirty"]}
        mlflow.log_params(params)
        metrics = {k: rec[k] for k in ("best_epoch", "epochs_run", "best_val_loss", "train_time_s", "n_parameters",
                                       "max_grad_norm", "mean_clip_fraction", "receptive_field")}
        metrics.update({f"validation_{k}": v for k, v in rec["validation"].items()})
        for key in ("model_size_bytes", "latency_ms_per_batch", "latency_ms_per_sample", "throughput_samples_per_s"):
            if key in rec:
                metrics[key] = rec[key]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(float(v))})
        mlflow.set_tag("status", rec["status"])
        mlflow.set_tag("stop_reason", rec["stop_reason"])
        for e in h:
            mlflow.log_metrics({k: e[k] for k in ("train_loss", "val_loss", "lr", "grad_norm_mean")
                                if np.isfinite(e[k])}, step=int(e["epoch"]))
        mlflow.log_dict(h, "training_history.json")
    return rec


def select_candidate(records, metric="rmse", tie_tolerance_rel=0.01):
    """Validation-only selection: lowest `metric`; candidates within the tolerance of the best are
    equivalent; among them fewest parameters, then lower inference latency, then grid order.
    Unstable runs are excluded. Returns (chosen record, rule description)."""
    usable = [r for r in records if r["status"] == "ok"]
    if not usable:
        raise ValueError("no stable candidates to select from")
    best = min(r["validation"][metric] for r in usable)
    equivalent = [r for r in usable if r["validation"][metric] <= best * (1.0 + tie_tolerance_rel)]
    chosen = min(equivalent, key=lambda r: (r["n_parameters"], r["latency_ms_per_batch"], r["candidate"]))
    same_params = [r for r in equivalent if r["n_parameters"] == chosen["n_parameters"]]
    level = "validation metric" if len(equivalent) == 1 else (
        "parameter count" if len(same_params) == 1 else "inference latency or grid order")
    rule = {"metric": metric, "best_validation_value": best, "tie_tolerance_rel": tie_tolerance_rel,
            "n_candidates": len(usable), "n_equivalent": len(equivalent),
            "tiebreak": "fewest parameters, then lower inference latency, then grid order",
            "decided_by": level}
    return chosen, rule


def load_historical_representatives(results_json, models=("random_forest", "gradient_boosting", "lstm", "gru")):
    """Frozen validation (and test) rows of the earlier families, read from the results table.

    Returns {model: {validation: row, test: row}} for the canonical rows (git_dirty False). Nothing
    is written back to the table.
    """
    path = Path(results_json)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())["results"]
    out = {}
    for model in models:
        mine = [r for r in rows if r["model"] == model and str(r.get("git_dirty")).lower() == "false"]
        by_set = {}
        for r in mine:
            by_set[r["eval_set"]] = r            # later rows of the same model win
        if "validation" in by_set:
            out[model] = by_set
    return out


def threshold_near_miss(records, chosen, rule):
    """Closest stable candidate that lies just OUTSIDE the equivalence threshold and has fewer parameters
    than the chosen one (the case in which the exact threshold would change the tie-break). The comparison
    uses the unrounded validation values. Returns None when there is no such candidate."""
    metric, threshold = rule["metric"], rule["best_validation_value"] * (1.0 + rule["tie_tolerance_rel"])
    outside = [r for r in records if r["status"] == "ok" and r["validation"][metric] > threshold
               and r["n_parameters"] < chosen["n_parameters"]]
    if not outside:
        return None
    r = min(outside, key=lambda r: r["validation"][metric])
    return {"run_name": r["run_name"], "n_parameters": r["n_parameters"], "value": r["validation"][metric],
            "threshold": threshold, "margin": r["validation"][metric] - threshold}
