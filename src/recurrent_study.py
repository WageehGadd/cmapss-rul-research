"""Stability gate, architecture ablation and validation-only selection for LSTM / GRU.

Nothing in this module can see the official test windows: every function receives the training
and validation arrays of a SequenceData object and uses `data.X['train']`, `data.X['validation']`
(and their targets) only. The test split is touched solely by `run_experiment` in the entry point,
after the architectures have been selected.
"""
import itertools
import tempfile
from pathlib import Path

import numpy as np

from .metrics import compute_metrics
from .recurrent_models import RecurrentRegressor, resolve_device, seed_everything
from .timing import measure_inference, time_training
from .tracking import mlflow
from .tuning import select

GRID_KEYS = ("hidden_size", "num_layers", "dropout")


def make_model(cfg, family, arch, data, name=None):
    """A RecurrentRegressor for `arch` ({hidden_size, num_layers, dropout}) on `data`.

    The validation split of `data` is attached for early stopping and checkpoint selection.
    """
    return RecurrentRegressor(
        name or family, family, hidden_size=arch["hidden_size"], num_layers=arch["num_layers"],
        dropout=arch["dropout"], sequence_length=data.sequence_length, n_sensors=len(data.sensors),
        training=cfg["training"], seed=int(cfg["benchmark"]["seed"]),
        device=resolve_device(cfg["execution"]["device"]),
        validation=(data.X["validation"], data.y["validation"]), target_scale=float(cfg["training"]["target_scale"]))


def stage_a_grid(cfg):
    a = cfg["ablation"]["stage_a"]
    return [dict(zip(GRID_KEYS, values)) for values in
            itertools.product(a["hidden_size"], a["num_layers"], a["dropout"])]


# ---------------------------------------------------------------------------------------------
# reference stability gate
# ---------------------------------------------------------------------------------------------
def tiny_batch_overfit(cfg, data):
    """A small fixed batch must be fitted almost perfectly: the training loss has to fall."""
    ref = cfg["reference"]
    n = int(ref["tiny_batch_samples"])
    idx = np.random.default_rng(int(cfg["benchmark"]["seed"])).choice(len(data.X["train"]), n, replace=False)
    X, y = data.X["train"][idx], data.y["train"][idx]
    training = {**cfg["training"], "learning_rate": ref["tiny_batch_learning_rate"],
                "max_epochs": ref["tiny_batch_epochs"], "batch_size": n,
                "early_stopping_patience": ref["tiny_batch_epochs"], "scheduler": {"type": "none"}}
    model = RecurrentRegressor("tiny", ref["family"], hidden_size=ref["hidden_size"],
                               num_layers=ref["num_layers"], dropout=0.0,
                               sequence_length=data.sequence_length, n_sensors=len(data.sensors),
                               training=training, seed=int(cfg["benchmark"]["seed"]),
                               device=resolve_device(cfg["execution"]["device"]), validation=(X, y),
                               target_scale=float(cfg["training"]["target_scale"]))
    model.fit(X, y)
    first, last = model.history_[0]["train_loss"], model.history_[-1]["train_loss"]
    return {"first_loss": first, "last_loss": last, "relative_drop": 1.0 - last / first,
            "passed": bool(np.isfinite(last) and last <= first * (1.0 - float(ref["min_loss_drop"])))}


def reference_run(cfg, data):
    """Train the reference model and check the stability criteria. Returns a diagnostics dict
    that includes the checks and `passed` (the ablation must not start unless it is True)."""
    ref = cfg["reference"]
    arch = {k: ref[k] for k in GRID_KEYS}
    model = make_model(cfg, ref["family"], arch, data, name="reference")
    model.fit(data.X["train"], data.y["train"])
    pred = model.predict(data.X["validation"])
    h = model.history_
    norms = [e["grad_norm_max"] for e in h]
    checks = {
        "training_ran_without_non_finite_values": bool(model.fit_summary_["stable"]),
        "training_loss_decreased": bool(h[-1]["train_loss"] < h[0]["train_loss"]),
        "validation_loss_decreased": bool(model.fit_summary_["best_val_loss"] < h[0]["val_loss"]),
        "validation_loss_finite": bool(np.isfinite(model.fit_summary_["best_val_loss"])),
        "predictions_finite": bool(np.isfinite(pred).all()),
        "predictions_in_sensible_range": bool(pred.min() > -50.0 and pred.max() < 175.0),
        "gradient_norm_bounded": bool(np.isfinite(norms).all() and max(norms) < 1.0e3),
    }
    tiny = tiny_batch_overfit(cfg, data)
    checks["tiny_batch_overfit"] = tiny["passed"]
    return {"family": ref["family"], "architecture": arch, "epochs_run": model.fit_summary_["epochs_run"],
            "best_epoch": model.fit_summary_["best_epoch"], "stop_reason": model.fit_summary_["stop_reason"],
            "first_train_loss": h[0]["train_loss"], "last_train_loss": h[-1]["train_loss"],
            "first_val_loss": h[0]["val_loss"], "best_val_loss": model.fit_summary_["best_val_loss"],
            "prediction_min": float(pred.min()), "prediction_max": float(pred.max()),
            "max_grad_norm": float(max(norms)), "mean_clip_fraction": float(np.mean([e["clip_fraction"] for e in h])),
            "validation_metrics": compute_metrics(data.y["validation"], pred),
            "tiny_batch": tiny, "checks": checks, "passed": all(checks.values()),
            "history": h}


# ---------------------------------------------------------------------------------------------
# ablation runs
# ---------------------------------------------------------------------------------------------
def _model_size_bytes(model):
    with tempfile.TemporaryDirectory() as tmp:
        return int(sum(Path(p).stat().st_size for p in model.save(Path(tmp))))


def run_candidate(cfg, timing_cfg, family, arch, data, tracker, stage, environment, tags=None):
    """Train one architecture on the training split, score it on VALIDATION, log it to MLflow.

    Failed or unstable runs are kept and flagged (status != 'ok'); they are logged, not hidden.
    """
    name = f"{family}_h{arch['hidden_size']}_l{arch['num_layers']}_d{arch['dropout']}_L{data.sequence_length}"
    model = make_model(cfg, family, arch, data, name=name)
    run_tags = {"stage": stage, "model": family, "git_commit": str(environment["git"]["git_commit"]),
                "git_dirty": str(environment["git"]["git_dirty"]), **(tags or {})}
    with tracker.start_run(name, tags=run_tags) as run_id:
        train_time = time_training(model, data.X["train"], data.y["train"])
        summary = model.fit_summary_
        record = {"stage": stage, "family": family, **arch, "sequence_length": data.sequence_length,
                  "run_name": name, "mlflow_run_id": run_id, "train_time_s": train_time,
                  "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
                  "stop_reason": summary["stop_reason"], "best_val_loss": summary["best_val_loss"],
                  "status": "ok" if summary["stable"] else "unstable",
                  "mean_epoch_time_s": float(np.mean([e["epoch_time_s"] for e in model.history_])),
                  "max_grad_norm": float(np.nanmax([e["grad_norm_max"] for e in model.history_])) if model.history_ else float("nan"),
                  "mean_clip_fraction": float(np.nanmean([e["clip_fraction"] for e in model.history_])) if model.history_ else float("nan")}
        if summary["stable"]:
            pred = model.predict(data.X["validation"])
            record["predictions"] = pred
            record["validation"] = compute_metrics(data.y["validation"], pred)
            batch = int(timing_cfg["batch_size"])
            timing = measure_inference(model.predict, data.X["validation"][:batch],
                                       int(timing_cfg["warmup_runs"]), int(timing_cfg["timed_runs"]))
            record.update(n_parameters=model.n_parameters(), model_size_bytes=_model_size_bytes(model),
                          latency_ms_per_batch=timing["latency_ms_per_batch"],
                          throughput_samples_per_s=timing["throughput_samples_per_s"],
                          weights_sha256=model.weights_sha256())
        else:
            record["validation"] = {k: float("nan") for k in ("rmse", "mae", "r2", "nasa_score")}
            record["n_parameters"] = model.n_parameters()
        record["history"] = model.history_

        mlflow.log_params({"model": family, **arch, "sequence_length": data.sequence_length,
                           "seed": cfg["benchmark"]["seed"], "optimizer": cfg["training"]["optimizer"],
                           "learning_rate": cfg["training"]["learning_rate"],
                           "batch_size": cfg["training"]["batch_size"], "grad_clip": cfg["training"]["grad_clip"],
                           "early_stopping_patience": cfg["training"]["early_stopping_patience"],
                           "max_epochs": cfg["training"]["max_epochs"], "device": model.device,
                           "target_type": data.target_type, "sensor_count": len(data.sensors),
                           "config_ref": "session4_recurrent", "git_commit": environment["git"]["git_commit"],
                           "git_dirty": environment["git"]["git_dirty"]})
        metrics = {"best_epoch": record["best_epoch"], "epochs_run": record["epochs_run"],
                   "best_val_loss": record["best_val_loss"], "train_time_s": train_time,
                   "n_parameters": record["n_parameters"], "max_grad_norm": record["max_grad_norm"],
                   "mean_clip_fraction": record["mean_clip_fraction"]}
        metrics.update({f"validation_{k}": v for k, v in record["validation"].items()})
        for key in ("model_size_bytes", "latency_ms_per_batch", "throughput_samples_per_s"):
            if key in record:
                metrics[key] = record[key]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(float(v))})
        mlflow.set_tag("status", record["status"])
        mlflow.set_tag("stop_reason", record["stop_reason"])
        for e in model.history_:
            step = int(e["epoch"])
            mlflow.log_metrics({k: e[k] for k in ("train_loss", "val_loss", "lr", "grad_norm_mean") if np.isfinite(e[k])},
                               step=step)
        mlflow.log_dict(model.history_, "training_history.json")
    return record


def stage_a(cfg, timing_cfg, data, tracker, environment, log=print):
    records = {}
    for family in cfg["ablation"]["families"]:
        records[family] = []
        for index, arch in enumerate(stage_a_grid(cfg)):
            rec = run_candidate(cfg, timing_cfg, family, arch, data, tracker, "ablation_L%d" % data.sequence_length,
                                environment)
            rec["candidate"] = index
            records[family].append(rec)
            log(f"  {rec['run_name']:<28} val RMSE {rec['validation']['rmse']:8.3f}  best epoch {rec['best_epoch']:>2}/"
                f"{rec['epochs_run']:<2} params {rec['n_parameters']:>7}  [{rec['status']}]")
    return records


def select_stage_a(cfg, family_records):
    """Validation-only selection per family; unstable runs are excluded from selection."""
    sel = cfg["selection"]
    out = {}
    for family, records in family_records.items():
        usable = [r for r in records if r["status"] == "ok"]
        chosen, rule = select(usable, sel["metric"], float(sel["tie_tolerance_rel"]))
        out[family] = (chosen, rule)
    return out


def common_endpoint_metrics(record, data, min_endpoint_cycle):
    """Validation metrics on windows whose final cycle is >= min_endpoint_cycle (comparable across L)."""
    mask = data.end_cycles["validation"] >= min_endpoint_cycle
    return compute_metrics(data.y["validation"][mask], record["predictions"][mask]), mask


def select_sequence_length(cfg, family, candidates, datasets, min_endpoint_cycle):
    """Compare one architecture trained at different window lengths on COMMON validation endpoints.

    candidates: records (same architecture) ordered by ascending sequence length.
    Returns (chosen record, rule dict, rows) where rows hold the common-endpoint metrics.
    """
    sel = cfg["selection"]
    ref_ends, rows, scored = None, [], []
    for rec in candidates:
        data = datasets[rec["sequence_length"]]
        metrics, mask = common_endpoint_metrics(rec, data, min_endpoint_cycle)
        keyed = (data.unit_ids["validation"][mask], data.end_cycles["validation"][mask])
        if ref_ends is None:
            ref_ends = keyed
        elif not (np.array_equal(keyed[0], ref_ends[0]) and np.array_equal(keyed[1], ref_ends[1])):
            raise ValueError("common-endpoint windows differ between sequence lengths")
        scored.append({"candidate": len(scored), "validation": metrics, "n_parameters": rec["n_parameters"],
                       "record": rec})
        rows.append({"sequence_length": rec["sequence_length"], "n_common_windows": int(mask.sum()), **metrics})
    chosen, rule = select(scored, sel["metric"], float(sel["tie_tolerance_rel"]))
    return chosen["record"], rule, rows
