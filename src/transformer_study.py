"""Stability gate, candidate runs and validation-only selection for the Transformer study.

Nothing in this module can see the official test windows: every function receives a SequenceData
object and uses only its training and validation splits. The test split is touched solely by
`run_experiment` in the entry point, after the final architecture has been selected.
Selection reuses the Session 5 rule (`convolutional_study.select_candidate`).
"""
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .metrics import compute_metrics
from .recurrent_models import resolve_device
from .recurrent_study import _model_size_bytes
from .timing import measure_inference, time_training
from .tracking import mlflow
from .transformer_models import TransformerRegressor


def stage_a_grid(cfg):
    """Declared Stage A candidates in grid order (d_model outermost)."""
    a = cfg["study"]["stage_a"]
    return [{"d_model": d, "num_heads": h, "num_layers": n, "pooling": a["pooling"]}
            for d, h, n in itertools.product(a["d_model"], a["num_heads"], a["num_layers"])]


def arch_label(arch):
    return f"tf_d{arch['d_model']}_h{arch['num_heads']}_l{arch['num_layers']}_{arch['pooling']}"


def make_model(cfg, arch, data, name=None, with_validation=True):
    model = cfg["model"]
    return TransformerRegressor(
        name or "transformer", d_model=arch["d_model"], num_heads=arch["num_heads"], num_layers=arch["num_layers"],
        pooling=arch["pooling"], ff_multiplier=model["ff_multiplier"], dropout=model["dropout"],
        activation=model["activation"], sequence_length=data.sequence_length, n_sensors=len(data.sensors),
        training=cfg["training"], seed=int(cfg["benchmark"]["seed"]), device=resolve_device(cfg["execution"]["device"]),
        validation=(data.X["validation"], data.y["validation"]) if with_validation else None,
        target_scale=float(cfg["training"]["target_scale"]))


def stability_gate(cfg, arch, data):
    """Checks that must all pass before the architecture study starts. Uses training and validation
    windows only. Returns a dict with every check and `passed`."""
    gate, seed = cfg["stability_gate"], int(cfg["benchmark"]["seed"])
    checks, scale = {}, float(cfg["training"]["target_scale"])
    n = int(gate["samples"])
    idx = np.random.default_rng(seed).choice(len(data.X["train"]), n, replace=False)
    X, y = data.X["train"][idx], data.y["train"][idx]

    model = make_model(cfg, arch, data, with_validation=False)
    net = model._build()
    xb = torch.from_numpy(X)
    out = net.train()(xb.to(model.device))
    checks["forward_shape_is_batch"] = tuple(out.shape) == (n,)
    loss = torch.nn.functional.mse_loss(out, torch.from_numpy(y / scale).to(model.device))
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    checks["loss_and_gradients_finite"] = bool(torch.isfinite(loss)) and all(
        g is not None and bool(torch.isfinite(g).all()) for g in grads)

    # tiny-batch training (training loss must fall) with the shared inherited loop
    model.validation = (X, y)
    model.training = {**cfg["training"], "learning_rate": gate["learning_rate"], "max_epochs": gate["epochs"],
                      "batch_size": n, "early_stopping_patience": gate["epochs"], "scheduler": {"type": "none"}}
    model.fit(X, y)
    first, last = model.history_[0]["train_loss"], model.history_[-1]["train_loss"]
    pred = model.predict(X)
    checks["tiny_batch_loss_decreases"] = bool(np.isfinite(last) and last <= first * (1.0 - float(gate["min_loss_drop"])))
    checks["predictions_finite"] = bool(np.isfinite(pred).all())
    # target scaling: predictions come back in cycles, i.e. network output x cap
    with torch.no_grad():
        raw = model.net_.eval()(xb.to(model.device)).cpu().numpy()
    checks["target_scaling_inverted_in_predict"] = bool(np.allclose(pred, raw.astype(np.float64) * scale))
    checks["predictions_in_cycles"] = bool(abs(float(pred.mean()) - float(y.mean())) < 0.25 * scale)

    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp))
        reloaded = TransformerRegressor.load(Path(tmp), device=model.device)
        checks["save_load_identical_predictions"] = bool(np.array_equal(reloaded.predict(X), pred))

    a, b = make_model(cfg, arch, data)._build(), make_model(cfg, arch, data)._build()
    checks["deterministic_initialisation"] = all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    short = {**cfg["training"], "max_epochs": 2, "batch_size": 16}
    digests = []
    for _ in range(2):
        m = make_model(cfg, arch, data, with_validation=False)
        m.validation, m.training = (X, y), short
        digests.append(m.fit(X, y).weights_sha256())
    checks["deterministic_training"] = digests[0] == digests[1]
    checks["device_as_configured"] = model.device == resolve_device(cfg["execution"]["device"])

    tr_units, va_units = set(data.unit_ids["train"].tolist()), set(data.unit_ids["validation"].tolist())
    checks["train_and_validation_engines_disjoint"] = not (tr_units & va_units)
    return {"pooling": arch["pooling"], "candidate": arch_label(arch), "first_loss": first, "last_loss": last,
            "relative_drop": 1.0 - last / first, "checks": checks, "passed": all(checks.values())}


def run_candidate(cfg, timing_cfg, arch, data, tracker, stage, environment, candidate, checkpoint_dir):
    """Train one candidate on the training split, score it on VALIDATION, log it to MLflow and keep
    its best checkpoint. Unstable runs are kept and flagged, never hidden."""
    name = arch_label(arch)
    model = make_model(cfg, arch, data, name=name)
    tags = {"stage": stage, "model": "transformer", "git_commit": str(environment["git"]["git_commit"]),
            "git_dirty": str(environment["git"]["git_dirty"])}
    with tracker.start_run(name, tags=tags) as run_id:
        train_time = time_training(model, data.X["train"], data.y["train"])
        s, h = model.fit_summary_, model.history_
        best = h[s["best_epoch"] - 1] if s["best_epoch"] else {}
        pred = model.predict(data.X["validation"]) if s["stable"] else None
        status = "ok" if s["stable"] and np.isfinite(pred).all() else (
            "non_finite_predictions" if s["stable"] else "unstable")
        rec = {"stage": stage, "family": "transformer", "candidate": candidate, "run_name": name,
               "mlflow_run_id": run_id, **arch, "ff_dim": model.ff_multiplier * model.d_model,
               "dropout": model.dropout, "sequence_length": data.sequence_length, "n_parameters": model.n_parameters(),
               "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "stop_reason": s["stop_reason"],
               "best_val_loss": s["best_val_loss"], "train_loss_at_best": best.get("train_loss", float("nan")),
               "train_time_s": train_time, "status": status,
               "mean_epoch_time_s": float(np.mean([e["epoch_time_s"] for e in h])) if h else float("nan"),
               "max_grad_norm": float(np.nanmax([e["grad_norm_max"] for e in h])) if h else float("nan"),
               "mean_clip_fraction": float(np.nanmean([e["clip_fraction"] for e in h])) if h else float("nan"),
               "history": h}
        if status == "ok":
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

        mlflow.log_params({
            "model": "transformer", **model.architecture(), "seed": cfg["benchmark"]["seed"],
            "optimizer": cfg["training"]["optimizer"], "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["batch_size"], "grad_clip": cfg["training"]["grad_clip"],
            "early_stopping_patience": cfg["training"]["early_stopping_patience"],
            "max_epochs": cfg["training"]["max_epochs"], "device": model.device, "target_type": data.target_type,
            "sensor_count": len(data.sensors), "config_ref": "session6_transformer",
            "git_commit": environment["git"]["git_commit"], "git_dirty": environment["git"]["git_dirty"]})
        metrics = {k: rec[k] for k in ("best_epoch", "epochs_run", "best_val_loss", "train_loss_at_best",
                                       "train_time_s", "n_parameters", "max_grad_norm", "mean_clip_fraction")}
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
        if "weights_sha256" in rec:
            mlflow.set_tag("weights_sha256", rec["weights_sha256"])
            mlflow.log_dict(json.loads(json.dumps(model.get_config())), "model_config.json")
    return rec
