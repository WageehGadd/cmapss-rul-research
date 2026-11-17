"""Model-agnostic evaluation: the single place where benchmark numbers are produced.

`evaluate_model` works with any object that follows the interface documented in
`src/baselines.py` (fit / predict / get_config / save), so later model families reuse it
unchanged. Training itself stays outside: the caller measures it with
`timing.time_training` and passes the result in.
"""
import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import compute_metrics
from .timing import measure_inference


def predictions_digest(pred):
    return hashlib.sha256(np.ascontiguousarray(pred, dtype=np.float64).tobytes()).hexdigest()


@dataclass
class EvaluationResult:
    run_id: str
    model_name: str
    target_type: str
    predictions: dict            # eval set -> float64 [n]
    metrics: dict                # eval set -> {rmse, mae, r2, nasa_score}
    prediction_digests: dict     # eval set -> sha256 of the predictions
    n_samples: dict              # eval set -> number of samples
    train_time_s: float
    timing: dict                 # output of timing.measure_inference
    model_size_bytes: int
    n_parameters: object         # int, or None when the model does not report it
    artifact_files: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def evaluate_model(model, eval_sets, timing_inputs, *, target_type, train_time_s, timing_config,
                   artifact_dir, run_id=None, metadata=None):
    """Score a fitted model and measure its inference cost.

    eval_sets     {set name: (X windows, y true)}, y in the benchmark target semantics.
    timing_inputs windows used to build the timing batch (first `batch_size` samples).
    train_time_s  wall-clock training time measured by the caller.
    timing_config {warmup_runs, timed_runs, batch_size}.
    artifact_dir  where the model is saved; the size of the saved files is the model size.
    run_id        unique run identifier; a UUID is generated when none is given.
    """
    predictions, metrics, digests, counts = {}, {}, {}, {}
    for name, (X, y_true) in eval_sets.items():
        pred = np.asarray(model.predict(X), dtype=np.float64)
        if pred.shape != (len(X),):
            raise ValueError(f"{model.name}: predict returned shape {pred.shape} for {len(X)} samples "
                             f"in set '{name}', expected ({len(X)},)")
        predictions[name] = pred
        metrics[name] = compute_metrics(y_true, pred)
        digests[name] = predictions_digest(pred)
        counts[name] = int(len(X))

    batch_size = int(timing_config["batch_size"])
    if len(timing_inputs) < batch_size:
        raise ValueError(f"timing set has {len(timing_inputs)} samples, fewer than batch_size={batch_size}")
    timing = measure_inference(model.predict, timing_inputs[:batch_size],
                               int(timing_config["warmup_runs"]), int(timing_config["timed_runs"]))

    files = [str(p) for p in model.save(artifact_dir)]
    size = sum(Path(p).stat().st_size for p in files)
    n_parameters = model.n_parameters() if hasattr(model, "n_parameters") else None
    return EvaluationResult(
        run_id=run_id or uuid.uuid4().hex, model_name=model.name, target_type=target_type,
        predictions=predictions, metrics=metrics, prediction_digests=digests, n_samples=counts,
        train_time_s=float(train_time_s), timing=timing, model_size_bytes=int(size),
        n_parameters=n_parameters, artifact_files=files, metadata=dict(metadata or {}))
