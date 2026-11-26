"""Inference-only application of the 174 canonical Session 7 models to the Session 8 datasets.

Nothing is trained. Every model file is checked against its canonical record before loading, and a
model is only used on new data after it has reproduced its stored FD001 validation predictions.
Predictions are stored once, with the identity of what produced them, and reused only when that
identity and the stored digest still match.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .benchmark_config import resolve_path
from .classical_models import build_model
from .convolutional_models import ConvolutionalRegressor
from .evaluation import predictions_digest
from .recurrent_models import RecurrentRegressor
from .session7_canonical import canonical_runs
from .session7_execution import atomic_write_json
from .session7_plan import CLASSICAL, load_frozen_sources, load_session7_config, sha256_file, sha256_json
from .session8_data import windows_digest
from .session8_guard import require_authorization, require_new_location
from .transformer_models import TransformerRegressor

DEEP_CLASSES = {"lstm": RecurrentRegressor, "gru": RecurrentRegressor, "tcn": ConvolutionalRegressor,
                "transformer": TransformerRegressor}


@dataclass(frozen=True)
class Canonical:
    conditions: dict      # experiment_id -> plan condition
    records: dict         # experiment_id -> canonical condition record
    run_dirs: dict
    sources: dict         # frozen definitions of Sessions 3-6


def load_canonical(cfg):
    """The canonical Session 7 evidence, fully validated by the Session 7 loader (read only)."""
    up = cfg["upstream"]
    manifest_path = resolve_path(up["session7_manifest"])
    working = resolve_path(up["session7_working_dir"])
    runs = canonical_runs(manifest_path, up["mlflow_dir"], working)
    plan = json.loads((manifest_path.parent / "experiment_plan.json").read_text())
    records = {k: json.loads((working / "conditions" / f"{k}.json").read_text()) for k in runs}
    s7, _, _ = load_session7_config(resolve_path(up["session7_config"]))
    return Canonical({c["experiment_id"]: c for c in plan["conditions"]}, records,
                     {k: working / "runs" / v for k, v in runs.items()}, load_frozen_sources(s7))


def load_model(condition, run_dir, record, sources, *, authorization=None):
    """The fitted canonical model of one condition, after checking every referenced file."""
    require_authorization(authorization, "loading a model for inference")
    for name, digest in record["files"].items():
        if sha256_file(Path(run_dir) / name) != digest:
            raise ValueError(f"{condition['experiment_id']}: {name} differs from the canonical record")
    folder, family = Path(run_dir) / "model", condition["model"]
    if family in CLASSICAL:
        d = sources["selected_models"]["models"][family]
        model = build_model(family, d["params"], sensors=condition["sensors"], features=d["temporal_features"],
                            sequence_length=30, seed=d["seed"], name=family)
        names = json.loads((folder / "feature_names.json").read_text())
        if names != model.transformer.feature_names:
            raise ValueError(f"{condition['experiment_id']}: feature names differ from the sensor subset")
        model.estimator_ = joblib.load(folder / f"{family}.joblib")
        if model.estimator_.n_features_in_ != len(names):
            raise ValueError(f"{condition['experiment_id']}: estimator input width differs")
        return model
    model = DEEP_CLASSES[family].load(folder, name=family, device="cpu")
    if model.n_sensors != len(condition["sensors"]):
        raise ValueError(f"{condition['experiment_id']}: checkpoint input width differs from the sensor subset")
    return model


def subset_inputs(windows, sensor_indices):
    """The condition's sensor columns, contiguous as in Session 7 (float64 trees depend on it)."""
    return np.ascontiguousarray(windows.X[:, :, list(sensor_indices)])


def prediction_path(cfg, dataset, experiment_id):
    return resolve_path(cfg["outputs"]["working_dir"]) / "predictions" / dataset / f"{experiment_id}.npz"


def stored_prediction(path, identity):
    """The stored prediction when it was produced from exactly `identity`; None when absent.
    Anything else is refused rather than overwritten."""
    side = path.with_suffix(".json")
    if not path.exists() or not side.exists():
        return None                                     # incomplete write: recompute
    meta = json.loads(side.read_text())
    if meta["identity"] != identity:
        raise RuntimeError(f"{path} was produced from other inputs or models; move it aside before rerunning")
    with np.load(path) as z:
        prediction = z["prediction"]
    if predictions_digest(prediction) != meta["predictions_sha256"]:
        raise RuntimeError(f"{path} does not match its recorded digest")
    return prediction


def save_prediction(path, identity, prediction, produced_by=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, prediction=prediction)
    tmp.replace(path)
    atomic_write_json(path.with_suffix(".json"), {"identity": identity,
                                                  "predictions_sha256": predictions_digest(prediction),
                                                  "produced_by": produced_by})


def identity_of(condition, record, windows_sha256):
    return {"experiment_id": condition["experiment_id"], "identity_sha256": condition["identity_sha256"],
            "model_files": sha256_json(record["files"]), "inputs": windows_sha256, "kind": "inference_only"}


def predict_all(cfg, canonical, inputs, say=print, produced_by=None, *, authorization=None):
    """{dataset: {experiment_id: predictions}}. FD001 validation goes first and must reproduce the
    canonical predictions of every model before that model touches FD003. Resuming an interrupted
    execution goes through here too and needs the same authorization."""
    require_authorization(authorization, "inference")
    require_new_location(cfg)
    digests = {k: windows_digest(w) for k, w in inputs.items()}
    out = {dataset: {} for dataset in inputs}
    ordered = sorted(inputs, key=lambda k: k != "fd001_validation")
    for n, (key, condition) in enumerate(sorted(canonical.conditions.items()), 1):
        record, model = canonical.records[key], None
        for dataset in ordered:
            identity = identity_of(condition, record, digests[dataset])
            path = prediction_path(cfg, dataset, key)
            prediction = stored_prediction(path, identity)
            if prediction is None:
                model = model or load_model(condition, canonical.run_dirs[key], record, canonical.sources,
                                            authorization=authorization)
                prediction = np.asarray(model.predict(subset_inputs(inputs[dataset], condition["sensor_indices"])),
                                        dtype=np.float64)
                if not np.isfinite(prediction).all() or len(prediction) != len(inputs[dataset].X):
                    raise RuntimeError(f"{key}: invalid predictions on {dataset}")
            if dataset == "fd001_validation" and predictions_digest(prediction) != record["predictions_sha256"]:
                raise RuntimeError(f"{key}: the loaded model does not reproduce its canonical FD001 predictions")
            if not path.exists() or not path.with_suffix(".json").exists():
                save_prediction(path, identity, prediction, produced_by)
            out[dataset][key] = prediction
        if n % 29 == 0:
            say(f"  {n}/{len(canonical.conditions)} conditions predicted")
    return out
