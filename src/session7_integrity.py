"""Integrity, provenance and locking rules for the Session 7 condition records.

Two separate questions are asked of a stored record:

  validate_record      Is the evidence intact? (identity, referenced files, predictions, every
                       metric, resource values, MLflow run). Used before any aggregation. It does
                       not care which source produced the record, so older verification records can
                       still be inspected.
  provenance_mismatch  Was it produced by exactly the present source state? (execution code,
                       settings, commit, clean/dirty). Required before a record may be reused by an
                       execution, so a clean-commit run can never adopt dirty verification results.

Everything here is read-only: the MLflow store is opened in SQLite read-only mode.
"""
import fcntl
import json
import math
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .evaluation import predictions_digest
from .metrics import compute_metrics

PROVENANCE_FIELDS = (("code_fingerprint", "execution code changed"),
                     ("execution_fingerprint", "execution settings changed"),
                     ("git_commit", "source commit changed"),
                     ("git_dirty", "working-tree clean/dirty state changed"))
# record field -> MLflow metric name (written independently of the record)
RESOURCE_METRICS = {"train_time_s": "train_time_s", "model_size_bytes": "model_size_bytes",
                    "inference_latency_ms_per_batch": "inference_latency_ms_per_batch",
                    "inference_latency_ms_per_sample": "inference_latency_ms_per_sample",
                    "throughput_samples_per_s": "throughput_samples_per_s", "n_parameters": "n_parameters"}


class RunStore:
    """Read-only view of the local MLflow SQLite store."""

    def __init__(self, tracking_dir):
        self.path = Path(tracking_dir) / "mlflow.db"

    def get(self, run_id):
        """The run with its experiment name, params, tags and metrics, or None."""
        if not self.path.exists():
            return None
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        try:
            row = db.execute("select r.status, r.lifecycle_stage, e.name from runs r join experiments e "
                             "on r.experiment_id = e.experiment_id where r.run_uuid = ?", (run_id,)).fetchone()
            if row is None:
                return None
            pairs = lambda table: dict(db.execute(f"select key, value from {table} where run_uuid = ?", (run_id,)))
            return {"status": row[0], "lifecycle_stage": row[1], "experiment": row[2], "params": pairs("params"),
                    "tags": pairs("tags"), "metrics": pairs("latest_metrics")}
        finally:
            db.close()


def _close(a, b):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def validate_record(condition, record, y_validation, store, experiment_name, sha256_file):
    """(valid, reason) for the evidence behind one completed record."""
    if record is None:
        return False, "no readable record"
    if record.get("status") != "completed":
        return False, f"status {record.get('status')}"
    if record.get("identity_sha256") != condition["identity_sha256"]:
        return False, "condition identity changed"
    for key in ("experiment_id", "model", "budget", "strategy", "sensors", "sensor_indices", "training_seed"):
        if record.get(key) != condition[key]:
            return False, f"record {key} differs from the plan"
    try:
        run_dir = Path(record["run_dir"])
        files = record["files"]
        if "predictions.npz" not in files or not any(f.startswith("model/") for f in files):
            return False, "record does not reference predictions and model files"
        for rel, digest in files.items():
            if sha256_file(run_dir / rel) != digest:
                return False, f"file changed: {rel}"
        if json.loads((run_dir / "sensors.json").read_text()) != condition["sensors"]:
            return False, "saved sensor list differs from the plan"
        with np.load(run_dir / "predictions.npz") as z:
            if z.files != ["validation"]:
                return False, "predictions contain a split other than validation"
            pred = z["validation"]
        if pred.shape != np.shape(y_validation) or predictions_digest(pred) != record["predictions_sha256"]:
            return False, "predictions do not match the record"
        for name, value in compute_metrics(y_validation, pred).items():
            if not _close(record["metrics"].get(name), value):
                return False, f"metrics do not match the predictions: {name}"

        run = store.get(record["mlflow_run_id"])
        if run is None:
            return False, "MLflow run does not exist"
        if run["lifecycle_stage"] != "active":
            return False, "MLflow run is deleted"
        if run["status"] != "FINISHED":
            return False, "MLflow run is not FINISHED"
        if run["experiment"] != experiment_name:
            return False, "MLflow run belongs to another experiment"
        if run["tags"].get("s7.experiment_id") != condition["experiment_id"] \
                or run["params"].get("s7.identity_sha256") != condition["identity_sha256"]:
            return False, "MLflow run identifies another condition"
        if run["tags"].get("s7.status") == "superseded":
            return False, "MLflow run is marked superseded"
        for key in ("git_commit", "git_dirty"):
            if str(run["params"].get(key)) != str(record.get(key)):
                return False, f"record {key} differs from its MLflow run"
        if any(name.startswith("test") for name in run["metrics"]):
            return False, "MLflow run has test metrics"
        for name, value in record["metrics"].items():
            if not _close(run["metrics"].get(f"validation_{name}"), value):
                return False, f"metrics do not match MLflow: {name}"
        for field, metric in RESOURCE_METRICS.items():
            if record.get(field) is not None and not _close(record[field], run["metrics"].get(metric)):
                return False, f"{field} does not match MLflow"

        if record.get("weights_sha256"):                      # deep models: the saved training summary
            saved = [f for f in files if f.endswith("_config.json")]
            summary = json.loads((run_dir / saved[0]).read_text())
            for key in ("best_epoch", "epochs_run", "stop_reason", "n_parameters"):
                if record.get(key) != summary.get(key):
                    return False, f"{key} does not match the saved model summary"
            import torch

            from .recurrent_models import weights_digest
            checkpoint = [f for f in files if f.endswith(".pt")][0]
            blob = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
            if weights_digest(blob["state_dict"]) != record["weights_sha256"]:
                return False, "weights do not match the record"
    except Exception as error:                                  # missing file, key or unreadable content
        return False, f"integrity check failed: {type(error).__name__}"
    return True, "ok"


def provenance_mismatch(record, current):
    """Reasons why `record` was not produced by the present source state (empty when it was)."""
    return [reason for field, reason in PROVENANCE_FIELDS if record.get(field) != current[field]]


@contextmanager
def execution_lock(working_dir):
    """One writer at a time, using a kernel advisory lock (POSIX flock).

    Acquisition is atomic and the lock disappears with its process, so there is no stale-lock
    clean-up and therefore no race around it. The lock file itself is never deleted."""
    path = Path(working_dir) / "execution.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            raise RuntimeError(f"another Session 7 execution is running (pid {handle.read().strip() or '?'}); "
                               f"lock {path}") from None
        handle.truncate(0)
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        handle.close()                                          # closing releases the lock
