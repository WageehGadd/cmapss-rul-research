"""MLflow experiment tracking on a local SQLite store (no server needed).

MLflow 3.x no longer accepts the plain file store by default, so runs are kept in
<tracking_dir>/mlflow.db with artifacts under <tracking_dir>/artifacts. The MLflow run ID is
the benchmark's unique experiment ID and is written into every results-table row.
"""
import os
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from .benchmark_config import resolve_path  # noqa: E402


class MlflowTracker:
    def __init__(self, tracking_dir, experiment_name):
        self.tracking_dir = resolve_path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{self.tracking_dir / 'mlflow.db'}"
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name,
                                     artifact_location=(self.tracking_dir / "artifacts").as_uri())
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name, tags=None):
        """Open an MLflow run and yield its unique run ID."""
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run.info.run_id

    @staticmethod
    def log_evaluation(result, params, run_dir):
        """Log params and metrics of an EvaluationResult to the active run, and upload run_dir
        (model files in model/, plus predictions, config, environment, sensors) as artifacts."""
        mlflow.log_params({k: ("" if v is None else v) for k, v in params.items()})
        metrics = {}
        for eval_set, values in result.metrics.items():
            metrics.update({f"{eval_set}_{name}": value for name, value in values.items()})
        t = result.timing
        metrics.update({
            "train_time_s": result.train_time_s,
            "inference_latency_ms_per_batch": t["latency_ms_per_batch"],
            "inference_latency_ms_per_sample": t["latency_ms_per_sample"],
            "inference_latency_ms_per_batch_p95": t["latency_ms_per_batch_p95"],
            "throughput_samples_per_s": t["throughput_samples_per_s"],
            "model_size_bytes": result.model_size_bytes,
        })
        if result.n_parameters is not None:
            metrics["n_parameters"] = result.n_parameters
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir))
