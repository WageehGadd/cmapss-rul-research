import csv
import json

import mlflow
import mlflow.artifacts
import numpy as np
import pytest

from src.baselines import build_baseline
from src.environment import capture_environment
from src.experiment import run_experiment
from src.results import RESULT_COLUMNS
from src.tracking import MlflowTracker
from tests.conftest import synthetic_data


def run_all(cfg, data):
    env = capture_environment(".", cfg["environment"]["library_names"], {"session2_config": cfg.ref})
    tracker = MlflowTracker(cfg["tracking"]["tracking_dir"], cfg["tracking"]["experiment_name"])
    return [run_experiment(build_baseline(spec), data, cfg, tracker, env) for spec in cfg["baselines"]], tracker


def test_end_to_end_runs_are_logged_exported_and_traceable(cfg, data):
    results, tracker = run_all(cfg, data)
    run_ids = [r.run_id for r in results]
    assert len(set(run_ids)) == 2

    # MLflow: metrics, params, tags and artifacts exist for each run ID.
    for result in results:
        run = mlflow.get_run(result.run_id)
        assert run.data.metrics["validation_rmse"] == pytest.approx(result.metrics["validation"]["rmse"])
        assert run.data.metrics["test_nasa_score"] == pytest.approx(result.metrics["test"]["nasa_score"])
        for name in ("train_time_s", "inference_latency_ms_per_batch", "throughput_samples_per_s",
                     "model_size_bytes"):
            assert name in run.data.metrics
        assert run.data.params["seed"] == "7"
        assert run.data.params["sensor_count"] == "14"
        assert run.data.params["sensors"].startswith("sensor_1,sensor_2")
        assert "git_commit" in run.data.tags
        artifacts = {a.path for a in mlflow.artifacts.list_artifacts(run_id=result.run_id)}
        assert {"model", "predictions.npz", "session2_config.yaml", "environment.json"} <= artifacts

    # CSV and JSON exports carry the run IDs and the required columns.
    json_rows = json.loads(open(cfg["outputs"]["results_json"]).read())["results"]
    with open(cfg["outputs"]["results_csv"], newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(json_rows) == len(csv_rows) == 4                    # 2 models x (validation, test)
    assert {r["run_id"] for r in json_rows} == set(run_ids)
    assert set(RESULT_COLUMNS) <= set(csv_rows[0])
    assert {r["model"] for r in json_rows} == {"mean_rul", "linear_regression"}

    # Run folders hold the model, predictions and the config needed to reproduce the run.
    for run_id in run_ids:
        run_dir = tracker.tracking_dir.parent / "runs" / run_id
        assert (run_dir / "session2_config.yaml").read_text() == cfg.text
        assert (run_dir / "model").is_dir() and (run_dir / "predictions.npz").exists()


def test_same_config_seed_and_data_reproduce_predictions_and_metrics(cfg):
    first, _ = run_all(cfg, synthetic_data(seed=0))
    second, _ = run_all(cfg, synthetic_data(seed=0))
    for a, b in zip(first, second):
        assert a.run_id != b.run_id                                 # every run has its own ID
        assert a.prediction_digests == b.prediction_digests
        for eval_set in a.metrics:
            assert a.metrics[eval_set] == b.metrics[eval_set]
            np.testing.assert_array_equal(a.predictions[eval_set], b.predictions[eval_set])


def test_mean_baseline_ignores_validation_and_test_labels(cfg):
    base = synthetic_data(seed=0)
    changed = synthetic_data(seed=0)
    changed.y["validation"][:] = 999.0
    changed.y["test"][:] = 999.0
    (a, _), (b, _) = run_all(cfg, base), run_all(cfg, changed)
    assert a[0].model_name == "mean_rul"
    np.testing.assert_array_equal(a[0].predictions["validation"], b[0].predictions["validation"])
    assert a[0].metrics["validation"]["rmse"] != b[0].metrics["validation"]["rmse"]
