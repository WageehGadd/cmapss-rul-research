import csv
import json

import numpy as np
import pytest

from src.evaluation import EvaluationResult, evaluate_model
from src.results import RESULT_COLUMNS, append_results, build_rows, load_results

TIMING = {"warmup_runs": 1, "timed_runs": 3, "batch_size": 8}
GIT = {"git_commit": "a" * 40, "git_dirty": False}


class ConstantModel:
    """A model that is not LinearRegression: the evaluator only needs the documented interface."""
    name = "constant_42"

    def predict(self, X):
        return np.full(len(X), 42.0)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "constant.txt"
        path.write_text("42")
        return [path]


def evaluate(model, tmp_path, X=None, y=None):
    X = np.zeros((20, 30, 14), dtype="float32") if X is None else X
    y = np.arange(20, dtype=float) if y is None else y
    return evaluate_model(model, {"validation": (X, y), "test": (X[:5], y[:5])}, X,
                          target_type="capped", train_time_s=0.5, timing_config=TIMING,
                          artifact_dir=tmp_path / "model", run_id="run-1",
                          metadata={"seed": 1})


def test_evaluator_output_schema(tmp_path):
    result = evaluate(ConstantModel(), tmp_path)
    assert isinstance(result, EvaluationResult)
    assert result.run_id == "run-1" and result.model_name == "constant_42"
    assert result.target_type == "capped"
    assert set(result.predictions) == set(result.metrics) == {"validation", "test"}
    assert result.predictions["validation"].shape == (20,)
    assert set(result.metrics["validation"]) == {"rmse", "mae", "r2", "nasa_score"}
    assert result.train_time_s == 0.5
    assert {"latency_ms_per_batch", "latency_ms_per_sample", "throughput_samples_per_s",
            "batch_size"} <= set(result.timing)
    assert result.timing["batch_size"] == 8
    assert result.model_size_bytes == 2                     # the 2-byte file written by save()
    assert result.n_parameters is None                      # optional for models
    assert result.n_samples == {"validation": 20, "test": 5}
    assert result.metadata == {"seed": 1}
    assert set(result.prediction_digests) == {"validation", "test"}


def test_evaluator_metrics_match_hand_values(tmp_path):
    result = evaluate(ConstantModel(), tmp_path, y=np.full(20, 40.0))
    m = result.metrics["validation"]
    assert m["mae"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)
    assert m["r2"] == 0.0                                    # constant target, imperfect prediction


def test_evaluator_generates_a_run_id_when_missing(tmp_path):
    X, y = np.zeros((20, 30, 14), dtype="float32"), np.arange(20, dtype=float)
    ids = {evaluate_model(ConstantModel(), {"validation": (X, y)}, X, target_type="capped",
                          train_time_s=0.0, timing_config=TIMING,
                          artifact_dir=tmp_path / f"m{i}").run_id for i in range(2)}
    assert len(ids) == 2


def test_evaluator_rejects_wrong_prediction_shape(tmp_path):
    class Bad(ConstantModel):
        def predict(self, X):
            return np.zeros((len(X), 2))

    with pytest.raises(ValueError, match="predict returned shape"):
        evaluate(Bad(), tmp_path)


def test_evaluator_needs_enough_timing_samples(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        evaluate(ConstantModel(), tmp_path, X=np.zeros((4, 30, 14), dtype="float32"), y=np.zeros(4))


def make_rows(tmp_path, run_id="run-1"):
    result = evaluate(ConstantModel(), tmp_path)
    result.run_id = run_id
    return build_rows(result, sensor_set="s14", sensor_count=14, seed=1, git=GIT,
                      config_ref="configs/x.yaml", config_sha256="abc", session1_config_ref="configs/s1.yaml")


def test_results_rows_contain_all_required_columns(tmp_path):
    rows = make_rows(tmp_path)
    assert [r["eval_set"] for r in rows] == ["validation", "test"]
    for row in rows:
        assert set(RESULT_COLUMNS) <= set(row)
        assert row["run_id"] == "run-1" and row["sensor_count"] == 14 and row["target_type"] == "capped"
    official = {"model", "sensor_set", "sensor_count", "seed", "rmse", "mae", "r2", "nasa_score",
                "inference_latency_ms_per_batch", "throughput_samples_per_s", "model_size_bytes",
                "train_time_s"}
    assert official <= set(RESULT_COLUMNS)


def test_results_export_csv_and_json_and_append(tmp_path):
    csv_path, json_path = tmp_path / "out" / "r.csv", tmp_path / "out" / "r.json"
    append_results(make_rows(tmp_path / "a", "run-1"), csv_path, json_path)
    append_results(make_rows(tmp_path / "b", "run-2"), csv_path, json_path)

    document = json.loads(json_path.read_text())
    assert document["schema_version"] == 1 and document["columns"] == RESULT_COLUMNS
    assert [r["run_id"] for r in document["results"]] == ["run-1", "run-1", "run-2", "run-2"]
    assert load_results(json_path) == document["results"]

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == RESULT_COLUMNS
    assert len(rows) == 4 and rows[2]["run_id"] == "run-2"
    assert float(rows[0]["rmse"]) == pytest.approx(document["results"][0]["rmse"])


def test_append_keeps_extra_columns_for_later_sessions(tmp_path):
    rows = make_rows(tmp_path)
    rows[0]["extra_field"] = 5
    append_results(rows, tmp_path / "r.csv", tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text())["columns"][-1] == "extra_field"
    with open(tmp_path / "r.csv", newline="") as f:
        assert csv.DictReader(f).fieldnames[-1] == "extra_field"


def test_append_rejects_rows_missing_required_columns(tmp_path):
    row = make_rows(tmp_path)[0]
    del row["nasa_score"]
    with pytest.raises(ValueError, match="nasa_score"):
        append_results([row], tmp_path / "r.csv", tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()


def test_load_results_of_missing_file_is_empty(tmp_path):
    assert load_results(tmp_path / "nope.json") == []
