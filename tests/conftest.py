import numpy as np
import pytest
import yaml

from src.benchmark_config import load_benchmark_config
from src.benchmark_data import BenchmarkData

SENSORS = [f"sensor_{i}" for i in range(1, 15)]
SEQ_LEN = 30


def synthetic_data(seed=0, target_type="capped"):
    """Small stand-in for the Session 1 dataset with the same array layout."""
    rng = np.random.default_rng(seed)
    sizes = {"train": 300, "validation": 120, "test": 40}
    X = {k: rng.normal(size=(n, SEQ_LEN, len(SENSORS))).astype("float32") for k, n in sizes.items()}
    y = {k: rng.uniform(1, 125, size=n).astype("float32") for k, n in sizes.items()}
    units = {k: np.arange(n, dtype="int64") for k, n in sizes.items()}
    return BenchmarkData(X=X, y=y, unit_ids=units, sensors=list(SENSORS), target_type=target_type,
                         rul_cap=125, sequence_length=SEQ_LEN)


@pytest.fixture
def data():
    return synthetic_data()


@pytest.fixture
def cfg(tmp_path):
    """A benchmark config whose outputs and MLflow store live in a temp directory."""
    values = {
        "session1": {"config": "configs/session1_fd001.yaml", "processed_dir": "data/processed/fd001"},
        "benchmark": {"target_type": "capped", "seed": 7, "evaluation_sets": ["validation", "test"],
                      "sensor_set_name": "synthetic_14"},
        "baselines": [{"name": "mean_rul", "type": "mean_rul"},
                      {"name": "linear_regression", "type": "linear_regression",
                       "representation": "last_timestep"}],
        "timing": {"warmup_runs": 1, "timed_runs": 3, "batch_size": 16, "latency_unit": "ms",
                   "timing_set": "validation", "hardware_description": "auto"},
        "tracking": {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_benchmark"},
        "outputs": {"results_csv": str(tmp_path / "results.csv"),
                    "results_json": str(tmp_path / "results.json"),
                    "runs_dir": str(tmp_path / "runs")},
        "environment": {"library_names": ["numpy", "scikit-learn"]},
    }
    path = tmp_path / "session2_test.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return load_benchmark_config(path)
