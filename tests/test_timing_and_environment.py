import time

import numpy as np
import pytest

from src.environment import UNAVAILABLE, capture_environment
from src.timing import measure_inference, time_training


class SleepyModel:
    def fit(self, X, y):
        time.sleep(0.02)

    def predict(self, X):
        time.sleep(0.002)
        return np.zeros(len(X))


def test_training_time_is_wall_clock_seconds():
    seconds = time_training(SleepyModel(), np.zeros((4, 3, 2)), np.zeros(4))
    assert 0.02 <= seconds < 1.0


def test_inference_timing_output_and_consistency():
    batch = np.zeros((32, 3, 2))
    t = measure_inference(SleepyModel().predict, batch, warmup_runs=2, timed_runs=5)
    assert set(t) >= {"latency_ms_per_batch", "latency_ms_per_sample", "throughput_samples_per_s",
                      "latency_ms_per_batch_p95", "batch_size", "warmup_runs", "timed_runs", "aggregate"}
    assert t["batch_size"] == 32 and t["warmup_runs"] == 2 and t["timed_runs"] == 5
    assert t["aggregate"] == "median" and t["latency_unit"] == "ms"
    assert 2.0 <= t["latency_ms_per_batch"] < 200.0
    assert t["latency_ms_per_sample"] == pytest.approx(t["latency_ms_per_batch"] / 32)
    assert t["throughput_samples_per_s"] == pytest.approx(32 / (t["latency_ms_per_batch"] / 1000.0))
    assert t["latency_ms_per_batch_p95"] >= t["latency_ms_per_batch"] >= t["latency_ms_per_batch_min"]


def test_warmup_calls_are_not_timed_but_are_executed():
    calls = []
    measure_inference(lambda b: calls.append(1), np.zeros((4, 1, 1)), warmup_runs=3, timed_runs=7)
    assert len(calls) == 10


@pytest.mark.parametrize("warmup, timed", [(-1, 5), (0, 0)])
def test_invalid_timing_protocol_raises(warmup, timed):
    with pytest.raises(ValueError):
        measure_inference(lambda b: None, np.zeros((2, 1, 1)), warmup, timed)


def test_environment_metadata_schema(tmp_path):
    env = capture_environment(tmp_path, ["numpy", "definitely-not-installed"],
                              {"session1_config": "a.yaml", "session2_config": "b.yaml@abc"})
    assert set(env) == {"os", "python", "hardware", "libraries", "thread_env", "git", "configs"}
    assert set(env["os"]) == {"system", "release", "version", "platform"}
    assert {"version", "implementation"} <= set(env["python"])
    assert {"cpu_brand", "physical_cpu_count", "logical_cpu_count", "memory_bytes", "machine",
            "processor", "description"} <= set(env["hardware"])
    assert env["libraries"]["numpy"] == np.__version__
    assert env["libraries"]["definitely-not-installed"] == UNAVAILABLE
    # tmp_path is not a git repository: reported as unavailable instead of invented
    assert env["git"] == {"git_commit": UNAVAILABLE, "git_dirty": UNAVAILABLE}
    assert env["configs"]["session1_config"] == "a.yaml"


def test_environment_records_git_commit_of_the_project():
    from src.benchmark_config import PROJECT_ROOT
    env = capture_environment(PROJECT_ROOT, [], {})
    commit = env["git"]["git_commit"]
    assert commit == UNAVAILABLE or (len(commit) == 40 and set(commit) <= set("0123456789abcdef"))


def test_hardware_note_is_kept():
    env = capture_environment(".", [], {}, hardware_note="lab machine 3")
    assert env["hardware"]["config_note"] == "lab machine 3"
