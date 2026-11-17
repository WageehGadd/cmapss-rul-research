import shutil

import numpy as np
import pytest

from src.benchmark_config import resolve_path
from src.benchmark_data import load_benchmark_data

PROCESSED = "data/processed/fd001"
SESSION1_CONFIG = "configs/session1_fd001.yaml"

pytestmark = pytest.mark.skipif(not (resolve_path(PROCESSED) / "manifest.json").exists(),
                                reason="Session 1 artifacts not generated")


def test_session1_shapes_and_capped_target():
    data = load_benchmark_data(PROCESSED, SESSION1_CONFIG, "capped")
    assert data.X["train"].shape == (14459, 30, 14)
    assert data.X["validation"].shape == (3272, 30, 14)
    assert data.X["test"].shape == (100, 30, 14)
    assert len(data.sensors) == 14 and data.rul_cap == 125 and data.sequence_length == 30
    for split in data.y:
        assert data.y[split].max() <= 125
    assert not set(data.unit_ids["train"]) & set(data.unit_ids["validation"])


def test_raw_target_is_uncapped_and_differs_only_above_the_cap():
    capped = load_benchmark_data(PROCESSED, SESSION1_CONFIG, "capped")
    raw = load_benchmark_data(PROCESSED, SESSION1_CONFIG, "raw")
    np.testing.assert_array_equal(raw.y["train"].clip(max=125), capped.y["train"])
    assert raw.y["train"].max() > 125


def test_modified_arrays_are_detected(tmp_path):
    copy = tmp_path / "fd001"
    shutil.copytree(resolve_path(PROCESSED), copy)
    with np.load(copy / "train_windows.npz") as z:
        arrays = {k: z[k] for k in z.files}
    arrays["train_y"] = arrays["train_y"] + 1
    np.savez(copy / "train_windows.npz", **arrays)
    with pytest.raises(ValueError, match="manifest"):
        load_benchmark_data(copy, SESSION1_CONFIG, "capped")


def test_missing_session1_artifacts_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_session1"):
        load_benchmark_data(tmp_path, SESSION1_CONFIG, "capped")
