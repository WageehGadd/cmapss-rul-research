"""Load the frozen Session 1 dataset for benchmarking.

The arrays are read from data/processed/fd001/ and checked against the SHA-256 manifest that
Session 1 wrote, so a benchmark can never silently run on modified data.

Target semantics: `target_type` selects capped RUL (Session 1 training target, min(RUL, cap))
or raw RUL. The same choice is used to train and to score every model and every split.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .benchmark_config import resolve_path

# Session 1 names the validation arrays with the prefix "val".
_FILE_PREFIX = {"train": "train", "validation": "val", "test": "test"}
SESSION1_COMMAND = "python -m src.run_session1 --config configs/session1_fd001.yaml"


def array_digest(arr):
    """SHA-256 of dtype, shape and contents (same convention as the Session 1 manifest)."""
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(f"{arr.dtype}{arr.shape}".encode() + arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class BenchmarkData:
    X: dict            # split -> float32 [n, sequence_length, n_sensors]
    y: dict            # split -> float32 [n] target selected by target_type
    unit_ids: dict     # split -> int64 [n] engine of each window
    sensors: list
    target_type: str
    rul_cap: int
    sequence_length: int


def load_benchmark_data(processed_dir, session1_config, target_type):
    processed_dir, session1_config = resolve_path(processed_dir), resolve_path(session1_config)
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run Session 1 first: {SESSION1_COMMAND}")
    manifest = json.loads(manifest_path.read_text())

    X, y, unit_ids = {}, {}, {}
    for split, prefix in _FILE_PREFIX.items():
        npz_path = processed_dir / f"{prefix}_windows.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found. Run Session 1 first: {SESSION1_COMMAND}")
        with np.load(npz_path) as z:
            arrays = {key: z[key] for key in z.files}
        for key, arr in arrays.items():
            if array_digest(arr) != manifest[key]:
                raise ValueError(f"{npz_path.name}:{key} does not match the Session 1 manifest; "
                                 f"re-run Session 1 ({SESSION1_COMMAND})")
        X[split] = arrays[f"{prefix}_X"]
        y[split] = arrays[f"{prefix}_y" if target_type == "capped" else f"{prefix}_y_raw"]
        unit_ids[split] = arrays[f"{prefix}_unit_ids"]

    session1 = yaml.safe_load(session1_config.read_text())
    processed = yaml.safe_load((processed_dir / "resolved_config.yaml").read_text())
    if session1["features"]["selected"] != processed["features"]["selected"] or \
            session1["rul"]["cap"] != processed["rul"]["cap"] or \
            session1["windows"]["sequence_length"] != processed["windows"]["sequence_length"]:
        raise ValueError(f"{session1_config.name} differs from the configuration the processed data "
                         f"was built with; re-run Session 1 ({SESSION1_COMMAND})")
    sensors = list(session1["features"]["selected"])
    seq_len = int(session1["windows"]["sequence_length"])
    for split, arr in X.items():
        if arr.ndim != 3 or arr.shape[1:] != (seq_len, len(sensors)):
            raise ValueError(f"{split} windows have shape {arr.shape}, expected [n, {seq_len}, {len(sensors)}]")
        if len(y[split]) != len(arr):
            raise ValueError(f"{split}: {len(arr)} windows but {len(y[split])} targets")
    return BenchmarkData(X=X, y=y, unit_ids=unit_ids, sensors=sensors, target_type=target_type,
                         rul_cap=int(session1["rul"]["cap"]), sequence_length=seq_len)
