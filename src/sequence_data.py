"""Sequence windows for the recurrent models, for the canonical and alternative window lengths.

The canonical length (Session 1, L=30) is loaded from the frozen, manifest-checked arrays.
Other lengths are regenerated with the frozen Session 1 functions from the same inputs:

  * the same train/validation engine split (processed_dir/split.json),
  * the same fitted scaler (processed_dir/scaler.joblib), only applied, never refitted,
  * the same capped-RUL definition, and the same window rules (stride 1, final-timestep
    target, test = last L cycles per engine with the official RUL).

Nothing in data/processed is written or replaced.
"""
import json
from dataclasses import dataclass

import joblib
import numpy as np
import yaml

from .benchmark_config import resolve_path
from .benchmark_data import load_benchmark_data
from .data_loading import apply_schema, read_raw_table, read_rul_file
from .rul import add_rul
from .scaling import apply_scaler
from .splitting import select_units
from .windows import make_test_windows, make_training_windows


@dataclass(frozen=True)
class SequenceData:
    X: dict            # split -> float32 [n, sequence_length, n_sensors]
    y: dict            # split -> float32 [n]
    unit_ids: dict
    end_cycles: dict   # split -> int64 [n] cycle of the final timestep of each window
    sensors: list
    target_type: str
    rul_cap: int
    sequence_length: int


def _canonical(processed_dir, session1_config, target_type):
    base = load_benchmark_data(processed_dir, session1_config, target_type)
    ends = {}
    for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
        with np.load(resolve_path(processed_dir) / f"{prefix}_windows.npz") as z:
            ends[split] = z[f"{prefix}_end_cycles"]
    return SequenceData(X=base.X, y=base.y, unit_ids=base.unit_ids, end_cycles=ends, sensors=base.sensors,
                        target_type=base.target_type, rul_cap=base.rul_cap,
                        sequence_length=base.sequence_length)


def regenerate_windows(processed_dir, session1_config, target_type, sequence_length):
    """Windows of an arbitrary length from the frozen split and scaler (no refitting)."""
    processed_dir, session1_config = resolve_path(processed_dir), resolve_path(session1_config)
    s1 = yaml.safe_load(session1_config.read_text())
    feats, cap, ds = s1["features"]["selected"], int(s1["rul"]["cap"]), s1["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    split = json.loads((processed_dir / "split.json").read_text())
    scaler = joblib.load(processed_dir / "scaler.joblib")

    train = add_rul(apply_schema(read_raw_table(raw_dir / f"train_{ds['subset']}.txt")), cap)
    test = apply_schema(read_raw_table(raw_dir / f"test_{ds['subset']}.txt"))
    official = read_rul_file(raw_dir / f"RUL_{ds['subset']}.txt")
    parts = {"train": apply_scaler(scaler, select_units(train, split["train_units"]), feats),
             "validation": apply_scaler(scaler, select_units(train, split["val_units"]), feats)}
    sets = {name: make_training_windows(df, feats, sequence_length) for name, df in parts.items()}
    sets["test"] = make_test_windows(apply_scaler(scaler, test, feats), feats, sequence_length, official, cap)
    key = "y" if target_type == "capped" else "y_raw"
    return SequenceData(
        X={k: w.X for k, w in sets.items()}, y={k: getattr(w, key) for k, w in sets.items()},
        unit_ids={k: w.unit_ids for k, w in sets.items()}, end_cycles={k: w.end_cycles for k, w in sets.items()},
        sensors=list(feats), target_type=target_type, rul_cap=cap, sequence_length=int(sequence_length))


def load_sequence_data(processed_dir, session1_config, target_type, sequence_length):
    """Canonical length -> frozen Session 1 arrays; other lengths -> regenerated windows."""
    canonical = int(yaml.safe_load(resolve_path(session1_config).read_text())["windows"]["sequence_length"])
    if int(sequence_length) == canonical:
        return _canonical(processed_dir, session1_config, target_type)
    return regenerate_windows(processed_dir, session1_config, target_type, sequence_length)


def window_counts(data):
    return {split: int(len(x)) for split, x in data.X.items()}
