"""Synthetic training/validation data with the Session 1 sensor names (no test split)."""
import numpy as np
import pandas as pd

from src.benchmark_config import PROJECT_ROOT
from src.sequence_data import SequenceData
from src.windows import make_training_windows

CONFIG = PROJECT_ROOT / "configs/session7_main_experiment.yaml"
CANONICAL = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
IDENTITY = {"manifest_sha256": "synthetic", "split_sha256": "synthetic", "config_sha256": "synthetic",
            "shapes": {"train": [0, 30, 14], "validation": [0, 30, 14]}, "n_engines": {"train": 6, "validation": 5}}


def synthetic(seed=0, n_train=None, n_validation=None):
    rng = np.random.default_rng(seed)
    X, y, units, ends = {}, {}, {}, {}
    for split, engine_ids in (("train", range(1, 7)), ("validation", range(7, 12))):
        parts = []
        for u in engine_ids:
            n = int(rng.integers(50, 60))
            df = pd.DataFrame(rng.normal(size=(n, 14)), columns=CANONICAL)
            df["unit_id"], df["cycle"] = u, np.arange(1, n + 1)
            df["rul_raw"] = n - df["cycle"]
            df["rul"] = df["rul_raw"].clip(upper=125)
            df["sensor_11"] += (1.0 - df["rul"] / 60.0) * 3
            parts.append(df)
        w = make_training_windows(pd.concat(parts, ignore_index=True), CANONICAL, 30)
        X[split], y[split], units[split], ends[split] = w.X, w.y, w.unit_ids, w.end_cycles
    for split, limit in (("train", n_train), ("validation", n_validation)):
        if limit:
            X[split], y[split], units[split], ends[split] = (a[:limit] for a in (X[split], y[split], units[split], ends[split]))
    X = {k: np.ascontiguousarray(v) for k, v in X.items()}          # layout of the frozen arrays
    return SequenceData(X=X, y=y, unit_ids=units, end_cycles=ends, sensors=list(CANONICAL), target_type="capped",
                        rul_cap=125, sequence_length=30)
