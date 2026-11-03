"""Sequence window generation.

Windowing semantics
-------------------
* A window holds `sequence_length` (L) CONSECUTIVE cycles of ONE engine, stride 1.
* X has shape [n_windows, L, n_features]; time runs along axis 1, oldest cycle first.
* The target of a window is the RUL of its FINAL timestep (the last cycle in the window).
* Windows never cross engine boundaries and are never padded: an engine with n cycles gives
  n - L + 1 training windows, the first one ending at cycle L.
* Test data: one window per engine, made of its last L cycles, labelled with the official RUL.
"""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .checks import SanityCheckError


@dataclass
class WindowSet:
    X: np.ndarray            # float32 [n_windows, L, n_features]
    y: np.ndarray            # float32 [n_windows]  capped RUL at the final timestep
    y_raw: np.ndarray        # float32 [n_windows]  uncapped RUL at the final timestep
    unit_ids: np.ndarray     # int64   [n_windows]  engine of each window
    end_cycles: np.ndarray   # int64   [n_windows]  cycle of the final timestep


def _unit_windows(features_arr, seq_len):
    """All stride-1 windows of one engine: [n - L + 1, L, n_features]."""
    return sliding_window_view(features_arr, seq_len, axis=0).transpose(0, 2, 1)


def _stack(parts):
    X, y, y_raw, units, ends = (np.concatenate(p) for p in zip(*parts))
    return WindowSet(X.astype("float32"), y.astype("float32"), y_raw.astype("float32"),
                     units.astype("int64"), ends.astype("int64"))


def _require_length(unit, n_cycles, seq_len):
    if n_cycles < seq_len:
        raise SanityCheckError(
            f"Engine {unit} has only {n_cycles} cycles, fewer than sequence_length={seq_len}")


def make_training_windows(df, features, seq_len):
    """Every stride-1 window of every engine in df (needs rul / rul_raw columns)."""
    parts = []
    for unit, g in df.groupby("unit_id", sort=True):
        _require_length(unit, len(g), seq_len)
        windows = _unit_windows(g[features].to_numpy(dtype="float64"), seq_len)
        parts.append((windows, g["rul"].to_numpy()[seq_len - 1:],
                      g["rul_raw"].to_numpy()[seq_len - 1:],
                      np.full(len(windows), unit), g["cycle"].to_numpy()[seq_len - 1:]))
    return _stack(parts)


def make_test_windows(df, features, seq_len, official_rul, rul_cap):
    """One window per test engine (its last L cycles); official_rul is ordered by unit_id."""
    parts = []
    for i, (unit, g) in enumerate(df.groupby("unit_id", sort=True)):
        _require_length(unit, len(g), seq_len)
        window = g[features].to_numpy(dtype="float64")[-seq_len:][None]
        raw = float(official_rul[i])
        parts.append((window, np.array([min(raw, rul_cap)]), np.array([raw]),
                      np.array([unit]), g["cycle"].to_numpy()[-1:]))
    return _stack(parts)
