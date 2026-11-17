"""Shared regression metrics for the benchmark. Every model is scored with these functions.

All metrics take 1-D arrays of true and predicted RUL (in cycles) of equal length.

NASA asymmetric score
---------------------
With d_i = predicted_RUL_i - true_RUL_i:

    d_i <  0  (early / conservative prediction):  s_i = exp(-d_i / 13) - 1
    d_i >= 0  (late / optimistic prediction):     s_i = exp( d_i / 10) - 1

The score is the sum of s_i (lower is better). A late prediction of a given size is
penalised more than an early prediction of the same size because its divisor is smaller.
"""
import numpy as np

NASA_EARLY_DIVISOR = 13.0
NASA_LATE_DIVISOR = 10.0


def _validate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got shapes {y_true.shape} and {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in length: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("metrics need at least one sample")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must be finite")
    return y_true, y_pred


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true, y_pred):
    """Coefficient of determination, 1 - SS_res / SS_tot.

    If y_true is constant (SS_tot == 0) the ratio is undefined: the score is 1.0 for perfect
    predictions and 0.0 otherwise, so the result is always finite.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def nasa_score(y_true, y_pred):
    """Standard C-MAPSS asymmetric score, summed over samples (lower is better)."""
    y_true, y_pred = _validate(y_true, y_pred)
    d = y_pred - y_true
    terms = np.where(d < 0.0,
                     np.exp(-d / NASA_EARLY_DIVISOR) - 1.0,
                     np.exp(d / NASA_LATE_DIVISOR) - 1.0)
    return float(terms.sum())


def compute_metrics(y_true, y_pred):
    """All four benchmark metrics as a dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
