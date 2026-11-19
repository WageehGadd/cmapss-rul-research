"""Window-to-tabular feature transformer for classical models.

Each Session 1 window [sequence_length, n_sensors] becomes one row of tabular features: six
temporal statistics per sensor. The transformer is stateless and deterministic, works on
existing windows only, and so never crosses a window (hence engine) boundary.

Feature definitions, for one sensor with values x[0..T-1] (T = sequence_length, x[T-1] is the
most recent cycle):

    last    x[T-1]
    mean    arithmetic mean of x
    std     population standard deviation of x (ddof = 0)
    min     min(x)
    max     max(x)
    slope   ordinary-least-squares slope of x against the cycle index t = 0..T-1, in sensor
            units per cycle:  sum((t - mean_t) * (x - mean_x)) / sum((t - mean_t)^2)

Column order is sensor-major: for sensor 0 all six features in the order of `FEATURES`, then
sensor 1, and so on. Names are "<sensor>__<feature>".
"""
import numpy as np

FEATURES = ("last", "mean", "std", "min", "max", "slope")
SEPARATOR = "__"


def ols_slope(windows):
    """Vectorised OLS slope over the time axis. windows [n, T, s] -> [n, s] (float64)."""
    windows = np.asarray(windows, dtype=np.float64)
    t = np.arange(windows.shape[1], dtype=np.float64)
    t_centered = t - t.mean()
    x_centered = windows - windows.mean(axis=1, keepdims=True)
    return np.einsum("t,nts->ns", t_centered, x_centered) / np.sum(t_centered ** 2)


def _compute(windows, feature):
    if feature == "last":
        return windows[:, -1, :]
    if feature == "mean":
        return windows.mean(axis=1)
    if feature == "std":
        return windows.std(axis=1)
    if feature == "min":
        return windows.min(axis=1)
    if feature == "max":
        return windows.max(axis=1)
    if feature == "slope":
        return ols_slope(windows)
    raise ValueError(f"unknown temporal feature '{feature}'")


class WindowFeatureTransformer:
    """Deterministic [n, T, n_sensors] -> [n, n_sensors * len(features)] transformer."""

    def __init__(self, sensors, features=FEATURES, sequence_length=None):
        features = tuple(features)
        unknown = [f for f in features if f not in FEATURES]
        if unknown or len(set(features)) != len(features) or not features:
            raise ValueError(f"features must be a non-empty, duplicate-free subset of {FEATURES}, "
                             f"got {features}")
        self.sensors = list(sensors)
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("sensors must be a non-empty list of unique names")
        self.features = features
        self.sequence_length = sequence_length

    @property
    def n_features(self):
        return len(self.sensors) * len(self.features)

    @property
    def feature_names(self):
        return [f"{s}{SEPARATOR}{f}" for s in self.sensors for f in self.features]

    @property
    def feature_sensors(self):
        """Sensor of each output column (same order as `feature_names`)."""
        return [s for s in self.sensors for _ in self.features]

    def sensor_to_columns(self):
        """{sensor: [column indices of its features]}."""
        width = len(self.features)
        return {s: list(range(i * width, (i + 1) * width)) for i, s in enumerate(self.sensors)}

    def sensor_to_features(self):
        """{sensor: [feature names]}."""
        return {s: [f"{s}{SEPARATOR}{f}" for f in self.features] for s in self.sensors}

    def transform(self, windows):
        windows = np.asarray(windows)
        if windows.ndim != 3 or windows.shape[2] != len(self.sensors):
            raise ValueError(f"expected windows of shape [n, T, {len(self.sensors)}], got {windows.shape}")
        if self.sequence_length is not None and windows.shape[1] != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} cycles per window, got {windows.shape[1]}")
        if windows.shape[1] < 2 and "slope" in self.features:
            raise ValueError("the slope feature needs at least 2 cycles per window")
        windows = windows.astype(np.float64, copy=False)       # never writes into the input
        out = np.empty((windows.shape[0], self.n_features), dtype=np.float64)
        width = len(self.features)
        for j, feature in enumerate(self.features):
            out[:, j::width] = _compute(windows, feature)
        return out

    def get_config(self):
        return {"features": list(self.features), "sensors": list(self.sensors),
                "sequence_length": self.sequence_length, "n_features": self.n_features,
                "slope": "ols over cycle index 0..T-1", "std": "population (ddof=0)"}
