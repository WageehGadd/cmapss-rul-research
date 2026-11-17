"""Session 2 baselines.

Every model plugged into the benchmark follows the same small interface, which the shared
evaluator relies on and later model families can implement:

    name                 str, unique within a benchmark
    fit(X, y)            X: Session 1 windows [n, sequence_length, n_sensors], y: target [n]
    predict(X)           -> 1-D array of n RUL predictions
    get_config()         -> dict of JSON-serialisable settings / learned state
    save(directory)      -> list of written files (their total size is the model size)

Models receive the full Session 1 window tensor and choose their own input representation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class MeanRULBaseline:
    """Lower-bound baseline: always predicts the mean of the TRAINING targets."""

    def __init__(self, name="mean_rul"):
        self.name = name
        self.constant_ = None
        self.n_train_targets_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        if y.size == 0:
            raise ValueError("cannot fit the mean baseline on an empty target")
        self.constant_ = float(y.mean())
        self.n_train_targets_ = int(y.size)
        return self

    def predict(self, X):
        self._check_fitted()
        return np.full(len(X), self.constant_, dtype=np.float64)

    def get_config(self):
        return {"type": "mean_rul", "constant": self.constant_,
                "n_train_targets": self.n_train_targets_, "fitted_on": "training targets only"}

    def n_parameters(self):
        return 1

    def save(self, directory):
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "mean_rul_baseline.json"
        path.write_text(json.dumps(self.get_config(), indent=2) + "\n")
        return [path]

    def _check_fitted(self):
        if self.constant_ is None:
            raise RuntimeError("MeanRULBaseline is not fitted")


class LinearRegressionBaseline:
    """Ordinary least squares on the sensor values of the LAST cycle of each window.

    Input window [30 cycles, 14 sensors] -> [14 sensor values]. No temporal summary features
    (mean, std, min, max, slope) are used; those belong to a later session.
    """

    REPRESENTATION = "last_timestep"

    def __init__(self, name="linear_regression", representation=REPRESENTATION):
        if representation != self.REPRESENTATION:
            raise ValueError(f"unsupported representation '{representation}'; "
                             f"only '{self.REPRESENTATION}' is defined")
        self.name = name
        self.estimator_ = None

    @staticmethod
    def represent(X):
        """[n, sequence_length, n_sensors] -> [n, n_sensors]: the final timestep of each window."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected windows of shape [n, sequence_length, n_sensors], got {X.shape}")
        return X[:, -1, :]

    def fit(self, X, y):
        self.estimator_ = LinearRegression().fit(self.represent(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X):
        self._check_fitted()
        return np.asarray(self.estimator_.predict(self.represent(X)), dtype=np.float64)

    def get_config(self):
        config = {"type": "linear_regression", "representation": self.REPRESENTATION,
                  "fit_intercept": True}
        if self.estimator_ is not None:
            config["n_features"] = int(self.estimator_.n_features_in_)
        return config

    def n_parameters(self):
        self._check_fitted()
        return int(self.estimator_.coef_.size + 1)

    def save(self, directory):
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "linear_regression.joblib"
        joblib.dump(self.estimator_, path)
        return [path]

    def _check_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError("LinearRegressionBaseline is not fitted")


def build_baseline(spec):
    """Create a baseline from its config entry ({name, type, ...})."""
    if spec["type"] == "mean_rul":
        return MeanRULBaseline(name=spec["name"])
    if spec["type"] == "linear_regression":
        return LinearRegressionBaseline(
            name=spec["name"], representation=spec.get("representation", LinearRegressionBaseline.REPRESENTATION))
    raise ValueError(f"unknown baseline type '{spec['type']}'")
