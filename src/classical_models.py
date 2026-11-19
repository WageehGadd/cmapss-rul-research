"""Classical (non-neural) regressors that plug into the Session 2 benchmark interface.

Each wrapper takes the Session 1 window tensor, converts it with the window-to-tabular
transformer, and fits a scikit-learn estimator on the resulting [n, 84] matrix. The window ->
features step is part of `predict`, so measured inference latency includes it.

Families
--------
random_forest      sklearn RandomForestRegressor
gradient_boosting  sklearn HistGradientBoostingRegressor (histogram gradient boosting).
                   XGBoost is not part of the project environment; this is the documented
                   boosting alternative.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from .tabular_features import WindowFeatureTransformer

FAMILIES = ("random_forest", "gradient_boosting")


class TabularRegressor:
    """Base wrapper: windows -> tabular features -> estimator."""

    family = None

    def __init__(self, name, transformer, params, seed, n_jobs=-1, predict_n_jobs=1):
        self.name = name
        self.transformer = transformer
        self.params = dict(params)
        self.seed = int(seed)
        self.n_jobs = n_jobs
        self.predict_n_jobs = predict_n_jobs
        self.estimator_ = None

    def _make_estimator(self):
        raise NotImplementedError

    def fit(self, X, y):
        self.fit_features(self.transformer.transform(X), y)
        return self

    def fit_features(self, F, y):
        self.estimator_ = self._make_estimator().fit(F, np.asarray(y, dtype=np.float64))
        self._after_fit()
        return self

    def _after_fit(self):
        pass

    def predict(self, X):
        return self.predict_features(self.transformer.transform(X))

    def predict_features(self, F):
        self._check_fitted()
        return np.asarray(self.estimator_.predict(F), dtype=np.float64)

    def get_config(self):
        return {"type": self.family, "params": dict(self.params), "seed": self.seed,
                "n_features": self.transformer.n_features, "features": list(self.transformer.features),
                "n_parameters": self.n_parameters() if self.estimator_ is not None else None}

    def n_parameters(self):
        """Total number of tree nodes (the model's complexity), or None if unavailable."""
        raise NotImplementedError

    def save(self, directory):
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{self.family}.joblib"
        joblib.dump(self.estimator_, model_path)
        names_path = directory / "feature_names.json"
        names_path.write_text(json.dumps(self.transformer.feature_names, indent=2) + "\n")
        return [model_path, names_path]

    def _check_fitted(self):
        if self.estimator_ is None:
            raise RuntimeError(f"{self.name} is not fitted")


class RandomForestModel(TabularRegressor):
    family = "random_forest"

    def _make_estimator(self):
        return RandomForestRegressor(random_state=self.seed, n_jobs=self.n_jobs, **self.params)

    def _after_fit(self):
        # Training uses all cores; single-batch inference is faster (and timed fairly) on one.
        self.estimator_.n_jobs = self.predict_n_jobs

    def n_parameters(self):
        self._check_fitted()
        return int(sum(tree.tree_.node_count for tree in self.estimator_.estimators_))


class GradientBoostingModel(TabularRegressor):
    family = "gradient_boosting"

    def _make_estimator(self):
        # early_stopping=False: the number of iterations is a searched hyperparameter and
        # no internal validation split is drawn from the training data.
        return HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params)

    def n_parameters(self):
        self._check_fitted()
        try:
            return int(sum(len(p.nodes) for stage in self.estimator_._predictors for p in stage))
        except AttributeError:      # private scikit-learn attribute; complexity is optional
            return None


_BUILDERS = {"random_forest": RandomForestModel, "gradient_boosting": GradientBoostingModel}


def build_model(family, params, *, sensors, features, sequence_length, seed, name=None,
                n_jobs=-1, predict_n_jobs=1):
    if family not in _BUILDERS:
        raise ValueError(f"unknown model family '{family}', expected one of {FAMILIES}")
    transformer = WindowFeatureTransformer(sensors, features, sequence_length)
    return _BUILDERS[family](name or family, transformer, params, seed,
                             n_jobs=n_jobs, predict_n_jobs=predict_n_jobs)
