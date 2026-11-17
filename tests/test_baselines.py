import json

import joblib
import numpy as np
import pytest

from src.baselines import LinearRegressionBaseline, MeanRULBaseline, build_baseline


def windows(n, seq_len=30, n_sensors=14, seed=0):
    return np.random.default_rng(seed).normal(size=(n, seq_len, n_sensors)).astype("float32")


def test_mean_baseline_uses_training_targets_only():
    model = MeanRULBaseline().fit(windows(4), [10.0, 20.0, 30.0, 40.0])
    assert model.constant_ == 25.0
    # Predicting on other data (whatever its labels) never changes the learned constant.
    pred = model.predict(windows(7, seed=1))
    assert pred.shape == (7,)
    assert (pred == 25.0).all()
    assert model.constant_ == 25.0
    assert model.n_train_targets_ == 4


def test_mean_baseline_requires_fit_and_nonempty_target():
    with pytest.raises(RuntimeError):
        MeanRULBaseline().predict(windows(2))
    with pytest.raises(ValueError):
        MeanRULBaseline().fit(windows(0), [])


def test_mean_baseline_saved_state(tmp_path):
    model = MeanRULBaseline().fit(windows(3), [1.0, 2.0, 6.0])
    (path,) = model.save(tmp_path)
    state = json.loads(path.read_text())
    assert state["constant"] == 3.0
    assert state["fitted_on"] == "training targets only"


def test_linear_representation_is_last_timestep():
    X = windows(5)
    rep = LinearRegressionBaseline.represent(X)
    assert rep.shape == (5, 14)
    np.testing.assert_array_equal(rep, X[:, -1, :])


def test_linear_representation_shapes_match_session1_windows():
    for n in (14459, 3272, 100):
        assert LinearRegressionBaseline.represent(np.zeros((n, 30, 14), dtype="float32")).shape == (n, 14)


def test_linear_representation_rejects_wrong_rank():
    with pytest.raises(ValueError):
        LinearRegressionBaseline.represent(np.zeros((5, 14)))


def test_linear_ignores_earlier_cycles_and_fits_exact_line():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 30, 14))
    y = 2.0 * X[:, -1, 0] - 3.0 * X[:, -1, 5] + 40.0
    model = LinearRegressionBaseline().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-9)
    X_changed = X.copy()
    X_changed[:, :-1, :] += 100.0               # alter every cycle except the last
    np.testing.assert_allclose(model.predict(X_changed), model.predict(X))
    assert model.get_config()["n_features"] == 14
    assert model.n_parameters() == 15


def test_linear_saved_model_round_trips(tmp_path):
    X = windows(50)
    y = np.linspace(1, 100, 50)
    model = LinearRegressionBaseline().fit(X, y)
    (path,) = model.save(tmp_path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict(LinearRegressionBaseline.represent(X)), model.predict(X))


def test_linear_rejects_other_representations():
    with pytest.raises(ValueError):
        LinearRegressionBaseline(representation="window_mean")


def test_build_baseline_from_config_entries():
    assert isinstance(build_baseline({"name": "m", "type": "mean_rul"}), MeanRULBaseline)
    lin = build_baseline({"name": "l", "type": "linear_regression", "representation": "last_timestep"})
    assert isinstance(lin, LinearRegressionBaseline) and lin.name == "l"
    with pytest.raises(ValueError):
        build_baseline({"name": "x", "type": "random_forest"})
