import numpy as np
import pytest
import torch

import src.recurrent_models as rm
from src.evaluation import evaluate_model
from src.recurrent_models import RecurrentNet, RecurrentRegressor, count_parameters, resolve_device, weights_digest

SEQ, SENS = 12, 5
TRAINING = {"learning_rate": 0.01, "batch_size": 32, "max_epochs": 6, "early_stopping_patience": 3,
            "min_delta": 1e-6, "grad_clip": 1.0, "scheduler": {"type": "none"}}


def data(n=96, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, SEQ, SENS)).astype("float32")
    y = (60 + 25 * X[:, -1, 0] + 10 * X[:, :, 1].mean(axis=1)).clip(0, 125).astype("float32")
    return X, y


def model(cell="lstm", training=None, validation=True, **kw):
    X, y = data(seed=1)
    args = dict(hidden_size=8, num_layers=1, dropout=0.0, sequence_length=SEQ, n_sensors=SENS,
                training={**TRAINING, **(training or {})}, seed=3, device="cpu")
    args.update(kw)
    return RecurrentRegressor(cell, cell, validation=(X, y) if validation else None, **args)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_input_output_shape_and_scalar_per_sample(cell):
    net = RecurrentNet(cell, SENS, 8, 2, 0.1).eval()
    out = net(torch.zeros(7, SEQ, SENS))
    assert out.shape == (7,)
    assert net.rnn.batch_first is True


def test_batch_first_layout_and_last_time_step_is_used():
    net = RecurrentNet("lstm", SENS, 8, 1, 0.0).eval()
    x = torch.randn(4, SEQ, SENS)
    with torch.no_grad():
        full = net(x)
        by_hand = net.head(net.rnn(x)[0][:, -1, :]).squeeze(-1)
        shifted = net(torch.cat([torch.randn(4, 3, SENS), x[:, 3:]], dim=1))
    assert torch.allclose(full, by_hand)
    assert not torch.allclose(full, shifted)             # earlier time steps matter: it is recurrent


def test_parameter_counts_match_the_closed_form():
    h, s = 8, SENS
    lstm = 4 * (h * (s + h) + 2 * h) + (h + 1)
    gru = 3 * (h * (s + h) + 2 * h) + (h + 1)
    assert count_parameters(RecurrentNet("lstm", s, h, 1, 0.0)) == lstm
    assert count_parameters(RecurrentNet("gru", s, h, 1, 0.0)) == gru
    two = count_parameters(RecurrentNet("lstm", s, h, 2, 0.0))
    assert two == lstm + 4 * (h * (h + h) + 2 * h)
    assert model("gru").n_parameters() == gru                    # also before fitting


def test_initialisation_is_deterministic_under_seed():
    a, b, c = model()._build(), model()._build(), model(seed=4)._build()
    assert weights_digest(a.state_dict()) == weights_digest(b.state_dict())
    assert weights_digest(a.state_dict()) != weights_digest(c.state_dict())


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_benchmark_interface_and_session2_evaluator(cell, tmp_path):
    X, y = data()
    m = model(cell).fit(X, y)
    result = evaluate_model(m, {"validation": (X, y)}, X, target_type="capped", train_time_s=1.0,
                            timing_config={"warmup_runs": 1, "timed_runs": 2, "batch_size": 16},
                            artifact_dir=tmp_path / "m")
    assert result.model_name == cell and result.n_parameters == m.n_parameters()
    assert set(result.metrics["validation"]) == {"rmse", "mae", "r2", "nasa_score"}
    assert np.isfinite(result.predictions["validation"]).all()
    assert result.model_size_bytes > 0
    assert m.get_config()["type"] == cell and m.get_config()["best_epoch"] >= 1


def test_predictions_are_finite_and_in_cycles():
    X, y = data()
    pred = model(training={"max_epochs": 10}).fit(X, y).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()
    assert pred.dtype == np.float64 and abs(pred.mean() - y.mean()) < 40       # RUL scale, not 0-1


def test_checkpoint_save_load_roundtrip(tmp_path):
    X, y = data()
    m = model("gru").fit(X, y)
    files = m.save(tmp_path)
    assert {f.suffix for f in files} == {".pt", ".json"} and (tmp_path / "training_history.json").exists()
    loaded = RecurrentRegressor.load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    assert loaded.weights_sha256() == m.weights_sha256()
    assert loaded.get_config()["best_epoch"] == m.get_config()["best_epoch"]


def test_tiny_batch_training_reduces_loss():
    X, y = data(32)
    m = RecurrentRegressor("t", "lstm", hidden_size=16, num_layers=1, dropout=0.0, sequence_length=SEQ,
                           n_sensors=SENS, seed=1, device="cpu", validation=(X, y),
                           training={**TRAINING, "max_epochs": 150, "batch_size": 32, "early_stopping_patience": 150})
    m.fit(X, y)
    first, last = m.history_[0]["train_loss"], m.history_[-1]["train_loss"]
    assert last < 0.25 * first


def test_early_stopping_on_validation_and_best_checkpoint_restored():
    X, y = data()
    # nothing counts as an improvement after epoch 1 (min_delta is huge) -> stops after `patience` more epochs
    m = model(training={"max_epochs": 20, "early_stopping_patience": 2, "min_delta": 10.0}).fit(X, y)
    s = m.fit_summary_
    assert s["stop_reason"] == "early_stopping" and s["best_epoch"] == 1 and s["epochs_run"] == 3
    # restoration: the final weights score the best validation loss recorded in the history
    m2 = model(training={"learning_rate": 0.05, "max_epochs": 12, "early_stopping_patience": 12}).fit(X, y)
    best = min(e["val_loss"] for e in m2.history_)
    pred = m2.predict(m2.validation[0])
    restored = float(np.mean(((pred - m2.validation[1]) / 125.0) ** 2))
    assert restored == pytest.approx(best, rel=1e-4) and m2.fit_summary_["best_epoch"] == \
        min(m2.history_, key=lambda e: e["val_loss"])["epoch"]


def test_max_epochs_stop_reason_and_validation_is_required():
    X, y = data()
    assert model(training={"max_epochs": 2, "early_stopping_patience": 5}).fit(X, y).fit_summary_["stop_reason"] == "max_epochs"
    with pytest.raises(ValueError, match="validation set is required"):
        model(validation=False).fit(X, y)


def test_gradient_clipping_is_active_and_recorded():
    X, y = data()
    tight = model(training={"grad_clip": 1e-4}).fit(X, y)
    loose = model(training={"grad_clip": 1e6}).fit(X, y)
    assert all(e["clip_fraction"] == 1.0 for e in tight.history_)
    assert all(e["clip_fraction"] == 0.0 for e in loose.history_)
    assert all(e["grad_norm_max"] > 1e-4 for e in tight.history_)       # the pre-clip norm is what is recorded
    for key in ("epoch", "train_loss", "val_loss", "lr", "grad_norm_mean", "grad_norm_max", "clip_fraction",
                "epoch_time_s"):
        assert key in tight.history_[0]


def test_non_finite_training_is_flagged_not_hidden():
    X, y = data()
    X_bad = X.copy()
    X_bad[0, 0, 0] = np.nan
    m = model().fit(X_bad, y)
    assert m.fit_summary_["stop_reason"] == "non_finite_loss_or_gradient" and not m.fit_summary_["stable"]


def test_fit_is_reproducible_on_cpu():
    X, y = data()
    a, b = model("gru").fit(X, y), model("gru").fit(X, y)
    assert a.weights_sha256() == b.weights_sha256()
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert [e["val_loss"] for e in a.history_] == [e["val_loss"] for e in b.history_]


def test_scheduler_reacts_to_validation_loss_only():
    X, y = data()
    sched = {"type": "reduce_on_plateau", "factor": 0.5, "patience": 0, "min_lr": 1e-6}
    m = model(training={"scheduler": sched, "max_epochs": 12, "early_stopping_patience": 12,
                        "learning_rate": 0.05}).fit(X, y)
    lrs = [e["lr"] for e in m.history_]
    assert lrs[0] == 0.05 and min(lrs) < 0.05 and all(b <= a for a, b in zip(lrs, lrs[1:]))


def test_wrong_input_shape_is_rejected_and_no_tabular_features_are_used():
    m = model()
    with pytest.raises(ValueError, match="expected windows"):
        m.fit(np.zeros((10, 84), dtype="float32"), np.zeros(10))              # Session 3 tabular layout
    with pytest.raises(ValueError, match="expected windows"):
        m.fit(np.zeros((10, SEQ + 1, SENS), dtype="float32"), np.zeros(10))   # wrong sequence length
    import pathlib
    for name in ("recurrent_models.py", "recurrent_study.py", "sequence_data.py", "run_session4.py"):
        text = (pathlib.Path(rm.__file__).parent / name).read_text()
        assert "tabular_features" not in text and "WindowFeatureTransformer" not in text


def test_device_resolution():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "mps", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="not available"):
            resolve_device("cuda")
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")
