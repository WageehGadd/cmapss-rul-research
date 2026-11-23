import numpy as np
import pytest
import torch

from src.convolutional_models import (CausalConv1d, ConvolutionalRegressor, PlainCNNNet, TCNNet, TemporalBlock,
                                      cnn_output_length, cnn_receptive_field, tcn_receptive_field)
from src.evaluation import evaluate_model
from src.recurrent_models import count_parameters, weights_digest

SEQ, SENS = 30, 14
TRAINING = {"learning_rate": 0.01, "batch_size": 32, "max_epochs": 5, "early_stopping_patience": 3,
            "min_delta": 1e-6, "grad_clip": 1.0, "scheduler": {"type": "none"}}


def data(n=96, seed=0, seq=SEQ, sens=SENS):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq, sens)).astype("float32")
    y = (60 + 25 * X[:, -1, 0] + 10 * X[:, :, 1].mean(axis=1)).clip(0, 125).astype("float32")
    return X, y


def model(kind="cnn", **kw):
    X, y = data(seed=1)
    args = dict(channels=8, kernel_size=3, dropout=0.0, sequence_length=SEQ, n_sensors=SENS,
                training=TRAINING, seed=3, device="cpu", validation=(X, y))
    args.update({"depth": 2} if kind == "cnn" else {"dilations": [1, 2]})
    args.update(kw)
    return ConvolutionalRegressor(kind, kind, **args)


# ---- shapes, layout, forward ------------------------------------------------------------------
def test_cnn_forward_shape_and_time_axis_shrinks():
    net = PlainCNNNet(SENS, 8, 5, 3, 0.0, SEQ).eval()
    x = torch.randn(7, SEQ, SENS)
    assert net(x).shape == (7,)
    assert net.features(x.transpose(1, 2)).shape == (7, 8, cnn_output_length(SEQ, 5, 3)) == (7, 8, 18)


def test_tcn_forward_shape_and_time_axis_is_preserved():
    net = TCNNet(SENS, 8, 3, [1, 2, 4], 0.0).eval()
    x = torch.randn(7, SEQ, SENS)
    assert net(x).shape == (7,)
    assert net.features(x.transpose(1, 2)).shape == (7, 8, SEQ)


@pytest.mark.parametrize("net", [PlainCNNNet(SENS, 8, 3, 2, 0.0, SEQ), TCNNet(SENS, 8, 3, [1, 2], 0.0)])
def test_input_is_btf_and_transposed_inside_the_model(net):
    net.eval()
    x = torch.randn(4, SEQ, SENS)
    before = x.clone()
    with torch.no_grad():
        out = net(x)
        by_hand = net.head(net.features(x.transpose(1, 2))[:, :, -1]).squeeze(-1)
    assert torch.allclose(out, by_hand) and torch.equal(x, before)        # input layout untouched
    with pytest.raises(RuntimeError):
        net(torch.randn(4, SENS, SEQ))                                     # [B, F, T] is the wrong layout


# ---- TCN internals ------------------------------------------------------------------------------
def test_residual_projection_only_when_channels_change():
    assert TemporalBlock(14, 8, 3, 1, 0.0).project is not None
    assert TemporalBlock(8, 8, 3, 1, 0.0).project is None
    block = TemporalBlock(8, 8, 3, 2, 0.0).eval()
    x = torch.randn(2, 8, 20)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()                                                      # conv branch = 0 -> output = relu(x)
        assert torch.allclose(block(x), torch.relu(x))
    tcn = TCNNet(SENS, 8, 3, [1, 2, 4], 0.0)
    assert tcn.blocks[0].project is not None and tcn.blocks[1].project is None and tcn.blocks[2].project is None


def test_causal_padding_output_never_depends_on_later_time_steps():
    net = TCNNet(SENS, 8, 3, [1, 2, 4], 0.0).eval()
    x = torch.randn(3, SENS, SEQ)
    changed = x.clone()
    changed[:, :, 20:] = torch.randn(3, SENS, SEQ - 20)                    # rewrite the future after t = 19
    with torch.no_grad():
        a, b = net.features(x), net.features(changed)
    assert torch.allclose(a[:, :, :20], b[:, :, :20], atol=1e-6)           # earlier outputs unchanged
    assert not torch.allclose(a[:, :, 20:], b[:, :, 20:])
    conv = CausalConv1d(2, 2, 3, 4)
    assert conv.pad == 8 and conv(torch.randn(1, 2, 10)).shape == (1, 2, 10)


def test_receptive_field_larger_than_the_window_still_runs_on_30_cycles():
    net = TCNNet(SENS, 4, 5, [1, 2, 4, 8], 0.0).eval()                    # RF 121 > 30
    assert tcn_receptive_field(5, [1, 2, 4, 8]) > SEQ
    assert net(torch.randn(3, SEQ, SENS)).shape == (3,)                   # the input is never extended


def test_receptive_field_formulas():
    assert tcn_receptive_field(3, [1, 2, 4]) == 29
    assert tcn_receptive_field(3, [1, 2, 4, 8]) == 61
    assert tcn_receptive_field(5, [1, 2, 4]) == 57 and tcn_receptive_field(5, [1, 2, 4, 8]) == 121
    assert cnn_receptive_field(3, 2) == 5 and cnn_receptive_field(5, 3) == 13
    assert model("tcn", dilations=[1, 2, 4]).receptive_field == 29 and model("cnn", depth=3).receptive_field == 7
    # an empirical check for the CNN readout: only the last RF cycles influence the output
    net = PlainCNNNet(SENS, 4, 3, 3, 0.0, SEQ).eval()
    x = torch.randn(2, SEQ, SENS)
    y = x.clone()
    y[:, : SEQ - 7] = torch.randn(2, SEQ - 7, SENS)
    with torch.no_grad():
        assert torch.allclose(net(x), net(y), atol=1e-6)


def test_receptive_field_is_checked_against_a_too_short_sequence():
    with pytest.raises(ValueError, match="do not fit"):
        PlainCNNNet(SENS, 4, 5, 8, 0.0, 30)
    with pytest.raises(ValueError, match="longer than the sequence"):
        model("cnn", kernel_size=5, depth=8)
    with pytest.raises(ValueError, match="dilations"):
        TCNNet(SENS, 4, 3, [], 0.0)
    with pytest.raises(ValueError, match="needs"):
        ConvolutionalRegressor("x", "tcn", channels=4, kernel_size=3, dropout=0.0, sequence_length=SEQ,
                               n_sensors=SENS, training=TRAINING, seed=1)


# ---- parameters, determinism ----------------------------------------------------------------------
def test_parameter_counts_match_closed_forms():
    c, k, f = 8, 3, SENS
    cnn = (f * c * k + c) + (c * c * k + c) + (c + 1)                      # 2 layers + head
    assert count_parameters(PlainCNNNet(f, c, k, 2, 0.0, SEQ)) == cnn == model("cnn").n_parameters()
    block1 = (f * c * k + c) + (c * c * k + c) + (f * c + c)               # two convs + 1x1 projection
    block2 = 2 * (c * c * k + c)
    assert model("tcn").n_parameters() == block1 + block2 + (c + 1)


def test_initialisation_and_training_are_deterministic():
    a, b, c = model("tcn")._build(), model("tcn")._build(), model("tcn", seed=4)._build()
    assert weights_digest(a.state_dict()) == weights_digest(b.state_dict()) != weights_digest(c.state_dict())
    X, y = data()
    m1, m2 = model("cnn").fit(X, y), model("cnn").fit(X, y)
    assert m1.weights_sha256() == m2.weights_sha256()
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# ---- fit / predict / save / load ---------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["cnn", "tcn"])
def test_save_load_roundtrip_and_config(kind, tmp_path):
    X, y = data()
    m = model(kind).fit(X, y)
    files = m.save(tmp_path)
    assert {f.suffix for f in files} == {".pt", ".json"} and (tmp_path / "training_history.json").exists()
    loaded = ConvolutionalRegressor.load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    assert loaded.weights_sha256() == m.weights_sha256() and loaded.kind == kind
    cfg = m.get_config()
    assert cfg["type"] == kind and cfg["receptive_field"] == m.receptive_field and cfg["n_parameters"] == m.n_parameters()
    assert loaded.get_config()["best_epoch"] == cfg["best_epoch"]


@pytest.mark.parametrize("kind", ["cnn", "tcn"])
def test_session2_evaluator_compatibility(kind, tmp_path):
    X, y = data()
    m = model(kind).fit(X, y)
    result = evaluate_model(m, {"validation": (X, y)}, X, target_type="capped", train_time_s=1.0,
                            timing_config={"warmup_runs": 1, "timed_runs": 2, "batch_size": 16},
                            artifact_dir=tmp_path / "m")
    assert result.model_name == kind and result.n_parameters == m.n_parameters()
    assert set(result.metrics["validation"]) == {"rmse", "mae", "r2", "nasa_score"}
    assert result.predictions["validation"].shape == (len(X),) and result.model_size_bytes > 0


@pytest.mark.parametrize("kind", ["cnn", "tcn"])
def test_tiny_batch_overfit_and_finite_predictions(kind):
    X, y = data(32)
    m = model(kind, channels=16, validation=(X, y),
              training={**TRAINING, "max_epochs": 200, "batch_size": 32, "early_stopping_patience": 200})
    m.fit(X, y)
    first, last = m.history_[0]["train_loss"], m.history_[-1]["train_loss"]
    assert last < 0.25 * first
    pred = m.predict(X)
    assert pred.dtype == np.float64 and np.isfinite(pred).all() and pred.shape == (32,)


def test_non_finite_inputs_are_flagged_not_hidden():
    X, y = data()
    X[0, 0, 0] = np.nan
    m = model("tcn").fit(X, y)
    assert m.fit_summary_["stop_reason"] == "non_finite_loss_or_gradient" and not m.fit_summary_["stable"]
    fitted = model("cnn").fit(*data())
    with pytest.raises(ValueError, match="expected windows"):
        fitted.predict(np.zeros((4, SEQ, SENS + 1), dtype="float32"))         # wrong sensor count
    with pytest.raises(ValueError, match="validation set is required"):
        model(validation=None).fit(*data())


def test_session4_protocol_is_inherited_not_reimplemented():
    from src.recurrent_models import RecurrentRegressor
    for name in ("fit", "predict", "n_parameters", "weights_sha256", "_loss"):
        assert getattr(ConvolutionalRegressor, name) is getattr(RecurrentRegressor, name)
    X, y = data()
    m = model("cnn", training={**TRAINING, "max_epochs": 20, "early_stopping_patience": 2, "min_delta": 10.0}).fit(X, y)
    assert m.fit_summary_["stop_reason"] == "early_stopping" and m.fit_summary_["best_epoch"] == 1
