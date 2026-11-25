import numpy as np
import pytest
import torch

from src.evaluation import evaluate_model
from src.recurrent_models import RecurrentRegressor, count_parameters, weights_digest
from src.transformer_models import (TransformerNet, TransformerRegressor, build_from_definition,
                                    require_positive_int, sinusoidal_encoding, validate_architecture)

T, F = 30, 14
TRAINING = {"optimizer": "adam", "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 32, "max_epochs": 5,
            "early_stopping_patience": 3, "min_delta": 1e-6, "grad_clip": 1.0, "scheduler": {"type": "none"},
            "target_scale": 125}


def data(n=96, seed=0, f=F):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, f)).astype("float32")
    y = (60 + 25 * X[:, -1, 0] + 10 * X[:, :, 1].mean(axis=1)).clip(0, 125).astype("float32")
    return X, y


def net(**kw):
    args = dict(n_sensors=F, sequence_length=T, d_model=16, num_heads=2, num_layers=2, dropout=0.0)
    args.update(kw)
    torch.manual_seed(0)
    return TransformerNet(**args)


def model(**kw):
    X, y = data(seed=1)
    args = dict(d_model=16, num_heads=2, num_layers=1, dropout=0.0, sequence_length=T, n_sensors=F,
                training=TRAINING, seed=3, device="cpu", validation=(X, y))
    args.update(kw)
    return TransformerRegressor("transformer", **args)


# ---- data flow ---------------------------------------------------------------------------------
def test_batch_time_feature_convention_projection_and_shapes():
    n = net().eval()
    x = torch.randn(5, T, F)
    assert (n.input_projection.in_features, n.input_projection.out_features) == (F, 16)
    assert n.embed(x).shape == (5, T, 16) and n.encode(x).shape == (5, T, 16)
    assert n.pool(n.encode(x)).shape == (5, 16) and n(x).shape == (5,)
    assert all(layer.self_attn.batch_first and layer.norm_first for layer in n.layers)
    with pytest.raises(ValueError, match="expected"):
        n(torch.randn(5, T + 1, F))
    with pytest.raises(RuntimeError):
        n(torch.randn(5, F, T))                                    # [B, F, T] is the wrong layout


def test_positional_encoding_is_deterministic_and_actually_used():
    table = sinusoidal_encoding(T, 16)
    assert table.shape == (T, 16) and torch.equal(table, sinusoidal_encoding(T, 16))
    assert table[0, 0] == 0 and table[0, 1] == 1 and len({tuple(r.tolist()) for r in table}) == T
    n = net().eval()
    x = torch.randn(2, T, F)
    assert torch.allclose(n.embed(x), n.input_projection(x) + table)
    permuted = x[:, torch.randperm(T)]
    mean_pool = net(pooling="mean").eval()
    with torch.no_grad():
        # order matters, even for mean pooling ...
        assert not torch.allclose(mean_pool(x), mean_pool(permuted), atol=1e-5)
        # ... and only because of the positional encoding
        mean_pool.positions.zero_()
        assert torch.allclose(mean_pool(x), mean_pool(permuted), atol=1e-5)


def test_attention_shapes_are_normalised_and_bidirectional():
    n = net(num_heads=4, num_layers=2)
    weights = n.attention_weights(torch.randn(3, T, F))
    assert len(weights) == 2 and all(w.shape == (3, 4, T, T) for w in weights)
    for w in weights:
        assert torch.allclose(w.sum(-1), torch.ones(3, 4, T), atol=1e-5)
    assert (weights[0][:, :, 0, -1] > 0).all()                     # no causal mask: cycle 0 attends to cycle 29


def test_attention_mask_is_honoured_when_supplied():
    n = net(num_layers=2).eval()
    x = torch.randn(2, T, F)
    causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)       # True = blocked
    for w in n.attention_weights(x, attn_mask=causal):
        assert float(w[..., causal].abs().max()) == 0.0
    changed = x.clone()
    changed[:, 20:] = torch.randn(2, 10, F)
    with torch.no_grad():
        assert torch.allclose(n.encode(x, causal)[:, :20], n.encode(changed, causal)[:, :20], atol=1e-5)
        assert not torch.allclose(n.encode(x)[:, :20], n.encode(changed)[:, :20], atol=1e-5)  # unmasked: all cycles


def test_pooling_behaviour():
    h = torch.randn(4, T, 16)
    assert torch.equal(net(pooling="last").pool(h), h[:, -1])
    assert torch.allclose(net(pooling="mean").pool(h), h.mean(1))
    assert count_parameters(net(pooling="last")) == count_parameters(net(pooling="mean"))


def test_encoder_layers_are_independently_initialised():
    n = net(num_layers=2)
    assert not torch.equal(n.layers[0].linear1.weight, n.layers[1].linear1.weight)


@pytest.mark.parametrize("batch", [1, 7])
def test_variable_batch_sizes_including_one(batch):
    X, y = data()
    m = model().fit(X, y)
    pred = m.predict(X[:batch])
    assert pred.shape == (batch,) and np.isfinite(pred).all()
    np.testing.assert_allclose(pred, m.predict(X)[:batch], atol=1e-3)


# ---- configuration -----------------------------------------------------------------------------
def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        validate_architecture(32, 3, 1, 2, 0.1, "last")
    with pytest.raises(ValueError, match="pooling"):
        net(pooling="max")
    with pytest.raises(ValueError, match="dropout"):
        net(dropout=1.0)
    with pytest.raises(ValueError, match="activation"):
        net(activation="tanh")
    with pytest.raises(ValueError, match="divisible"):
        model(d_model=10, num_heads=4)
    net(d_model=64, num_heads=4)                                     # valid combination


def test_parameter_count_matches_closed_form():
    d, h, ff, L = 16, 2, 32, 2
    attention = 4 * (d * d + d)
    feed_forward = (d * ff + ff) + (ff * d + d)
    norms = 2 * 2 * d
    expected = (F * d + d) + L * (attention + feed_forward + norms) + 2 * d + (d + 1)
    assert count_parameters(net()) == expected
    assert count_parameters(net(num_heads=4)) == expected               # heads do not change the size
    assert model(num_layers=2).n_parameters() == expected


# ---- training ----------------------------------------------------------------------------------
def test_gradients_are_finite_and_training_protocol_is_inherited():
    n = net()
    n(torch.randn(8, T, F)).pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in n.parameters())
    for name in ("fit", "predict", "n_parameters", "weights_sha256", "_loss"):
        assert getattr(TransformerRegressor, name) is getattr(RecurrentRegressor, name)


def test_tiny_batch_overfit_and_loss_decrease():
    X, y = data(32)
    m = model(validation=(X, y), training={**TRAINING, "max_epochs": 200, "early_stopping_patience": 200})
    m.fit(X, y)
    assert m.history_[-1]["train_loss"] < 0.25 * m.history_[0]["train_loss"]
    assert np.isfinite(m.predict(X)).all()


def test_target_scaling_and_inverse():
    X, y = data()
    m = model().fit(X, y)
    with torch.no_grad():
        raw = m.net_.eval()(torch.from_numpy(X)).numpy().astype(np.float64)
    np.testing.assert_allclose(m.predict(X), raw * 125.0)
    assert np.abs(raw).max() < 3 and abs(m.predict(X).mean() - y.mean()) < 40


def test_deterministic_initialisation_and_training():
    a, b, c = model()._build(), model()._build(), model(seed=4)._build()
    assert weights_digest(a.state_dict()) == weights_digest(b.state_dict()) != weights_digest(c.state_dict())
    X, y = data()
    m1, m2 = model(dropout=0.1).fit(X, y), model(dropout=0.1).fit(X, y)
    assert m1.weights_sha256() == m2.weights_sha256()
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_non_finite_training_is_flagged():
    X, y = data()
    X[0, 0, 0] = np.nan
    m = model().fit(X, y)
    assert not m.fit_summary_["stable"] and m.fit_summary_["stop_reason"] == "non_finite_loss_or_gradient"


# ---- persistence, engine, Session 7 ------------------------------------------------------------------
@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_save_load_consistency(pooling, tmp_path):
    X, y = data()
    m = model(pooling=pooling, dropout=0.1).fit(X, y)
    files = m.save(tmp_path)
    assert {f.name for f in files} == {"transformer.pt", "transformer_config.json"}
    loaded = TransformerRegressor.load(tmp_path)
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    assert loaded.weights_sha256() == m.weights_sha256() and loaded.pooling == pooling
    cfg = m.get_config()
    assert cfg["type"] == "transformer" and cfg["ff_dim"] == 32 and cfg["attention_mask"] == "none"


def test_session2_evaluator_integration(tmp_path):
    X, y = data()
    m = model().fit(X, y)
    result = evaluate_model(m, {"validation": (X, y)}, X, target_type="capped", train_time_s=1.0,
                            timing_config={"warmup_runs": 1, "timed_runs": 2, "batch_size": 16},
                            artifact_dir=tmp_path / "m")
    assert result.model_name == "transformer" and result.n_parameters == m.n_parameters()
    assert set(result.metrics["validation"]) == {"rmse", "mae", "r2", "nasa_score"}
    assert result.model_size_bytes > 0


@pytest.mark.parametrize("sensors", [14, 10, 7, 4, 2])
def test_frozen_definition_rebuilds_for_reduced_sensor_counts(sensors):
    base = model(d_model=32, num_heads=4, num_layers=2, dropout=0.1, pooling="mean")
    definition = {"architecture": base.architecture(), "training": {**TRAINING, "seed": 3, "device": "cpu"}}
    X, y = data(f=sensors)
    rebuilt = build_from_definition(definition, n_sensors=sensors, validation=(X, y))
    for key in ("d_model", "num_heads", "num_layers", "ff_multiplier", "dropout", "pooling", "activation",
                "sequence_length"):
        assert getattr(rebuilt, key) == getattr(base, key)
    assert rebuilt.training == TRAINING and rebuilt.seed == 3
    assert rebuilt._build().input_projection.in_features == sensors
    assert base.n_parameters() - rebuilt.n_parameters() == (F - sensors) * 32   # only the projection changes
    pred = rebuilt.fit(X, y).predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


@pytest.mark.parametrize("field,value", [("num_heads", 4.5), ("num_heads", 4.0), ("num_heads", True),
                                         ("d_model", 64.5), ("d_model", "64"), ("num_layers", 1.5),
                                         ("num_layers", False), ("ff_multiplier", 2.5), ("sequence_length", 30.5),
                                         ("n_sensors", 14.2), ("n_sensors", True), ("n_sensors", 0)])
def test_non_integer_architecture_values_are_rejected_not_truncated(field, value):
    args = {"d_model": 64, "num_heads": 4, "num_layers": 2, "ff_multiplier": 2, "sequence_length": T, "n_sensors": F}
    args[field] = value
    with pytest.raises(ValueError, match=field):
        validate_architecture(dropout=0.1, pooling="last", **args)
    with pytest.raises(ValueError, match=field):
        TransformerRegressor("x", training=TRAINING, seed=1, **args)
    with pytest.raises(ValueError, match=field):
        TransformerNet(**args)


def test_integer_like_values_are_accepted_and_bad_dropout_rejected():
    validate_architecture(np.int64(64), np.int32(4), 2, 2, 0.0, "last", sequence_length=np.int64(30), n_sensors=2)
    assert require_positive_int("n", np.int64(7)) == 7
    for bad in (False, True, "0.1", 1.0, -0.1):
        with pytest.raises(ValueError, match="dropout"):
            validate_architecture(64, 4, 2, 2, bad, "last")
    definition = {"architecture": model().architecture(), "training": {**TRAINING, "seed": 3, "device": "cpu"}}
    with pytest.raises(ValueError, match="n_sensors"):
        build_from_definition(definition, n_sensors=7.5)
