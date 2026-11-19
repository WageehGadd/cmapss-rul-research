import numpy as np
import pytest

from src.tabular_features import FEATURES, WindowFeatureTransformer, ols_slope

SENSORS = ["a", "b", "c"]


def transformer(T=30):
    return WindowFeatureTransformer(SENSORS, FEATURES, T)


def windows(n=5, T=30, seed=0):
    return np.random.default_rng(seed).normal(size=(n, T, len(SENSORS))).astype("float32")


def test_output_shape_and_84_features_for_14_sensors():
    t = WindowFeatureTransformer([f"s{i}" for i in range(14)], FEATURES, 30)
    out = t.transform(np.zeros((7, 30, 14), dtype="float32"))
    assert out.shape == (7, 84) and t.n_features == 84


def test_feature_names_and_order_are_sensor_major():
    t = transformer()
    assert t.feature_names[:6] == [f"a__{f}" for f in FEATURES]
    assert t.feature_names[6] == "b__last" and t.feature_names[-1] == "c__slope"
    assert len(t.feature_names) == len(set(t.feature_names)) == 18
    assert t.sensor_to_columns() == {"a": list(range(0, 6)), "b": list(range(6, 12)), "c": list(range(12, 18))}
    assert t.feature_sensors[:7] == ["a"] * 6 + ["b"]
    assert t.sensor_to_features()["c"][0] == "c__last"


def test_statistics_match_numpy_per_sensor():
    w = windows()
    out = transformer().transform(w)
    w64 = w.astype(np.float64)
    for s in range(3):
        block = out[:, s * 6:(s + 1) * 6]
        np.testing.assert_allclose(block[:, 0], w64[:, -1, s])
        np.testing.assert_allclose(block[:, 1], w64[:, :, s].mean(axis=1))
        np.testing.assert_allclose(block[:, 2], w64[:, :, s].std(axis=1))      # population std
        np.testing.assert_allclose(block[:, 3], w64[:, :, s].min(axis=1))
        np.testing.assert_allclose(block[:, 4], w64[:, :, s].max(axis=1))


def test_hand_checked_values_on_a_tiny_window():
    w = np.array([[[1.0], [2.0], [4.0]]])                    # T = 3, one sensor
    out = WindowFeatureTransformer(["x"], FEATURES, 3).transform(w)[0]
    last, mean, std, mn, mx, slope = out
    assert (last, mn, mx) == (4.0, 1.0, 4.0)
    assert mean == pytest.approx(7 / 3)
    assert std == pytest.approx(np.sqrt(((1 - 7 / 3) ** 2 + (2 - 7 / 3) ** 2 + (4 - 7 / 3) ** 2) / 3))
    assert slope == pytest.approx(1.5)                        # OLS slope of (1, 2, 4) over (0, 1, 2)


def test_slope_on_known_linear_sequence():
    t = np.arange(30)
    w = np.stack([2.5 * t + 1.0, -0.4 * t + 7.0, np.full(30, 3.0)], axis=1)[None]   # [1, 30, 3]
    slope = transformer().transform(w)[0, 5::6]
    np.testing.assert_allclose(slope, [2.5, -0.4, 0.0], atol=1e-12)


def test_slope_matches_numpy_polyfit():
    w = windows(4)
    expected = np.array([[np.polyfit(np.arange(30), w[i, :, s].astype(np.float64), 1)[0]
                          for s in range(3)] for i in range(4)])
    np.testing.assert_allclose(ols_slope(w), expected, atol=1e-10)


def test_transform_is_deterministic():
    w = windows()
    np.testing.assert_array_equal(transformer().transform(w), transformer().transform(w))


def test_transform_does_not_mutate_input():
    w = windows()
    before = w.copy()
    transformer().transform(w)
    np.testing.assert_array_equal(w, before)
    assert w.dtype == np.float32


def test_windows_are_independent_rows():
    w = windows(6)
    full = transformer().transform(w)
    np.testing.assert_array_equal(transformer().transform(w[2:3]), full[2:3])    # no cross-window state
    np.testing.assert_array_equal(transformer().transform(w[::-1]), full[::-1])


def test_rejects_wrong_shapes_and_bad_config():
    with pytest.raises(ValueError, match="expected windows"):
        transformer().transform(np.zeros((4, 30, 5)))
    with pytest.raises(ValueError, match="30 cycles"):
        transformer().transform(np.zeros((4, 20, 3)))
    with pytest.raises(ValueError, match="features"):
        WindowFeatureTransformer(SENSORS, ["last", "median"])
    with pytest.raises(ValueError, match="unique"):
        WindowFeatureTransformer(["a", "a"])
