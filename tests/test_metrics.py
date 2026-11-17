"""Metric tests with small arrays whose expected values can be checked by hand.

NASA score landmarks used below: d = -13 (early) and d = +10 (late) both give exp(1) - 1.
"""
import math

import numpy as np
import pytest

from src.metrics import compute_metrics, mae, nasa_score, r2_score, rmse

E_MINUS_1 = 1.718281828459045          # exp(1) - 1
EARLY_10 = 1.1581055339484458          # exp(10 / 13) - 1
EARLY_5 = 0.46904919384901667          # exp(5 / 13) - 1
LATE_5 = 0.6487212707001282            # exp(5 / 10) - 1


def test_perfect_predictions():
    y = [10.0, 20.0, 30.0]
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2_score(y, y) == 1.0
    assert nasa_score(y, y) == 0.0


def test_symmetric_small_errors():
    y_true, y_pred = [10, 20, 30, 40], [12, 18, 32, 38]
    assert rmse(y_true, y_pred) == pytest.approx(2.0)
    assert mae(y_true, y_pred) == pytest.approx(2.0)
    # SS_res = 16, SS_tot = 225 + 25 + 25 + 225 = 500
    assert r2_score(y_true, y_pred) == pytest.approx(1 - 16 / 500)


def test_one_large_error_hits_rmse_harder_than_mae():
    y_true, y_pred = [10, 20, 30, 40], [10, 20, 30, 80]
    assert mae(y_true, y_pred) == pytest.approx(10.0)             # 40 / 4
    assert rmse(y_true, y_pred) == pytest.approx(20.0)            # sqrt(1600 / 4)
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)
    assert r2_score(y_true, y_pred) == pytest.approx(1 - 1600 / 500)


def test_known_r2():
    y_true, y_pred = [1, 2, 3, 4], [1, 2, 3, 5]
    assert r2_score(y_true, y_pred) == pytest.approx(0.8)        # 1 - 1 / 5
    assert rmse(y_true, y_pred) == pytest.approx(0.5)
    assert mae(y_true, y_pred) == pytest.approx(0.25)


def test_r2_predicting_the_mean_is_zero():
    assert r2_score([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)


def test_r2_constant_target_is_finite():
    assert r2_score([5, 5, 5], [5, 5, 5]) == 1.0
    assert r2_score([5, 5, 5], [6, 6, 6]) == 0.0


def test_nasa_all_early_predictions():
    # d = -13 and -10 (prediction below the truth)
    assert nasa_score([100, 100], [87, 90]) == pytest.approx(E_MINUS_1 + EARLY_10)


def test_nasa_all_late_predictions():
    # d = +10 and +5 (prediction above the truth)
    assert nasa_score([100, 100], [110, 105]) == pytest.approx(E_MINUS_1 + LATE_5)


def test_nasa_mixed_early_and_late():
    # d = -13 and +10 both give exp(1) - 1
    assert nasa_score([50, 50], [37, 60]) == pytest.approx(2 * E_MINUS_1)


def test_nasa_sign_convention_late_is_penalised_more():
    early = nasa_score([100], [90])      # d = -10 -> exp(10 / 13) - 1
    late = nasa_score([100], [110])      # d = +10 -> exp(10 / 10) - 1
    assert early == pytest.approx(EARLY_10)
    assert late == pytest.approx(E_MINUS_1)
    assert late > early
    for size in (1, 5, 10, 25, 50):
        assert nasa_score([100], [100 + size]) > nasa_score([100], [100 - size])


def test_nasa_zero_error_adds_nothing_and_is_additive():
    assert nasa_score([40, 40, 40], [40, 45, 40]) == pytest.approx(LATE_5)
    assert nasa_score([40, 40], [35, 45]) == pytest.approx(EARLY_5 + LATE_5)


def test_compute_metrics_keys_and_values():
    m = compute_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    assert set(m) == {"rmse", "mae", "r2", "nasa_score"}
    assert m["r2"] == pytest.approx(0.8)
    assert m["nasa_score"] == pytest.approx(math.exp(0.1) - 1)    # one late error of +1


@pytest.mark.parametrize("y_true, y_pred", [
    ([1, 2, 3], [1, 2]),               # length mismatch
    ([], []),                          # empty
    ([1.0, np.nan], [1.0, 2.0]),       # NaN in truth
    ([1.0, 2.0], [1.0, np.inf]),       # inf in prediction
    ([[1, 2]], [[1, 2]]),              # not 1-D
])
def test_invalid_inputs_raise(y_true, y_pred):
    for metric in (rmse, mae, r2_score, nasa_score):
        with pytest.raises(ValueError):
            metric(y_true, y_pred)
