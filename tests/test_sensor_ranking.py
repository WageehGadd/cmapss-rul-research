import json

import numpy as np
import pytest

from src.classical_models import build_model
from src.run_session3 import sensor_intelligence
from src.sensor_ranking import (aggregate_to_sensors, compare_rankings, correlation_ranking,
                                feature_permutation_importance, permutation_ranking,
                                random_k_subsets, top_k_subsets)
from src.tabular_features import FEATURES, WindowFeatureTransformer
from tests.conftest import SENSORS, SEQ_LEN, synthetic_data

T = WindowFeatureTransformer(SENSORS, FEATURES, SEQ_LEN)


def informative_data(seed=0):
    """Windows whose target depends strongly on sensor_3, weakly on sensor_5, not on the rest."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, n in (("train", 400), ("validation", 150), ("test", 40)):
        X = rng.normal(size=(n, SEQ_LEN, 14)).astype("float32")
        y = 60 + 25 * X[:, -1, 2] + 5 * X[:, -1, 4] + rng.normal(scale=0.5, size=n)
        out[split] = (X, y.astype("float32"))
    return out


def test_correlation_ranking_orders_by_absolute_correlation():
    d = informative_data()
    F, y = T.transform(d["train"][0]), d["train"][1]
    ranking = correlation_ranking(F, y, T)
    assert [r["sensor"] for r in ranking[:2]] == ["sensor_3", "sensor_5"]
    assert [r["rank"] for r in ranking] == list(range(1, 15))
    assert all(r["abs_correlation"] == abs(r["correlation"]) for r in ranking)
    expected = np.corrcoef(d["train"][0][:, -1, 2].astype(np.float64), y)[0, 1]
    assert ranking[0]["correlation"] == pytest.approx(expected)


def test_correlation_ranking_depends_on_training_arguments_only():
    d = informative_data()
    F, y = T.transform(d["train"][0]), d["train"][1]
    a = correlation_ranking(F, y, T)
    d["validation"][1][:] = 0.0                          # changing other splits cannot matter
    d["test"][1][:] = np.nan
    assert correlation_ranking(F, y, T) == a


def test_constant_feature_gets_zero_correlation():
    F = np.zeros((50, T.n_features))
    ranking = correlation_ranking(F, np.arange(50.0), T)
    assert all(r["correlation"] == 0.0 for r in ranking)
    assert [r["sensor"] for r in ranking] == SENSORS          # ties broken by Session 1 order


def test_feature_to_sensor_aggregation_sum_and_mean():
    importance = np.arange(T.n_features, dtype=float)
    summed = aggregate_to_sensors(importance, T, "sum")
    assert summed["sensor_1"][0] == sum(range(0, 6))
    assert summed["sensor_2"][0] == sum(range(6, 12))
    assert summed["sensor_2"][1] == {f: float(6 + i) for i, f in enumerate(FEATURES)}
    assert aggregate_to_sensors(importance, T, "mean")["sensor_14"][0] == pytest.approx(np.mean(range(78, 84)))
    with pytest.raises(ValueError, match="expected 84"):
        aggregate_to_sensors(np.zeros(10), T)
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_to_sensors(importance, T, "max")


def _fit_rf(d):
    model = build_model("random_forest", {"n_estimators": 20, "max_depth": 6, "max_features": 0.5},
                        sensors=SENSORS, features=FEATURES, sequence_length=SEQ_LEN, seed=3)
    return model.fit(d["train"][0], d["train"][1])


def test_permutation_ranking_finds_the_informative_sensor_and_is_deterministic():
    d = informative_data()
    model = _fit_rf(d)
    F_val, y_val = T.transform(d["validation"][0]), d["validation"][1]
    ranking, detail = permutation_ranking(model, F_val, y_val, T, n_repeats=3, seed=5)
    assert ranking[0]["sensor"] == "sensor_3"
    assert set(ranking[0]["components"]) == set(FEATURES)
    assert ranking[0]["importance"] == pytest.approx(sum(ranking[0]["components"].values()))
    assert len(detail["importance_mean"]) == 84
    again, _ = permutation_ranking(model, F_val, y_val, T, n_repeats=3, seed=5)
    assert again == ranking


def test_permutation_importance_uses_validation_data_only():
    d = informative_data()
    model = _fit_rf(d)
    F_val, y_val = T.transform(d["validation"][0]), d["validation"][1]
    base = feature_permutation_importance(model, F_val, y_val, 2, 1)
    d["test"][0][:] = np.nan                                    # test arrays are not reachable
    d["test"][1][:] = np.nan
    again = feature_permutation_importance(model, F_val, y_val, 2, 1)
    np.testing.assert_array_equal(base[0], again[0])


def test_permutation_importance_leaves_features_unchanged():
    d = informative_data()
    model = _fit_rf(d)
    F_val = T.transform(d["validation"][0])
    before = F_val.copy()
    feature_permutation_importance(model, F_val, d["validation"][1], 2, 1)
    np.testing.assert_array_equal(F_val, before)


def _ranking(order):
    return [{"sensor": s, "rank": i + 1} for i, s in enumerate(order)]


def test_spearman_comparison_identical_reversed_and_disagreements():
    same = compare_rankings(_ranking(SENSORS), _ranking(SENSORS), [14, 7, 2])
    assert same["spearman_rho"] == pytest.approx(1.0)
    assert same["largest_disagreements"][0]["abs_rank_difference"] == 0
    assert same["top_k_overlap"]["top_7"]["jaccard"] == 1.0
    rev = compare_rankings(_ranking(SENSORS), _ranking(SENSORS[::-1]), [14, 7, 2])
    assert rev["spearman_rho"] == pytest.approx(-1.0)
    assert rev["largest_disagreements"][0]["sensor"] == "sensor_1"
    assert rev["largest_disagreements"][0]["rank_difference"] == 13
    assert rev["top_k_overlap"]["top_2"]["shared"] == []
    swapped = SENSORS.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    part = compare_rankings(_ranking(SENSORS), _ranking(swapped), [14, 2])
    assert part["spearman_rho"] == pytest.approx(1 - 6 * 2 / (14 * (14 ** 2 - 1)))
    assert part["top_k_overlap"]["top_2"]["jaccard"] == 1.0
    with pytest.raises(ValueError, match="different sensors"):
        compare_rankings(_ranking(SENSORS), _ranking(SENSORS[:-1] + ["other"]), [2])


def test_top_k_subsets_nested_deterministic_and_sized():
    ranking = _ranking(SENSORS[::-1])
    a = top_k_subsets(ranking, [14, 10, 7, 4, 2], SENSORS)
    assert a == top_k_subsets(ranking, [14, 10, 7, 4, 2], SENSORS)
    assert [len(a[f"top_{k}"]["sensors"]) for k in (14, 10, 7, 4, 2)] == [14, 10, 7, 4, 2]
    assert a["top_2"]["sensors_by_rank"] == ["sensor_14", "sensor_13"]
    assert a["top_2"]["sensors"] == ["sensor_13", "sensor_14"]            # Session 1 order
    assert set(a["top_4"]["sensors"]) <= set(a["top_7"]["sensors"])
    with pytest.raises(ValueError, match="outside"):
        top_k_subsets(ranking, [15], SENSORS)


def test_random_k_subsets_are_reproducible_sized_and_seed_dependent():
    seeds, budgets = [101, 102, 103], [14, 10, 7, 4, 2]
    a = random_k_subsets(SENSORS, budgets, seeds)
    assert a == random_k_subsets(SENSORS, budgets, seeds)
    for k in budgets:
        for seed in seeds:
            subset = a[f"random_{k}"]["draws"][str(seed)]
            assert len(subset) == len(set(subset)) == k and set(subset) <= set(SENSORS)
            assert subset == [s for s in SENSORS if s in subset]            # Session 1 order
    assert a["random_14"]["draws"]["101"] == SENSORS
    assert len({tuple(v) for v in a["random_4"]["draws"].values()}) > 1
    assert a["random_7"]["draws"]["101"] != random_k_subsets(SENSORS, [7], [999])["random_7"]["draws"]["999"]


def test_sensor_intelligence_reproduces_and_ignores_test_data(cfg):
    d = informative_data()
    scfg = {"ranking": {"correlation": {"feature": "last"},
                        "permutation": {"n_repeats": 2, "seed": 9, "aggregation": "sum"},
                        "top_disagreements": 3},
            "budgets": {"sensor_counts": [14, 10, 7, 4, 2], "random_control_seeds": [1, 2]}}
    model = _fit_rf(d)
    F_tr, F_va = T.transform(d["train"][0]), T.transform(d["validation"][0])
    first = sensor_intelligence(scfg, T, F_tr, d["train"][1], F_va, d["validation"][1], model)
    d["test"][0][:] = np.nan
    d["test"][1][:] = np.nan
    second = sensor_intelligence(scfg, T, F_tr, d["train"][1], F_va, d["validation"][1], model)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    ranking_a, ranking_b, _, comparison, topk, randomk = first
    assert set(topk) == {"correlation", "permutation"} and len(randomk) == 5
    assert -1.0 <= comparison["spearman_rho"] <= 1.0
