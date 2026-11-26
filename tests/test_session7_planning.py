"""Frozen inputs, the predeclared matrix, model construction and the analysis functions."""
import json

import numpy as np
import pytest
import yaml

from src.benchmark_config import PROJECT_ROOT, load_benchmark_config, resolve_path
from src.session7_analysis import (degradation_rows, dominates, holm_adjust, paired_test, pareto_front,
                                   per_engine_metrics, planned_comparisons, random_control_rows,
                                   ranked_vs_random_rows, relative_rmse_increase_pct, robustness_rows)
from src.session7_execution import reproducibility_gate
from src.session7_models import build_condition_model
from src.session7_plan import (build_plan, load_frozen_sources, load_session7_config, load_train_validation,
                               model_definition, sensor_indices, sensor_subsets, subset_data)
from src.tabular_features import WindowFeatureTransformer
from tests.session7_helpers import CANONICAL, CONFIG, IDENTITY, synthetic

TIMING = load_benchmark_config(PROJECT_ROOT / "configs/session2_benchmark.yaml")["timing"]
MODELS = ("random_forest", "gradient_boosting", "lstm", "gru", "tcn", "transformer")


@pytest.fixture(scope="module")
def cfg():
    return load_session7_config(CONFIG)[0]


@pytest.fixture(scope="module")
def sources(cfg):
    return load_frozen_sources(cfg)


@pytest.fixture(scope="module")
def plan(cfg, sources):
    return build_plan(cfg, sources, synthetic(), IDENTITY, TIMING)


# ---- frozen inputs ---------------------------------------------------------------------------
def test_subsets_come_from_the_frozen_artifacts_in_session1_order(cfg, sources):
    topk = json.loads((PROJECT_ROOT / "artifacts/session3/sensor_subsets_topk.json").read_text())["subsets"]
    randomk = json.loads((PROJECT_ROOT / "artifacts/session3/sensor_subsets_randomk.json").read_text())["subsets"]
    subsets = sensor_subsets(cfg, sources, CANONICAL)
    assert len(subsets) == 29
    for s in subsets:
        assert len(s["sensors"]) == len(set(s["sensors"])) == s["budget"] and set(s["sensors"]) <= set(CANONICAL)
        assert s["indices"] == sorted(s["indices"]) == [CANONICAL.index(n) for n in s["sensors"]]
        if s["strategy"] in ("correlation", "permutation"):
            assert s["sensors"] == topk[s["strategy"]][f"top_{s['budget']}"]["sensors"]
        elif s["strategy"] == "random":
            assert s["sensors"] == randomk[f"random_{s['budget']}"]["draws"][str(s["subset_seed"])]


def test_sensor_index_mapping_rejects_duplicates_unknown_and_wrong_order():
    assert sensor_indices(["sensor_2", "sensor_11"], CANONICAL) == [0, 6]
    for bad, message in ((["sensor_2", "sensor_2"], "duplicate"), (["sensor_1"], "unknown"),
                         (["sensor_11", "sensor_2"], "order")):
        with pytest.raises(ValueError, match=message):
            sensor_indices(bad, CANONICAL)


def test_tampered_frozen_source_is_rejected(cfg, tmp_path):
    bad = yaml.safe_load(yaml.safe_dump(cfg))
    copy = tmp_path / "selected_gru.json"
    copy.write_text((PROJECT_ROOT / "artifacts/session4/selected_gru.json").read_text() + " ")
    bad["frozen_sources"]["gru"] = str(copy)
    with pytest.raises(ValueError, match="manifest"):
        load_frozen_sources(bad)


@pytest.mark.parametrize("section,key,value,match", [
    ("benchmark", "evaluation_sets", ["validation", "test"], "test set is not part"),
    ("execution", "device", "mps", "cpu"),
    ("matrix", "training_seeds", [42, 101], "training_seeds"),
    ("matrix", "budgets", [14, 4], "budgets")])
def test_config_validation(tmp_path, section, key, value, match):
    raw = yaml.safe_load(CONFIG.read_text())
    raw[section][key] = value
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        load_session7_config(path)


@pytest.mark.skipif(not (resolve_path("data/processed/fd001") / "manifest.json").exists(),
                    reason="Session 1 artifacts not generated")
def test_data_loading_never_opens_the_test_file(cfg, monkeypatch):
    opened, real = [], np.load
    monkeypatch.setattr(np, "load", lambda path, *a, **k: opened.append(str(path)) or real(path, *a, **k))
    data, identity = load_train_validation(cfg)
    assert set(data.X) == set(data.y) == {"train", "validation"} and not any("test" in p.split("/")[-1] for p in opened)
    assert data.X["train"].shape == (14459, 30, 14) and data.X["validation"].shape == (3272, 30, 14)
    assert data.sensors == CANONICAL and identity["n_engines"] == {"train": 80, "validation": 20}


def test_sliced_arrays_keep_the_frozen_memory_layout_and_features_bit_for_bit():
    data = synthetic()
    full = subset_data(data, list(range(14)))
    t = WindowFeatureTransformer(CANONICAL, sequence_length=30)
    for split in ("train", "validation"):
        assert full.X[split].flags.c_contiguous
        np.testing.assert_array_equal(t.transform(full.X[split]), t.transform(data.X[split]))   # exact, not close
    assert subset_data(data, [0, 5, 9]).X["train"].flags.c_contiguous


@pytest.mark.skipif(not (resolve_path("data/processed/fd001") / "manifest.json").exists(),
                    reason="Session 1 artifacts not generated")
def test_full_subset_of_the_real_data_gives_bit_identical_features(cfg):
    data, _ = load_train_validation(cfg)
    t = WindowFeatureTransformer(data.sensors, sequence_length=30)
    full = subset_data(data, list(range(14)))
    np.testing.assert_array_equal(t.transform(full.X["validation"]), t.transform(data.X["validation"]))


# ---- predeclared matrix ----------------------------------------------------------------------------
def test_matrix_has_174_unique_deterministic_conditions(cfg, sources, plan):
    c = plan["conditions"]
    assert plan["n_conditions"] == len(c) == 174 and set(plan["per_model"].values()) == {29}
    assert len({x["experiment_id"] for x in c}) == 174
    assert [sum(x["stage"] == s for x in c) for s in (1, 2, 3)] == [6, 48, 120]
    assert [x["stage"] for x in c] == sorted(x["stage"] for x in c)
    again = build_plan(cfg, sources, synthetic(), IDENTITY, TIMING)
    assert [x["experiment_id"] for x in again["conditions"]] == [x["experiment_id"] for x in c]
    assert all(x["training_seed"] == 42 and x["evaluation_sets"] == ["validation"] for x in c)
    changed = build_plan(cfg, sources, synthetic(), {**IDENTITY, "manifest_sha256": "other"}, TIMING)
    assert changed["conditions"][0]["experiment_id"] != c[0]["experiment_id"]       # identity = scientific inputs


def test_full_sensor_set_is_planned_once_with_all_memberships(plan):
    full = [x for x in plan["conditions"] if x["budget"] == 14]
    assert len(full) == 6 and all(x["strategy"] == "full" and x["sensors"] == CANONICAL for x in full)
    assert all({"correlation", "permutation", "random:101", "random:105"} <= set(x["memberships"]) for x in full)
    keys = [(x["model"], tuple(x["sensor_indices"])) for x in plan["conditions"]]
    assert len(keys) == len(set(keys))                                              # no duplicate training


# ---- frozen model construction -----------------------------------------------------------------------
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("k", [14, 4, 2])
def test_factories_keep_hyperparameters_and_adapt_only_the_inputs(cfg, sources, model, k):
    data = subset_data(synthetic(), list(range(k)))
    m = build_condition_model(model, sources, data, execution=cfg["execution"])
    if model in ("random_forest", "gradient_boosting"):
        frozen = sources["selected_models"]["models"][model]
        assert m.params == frozen["params"] and m.seed == 42
        assert m.transformer.n_features == 6 * k and m.transformer.sensors == data.sensors
        return
    frozen = sources[model]["architecture"]
    arch = m.architecture() if hasattr(m, "architecture") else {
        "hidden_size": m.hidden_size, "num_layers": m.num_layers, "dropout": m.dropout,
        "sequence_length": m.sequence_length, "n_sensors": m.n_sensors}
    assert arch["n_sensors"] == k
    for key, value in arch.items():
        if key in frozen and key not in ("n_sensors", "n_parameters", "receptive_field", "readout"):
            assert value == frozen[key], key
    training = {x: v for x, v in sources[model]["training"].items() if x not in ("seed", "device")}
    assert m.training == training and m.seed == 42 and m.device == "cpu"
    assert m._build()(__import__("torch").zeros(3, 30, k)).shape == (3,)


def test_classical_features_are_computed_per_selected_sensor_not_sliced_blindly():
    data = synthetic()
    indices = [1, 5, 13]
    full = WindowFeatureTransformer(CANONICAL, sequence_length=30)
    sub = WindowFeatureTransformer([CANONICAL[i] for i in indices], sequence_length=30)
    F_full, F_sub = full.transform(data.X["validation"]), sub.transform(subset_data(data, indices).X["validation"])
    assert F_sub.shape[1] == 18
    columns = [full.feature_names.index(n) for n in sub.feature_names]
    np.testing.assert_allclose(F_sub, F_full[:, columns])
    assert not np.allclose(F_sub, F_full[:, :18])
    with pytest.raises(ValueError, match="test"):
        bad = synthetic()
        bad.X["test"] = bad.X["validation"]
        subset_data(bad, indices)


@pytest.mark.parametrize("model", MODELS)
def test_reduced_sensor_models_train_and_predict(cfg, sources, model):
    data = subset_data(synthetic(n_train=96, n_validation=64), [2, 9])
    m = build_condition_model(model, sources, data, execution=cfg["execution"])
    if hasattr(m, "training"):
        m.training = {**m.training, "max_epochs": 2}
    pred = m.fit(data.X["train"], data.y["train"]).predict(data.X["validation"])
    assert pred.shape == (len(data.y["validation"]),) and np.isfinite(pred).all()


# ---- analysis ----------------------------------------------------------------------------------------
def _row(model, budget, strategy, rmse, seed="", latency=1.0, status="completed"):
    return {"experiment_id": f"{model}-{budget}-{strategy}{seed}", "model": model, "budget": budget,
            "strategy": strategy, "subset_seed": seed, "val_rmse": rmse, "val_mae": rmse - 1,
            "val_nasa_score": rmse * 10, "inference_latency_ms_per_batch": latency,
            "throughput_samples_per_s": 100.0 / latency, "model_size_bytes": 100 * budget, "status": status}


def test_relative_degradation_keeps_negative_values():
    assert relative_rmse_increase_pct(11.0, 10.0) == pytest.approx(10.0)
    assert relative_rmse_increase_pct(9.0, 10.0) == pytest.approx(-10.0)
    rows = [_row("m", 14, "full", 10.0), _row("m", 4, "correlation", 9.0, latency=2.0)]
    d = [x for x in degradation_rows(rows) if x["budget"] == 4][0]
    assert d["relative_rmse_increase_pct"] == pytest.approx(-10.0) and d["sensors_removed"] == 10
    assert d["sensor_reduction_pct"] == pytest.approx(100 * 10 / 14) and d["latency_change_pct"] == pytest.approx(100)
    assert d["model_size_change_bytes"] == -1000


def test_random_controls_are_summarised_and_compared():
    rows = [_row("m", 14, "full", 10.0), _row("m", 4, "correlation", 12.0)] + \
        [_row("m", 4, "random", v, seed=s) for s, v in zip(range(101, 106), [11, 12.5, 13, 14, 19.5])]
    s = random_control_rows(rows)[0]
    assert (s["n_draws"], s["rmse_mean"], s["rmse_median"], s["rmse_min"], s["rmse_max"]) == (5, 14.0, 13.0, 11.0, 19.5)
    assert s["rmse_std"] == pytest.approx(np.std([11, 12.5, 13, 14, 19.5], ddof=1))
    v = ranked_vs_random_rows(rows)[0]
    assert v["difference_from_random_mean"] == pytest.approx(-2.0) and v["random_draws_better"] == 1
    r = [x for x in robustness_rows(rows, 5.0, [4]) if x["strategy"] == "correlation"][0]
    assert r["increase_pct_k4"] == pytest.approx(20.0) and r["smallest_budget_within_tolerance"] == ""


def test_pareto_dominance_rule():
    a, b, c = {"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 2, "y": 0}
    assert dominates(a, b, ["x", "y"]) and not dominates(b, a, ["x", "y"])
    assert not dominates(a, a, ["x", "y"]) and not dominates(a, c, ["x", "y"]) and not dominates(c, a, ["x", "y"])
    rows = [_row("a", 14, "full", 10, latency=5), _row("b", 4, "correlation", 12, latency=2),
            _row("c", 4, "correlation", 12.5, latency=3), _row("d", 2, "correlation", 9, latency=9, status="failed"),
            _row("e", 2, "correlation", 15, latency=2.5)]
    assert pareto_front(rows, ["val_rmse", "inference_latency_ms_per_batch"]) == ["a-14-full", "b-4-correlation"]
    assert "e-2-correlation" in pareto_front(rows, ["val_rmse", "inference_latency_ms_per_batch", "budget"])


def test_per_engine_aggregation():
    out = per_engine_metrics([10, 10, 20], [13, 7, 25], np.array([1, 1, 2]))
    assert [o["unit_id"] for o in out] == [1, 2] and [o["n_windows"] for o in out] == [2, 1]
    assert out[0]["rmse"] == pytest.approx(3.0) and out[0]["bias"] == pytest.approx(0.0)
    assert out[1]["rmse"] == pytest.approx(5.0) and out[1]["bias"] == pytest.approx(5.0)


def test_paired_test_and_holm_correction():
    worse = paired_test(np.linspace(0.5, 3.0, 20), resamples=500, seed=1)
    assert worse["n_engines"] == 20 and worse["p_value"] < 1e-4 and worse["rank_biserial"] == pytest.approx(1.0)
    assert worse["engines_worse"] == 20 and 0 < worse["ci95_low"] <= worse["median_difference"] <= worse["ci95_high"]
    assert paired_test(-np.linspace(0.5, 3.0, 20), resamples=100)["rank_biserial"] == pytest.approx(-1.0)
    null = paired_test(np.zeros(20), resamples=100)
    assert null["p_value"] == 1.0 and null["rank_biserial"] == 0.0 and null["n_nonzero"] == 0
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.9, 0.8]) == [1.0, 1.0] and holm_adjust([]) == []


def test_planned_comparisons_are_limited_and_report_unavailable_ones():
    rows = [_row("m", 14, "full", 10), _row("m", 4, "correlation", 12)] + \
        [_row("m", 4, "random", 11, seed=s) for s in (101, 102)]
    engines = {r["experiment_id"]: {u: r["val_rmse"] + 0.1 * u for u in range(20)} for r in rows}
    settings = {"families": ["reduced_ranked_vs_full", "ranked_vs_random_mean"], "alpha": 0.05,
                "bootstrap_resamples": 200, "bootstrap_seed": 1}
    out = planned_comparisons(rows, engines, settings, ["m"], [4])
    assert len(out) == 4                                          # 2 families x 1 model x 1 budget x 2 strategies
    available = {(o["family"], o["strategy"]): o for o in out if o["available"]}
    assert set(available) == {("reduced_ranked_vs_full", "correlation"), ("ranked_vs_random_mean", "correlation")}
    assert available[("reduced_ranked_vs_full", "correlation")]["median_difference"] == pytest.approx(2.0)
    assert available[("ranked_vs_random_mean", "correlation")]["median_difference"] == pytest.approx(1.0)
    assert all(o["family_size"] == 1 and "p_holm" in o for o in available.values())


def test_reproducibility_gate_requires_identical_predictions(cfg, sources, tmp_path):
    refs = [{"model": "lstm", "eval_set": "validation", "git_dirty": False, "rmse": 1.0, "predictions_sha256": "p",
             "n_parameters": 5}]
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"results": refs}))
    local = {**cfg, "frozen_sources": {**cfg["frozen_sources"], "historical_results": str(path)},
             "matrix": {**cfg["matrix"], "models": ["lstm"]}}
    good = {"model": "lstm", "status": "completed", "experiment_id": "x", "budget": 14, "metrics": {"rmse": 1.0},
            "predictions_sha256": "p", "n_parameters": 5,
            "weights_sha256": sources["lstm"]["checkpoint"]["weights_sha256"]}
    assert reproducibility_gate(local, sources, [good])["passed"]
    for change in ({"predictions_sha256": "q"}, {"weights_sha256": "w"}, {"n_parameters": 6}, {"status": "failed"}):
        assert not reproducibility_gate(local, sources, [{**good, **change}])["passed"], change


def test_definition_hashes_cover_the_frozen_architecture(sources):
    d = model_definition("transformer", sources)
    assert d["architecture"]["d_model"] == 64 and "n_sensors" not in d["architecture"]
    assert model_definition("random_forest", sources)["params"]["n_estimators"] == 100
