"""Session 8 unit and integration tests on synthetic data and temporary folders. No test here reads
the real FD003 test labels, trains a real model or writes to project stores."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

import src.session8_data as s8
import src.session8_inference as inference
from src.benchmark_config import PROJECT_ROOT
from src.checks import SanityCheckError
from src.data_loading import CANONICAL_COLUMNS
from src.metrics import nasa_score
from src.session8_guard import ExecutionAuthorization, require_new_location
from src.session7_plan import sha256_file
from src.session8_analysis import (across_engines, bootstrap_ci, dataset_metrics, error_summary, family_agreement,
                                   holm_adjust, paired_family, per_engine, signed_error, stage_metrics, stage_of,
                                   worst_engines)
from src.session8_evaluate import candidate_members, read_rows, targets, write_rows

CONFIG = PROJECT_ROOT / "configs/session8_evaluation.yaml"
AUTH = ExecutionAuthorization("synthetic test")
CFG = yaml.safe_load(CONFIG.read_text())
STAGES = CFG["evaluation"]["stages"]
FEATURES = [f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]


def frame(lengths, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for unit, n in enumerate(lengths, 1):
        df = pd.DataFrame(rng.normal(500, 3, size=(n, len(CANONICAL_COLUMNS))), columns=CANONICAL_COLUMNS)
        df["unit_id"], df["cycle"] = unit, np.arange(1, n + 1)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def write_raw(path, df):
    df.to_csv(path, sep=" ", header=False, index=False)


# ---- FD003 schema and preprocessing ------------------------------------------------------------

def test_raw_file_is_checked_against_the_schema(tmp_path):
    good = frame([40, 35])
    write_raw(tmp_path / "ok.txt", good)
    assert len(s8.read_fd(tmp_path / "ok.txt", "ok", 2, 75, FEATURES, 30)) == 75
    cases = {"columns": good.iloc[:, :-1], "short": frame([40, 20]),
             "gap": good.drop(index=5).reset_index(drop=True), "missing": good.assign(sensor_2=np.nan),
             "infinite": good.assign(sensor_3=np.inf)}
    for name, df in cases.items():
        write_raw(tmp_path / f"{name}.txt", df)
        with pytest.raises((SanityCheckError, ValueError)):
            s8.read_fd(tmp_path / f"{name}.txt", name, 2, len(df), FEATURES, 30)
    with pytest.raises(SanityCheckError):
        s8.read_fd(tmp_path / "ok.txt", "count", 3, 75, FEATURES, 30)
    (tmp_path / "empty.txt").write_text("")
    with pytest.raises(Exception):
        s8.read_fd(tmp_path / "empty.txt", "empty", 2, 0, FEATURES, 30)


def test_sensor_order_is_the_session1_order():
    features, length, cap = s8.sensor_list({"upstream": CFG["upstream"]})
    assert features == FEATURES and (length, cap) == (30, 125)


def test_windows_use_the_given_scaler_in_order_and_never_cross_engines():
    df = frame([45, 33])
    scaler = StandardScaler().fit(frame([60], seed=9)[FEATURES].to_numpy())
    before = (scaler.mean_.copy(), scaler.scale_.copy(), scaler.n_samples_seen_)
    ws = s8.trajectory_windows(df, scaler, FEATURES, 30, 125)
    assert np.array_equal(scaler.mean_, before[0]) and np.array_equal(scaler.scale_, before[1])
    assert scaler.n_samples_seen_ == before[2]                               # applied, never refitted
    assert len(ws.X) == (45 - 29) + (33 - 29) and ws.X.flags["C_CONTIGUOUS"]
    expected = scaler.transform(df[FEATURES].to_numpy("float64")).astype("float32")
    for i in (0, 15, 16, len(ws.X) - 1):
        unit, end = ws.unit_ids[i], ws.end_cycles[i]
        rows = np.flatnonzero((df.unit_id == unit) & (df.cycle <= end) & (df.cycle > end - 30))
        assert len(rows) == 30 and np.array_equal(ws.X[i], expected[rows])
        assert ws.y_raw[i] == df.cycle[df.unit_id == unit].max() - end       # target alignment


def test_terminal_windows_are_one_per_engine_without_labels():
    df = frame([45, 33, 31])
    scaler = StandardScaler().fit(df[FEATURES].to_numpy())
    ws = s8.terminal_windows(df, scaler, FEATURES, 30, 125)
    assert ws.y_raw is None and ws.unit_ids.tolist() == [1, 2, 3] and ws.end_cycles.tolist() == [45, 33, 31]


def test_subset_columns_keep_order_and_are_contiguous():
    ws = s8.Windows(np.arange(2 * 30 * 14, dtype="float32").reshape(2, 30, 14), np.array([1, 2]), np.array([30, 30]))
    sub = inference.subset_inputs(ws, [1, 5, 13])
    assert sub.flags["C_CONTIGUOUS"] and np.array_equal(sub[1, 3], ws.X[1, 3, [1, 5, 13]])


# ---- targets and labels ------------------------------------------------------------------------

def test_targets_are_capped_and_aligned_by_unit():
    test = s8.Windows(np.zeros((3, 30, 14), "float32"), np.array([1, 2, 3]), np.array([40, 41, 42]))
    traj = s8.Windows(test.X, test.unit_ids, test.end_cycles, np.array([200.0, 60.0, 0.0]))
    inputs = {"fd001_validation": traj, "fd003_train": traj, "fd003_test": test}
    out = targets({"evaluation": {"rul_cap": 125}}, inputs, np.array([10.0, 145.0, 125.0]))
    assert out["fd003_test"][0].tolist() == [10, 125, 125] and out["fd003_test"][1].tolist() == [10, 145, 125]
    assert out["fd001_validation"][0].tolist() == [125, 60, 0]
    assert "fd003_test" not in targets({"evaluation": {"rul_cap": 125}}, inputs)
    with pytest.raises(ValueError):
        targets({"evaluation": {"rul_cap": 125}}, inputs, np.array([1.0, 2.0]))
    shuffled = {**inputs, "fd003_test": s8.Windows(test.X, np.array([2, 1, 3]), test.end_cycles)}
    with pytest.raises(ValueError):
        targets({"evaluation": {"rul_cap": 125}}, shuffled, np.array([1.0, 2.0, 3.0]))


def label_cfg(tmp_path):
    (tmp_path / "RUL.txt").write_text("5 \n7 \n")
    (tmp_path / "config.yaml").write_text("a: 1\n")
    return {"_path": tmp_path / "config.yaml", "outputs": {"working_dir": str(tmp_path / "work")},
            "fd003": {"raw_dir": str(tmp_path), "files": {"rul": "RUL.txt"},
                      "sha256": {"rul": sha256_file(tmp_path / "RUL.txt")}, "expected": {"test_engines": 2}}}


def test_every_label_access_is_recorded_and_a_changed_design_is_refused(tmp_path):
    cfg = label_cfg(tmp_path)
    assert s8.load_test_labels(cfg, "first", authorization=AUTH).tolist() == [5, 7]
    s8.load_test_labels(cfg, "second", authorization=AUTH)
    ledger = json.loads(s8.ledger_path(cfg).read_text())
    assert [e["purpose"] for e in ledger] == ["first", "second"]
    (tmp_path / "config.yaml").write_text("a: 2\n")                         # e.g. new candidates
    with pytest.raises(RuntimeError, match="another protocol"):
        s8.load_test_labels(cfg, "third", authorization=AUTH)
    assert len(json.loads(s8.ledger_path(cfg).read_text())) == 2


def test_inputs_and_predictions_cannot_reach_the_test_labels():
    """Only session8_data reads the RUL file; the inference module never asks for labels."""
    for module in ("session8_inference", "session8_analysis", "session8_sensitivity"):
        text = (PROJECT_ROOT / f"src/{module}.py").read_text()
        assert "load_test_labels" not in text and "read_rul_file" not in text and "RUL_FD003" not in text
    sensitivity = (PROJECT_ROOT / "src/session8_sensitivity.py").read_text()
    assert "load_inputs" not in sensitivity and "fd003" not in sensitivity.lower().replace("no fd003 data", "")


def test_candidates_are_resolved_from_the_frozen_matrix_alone():
    plan = json.loads((PROJECT_ROOT / "artifacts/session7/experiment_plan.json").read_text())
    conditions = {c["experiment_id"]: c for c in plan["conditions"]}
    members = candidate_members(CFG, conditions)
    assert len(members["gb_k10_random"]) == 5 and len(members["gru_k10_perm"]) == 1
    assert set(members) == {c["key"] for c in CFG["candidates"]}
    with pytest.raises(ValueError):
        candidate_members({"candidates": [{"key": "x", "model": "gru", "budget": 3, "strategy": "full"}]}, conditions)


@pytest.mark.parametrize("change", [
    lambda c: c.pop("candidates"), lambda c: c.update(preferred_reduced="nobody"),
    lambda c: c["evaluation"]["stages"][1].update(low=60),
    lambda c: c["candidates"].append(dict(c["candidates"][0]))])
def test_malformed_configuration_is_refused(tmp_path, change):
    cfg = yaml.safe_load(CONFIG.read_text())
    change(cfg)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        s8.load_config(tmp_path / "c.yaml")


# ---- errors, stages, NASA score ----------------------------------------------------------------

def test_signed_error_is_predicted_minus_true():
    s = error_summary([50, 50, 50], [60, 45, 50])
    assert signed_error([50], [60]).tolist() == [10]
    assert s["mean_signed"] == pytest.approx(5 / 3) and s["median_abs"] == 5
    assert (s["over_fraction"], s["under_fraction"]) == (pytest.approx(1 / 3), pytest.approx(1 / 3))
    assert (s["mean_late_magnitude"], s["mean_early_magnitude"]) == (10, 5)


def test_late_predictions_cost_more_in_the_nasa_score():
    late, early = nasa_score([50.0], [60.0]), nasa_score([50.0], [40.0])
    assert late == pytest.approx(np.exp(1.0) - 1) and early == pytest.approx(np.exp(10 / 13) - 1) and late > early
    assert error_summary([50, 50], [60, 40])["nasa_late_share"] == pytest.approx(late / (late + early))
    assert error_summary([50], [50])["nasa_late_share"] == 0.0


def test_stage_boundaries():
    assert stage_of([0, 50, 51, 100, 101, 400], STAGES).tolist() == [
        "near_failure", "near_failure", "intermediate", "intermediate", "early_life", "early_life"]
    for bad in ([], [-1.0], [np.nan], [50.5]):
        with pytest.raises(ValueError):
            stage_of(bad, STAGES)


def test_errors_are_grouped_by_engine_not_pooled():
    units = np.array([1] * 9 + [2])
    y, p = np.full(10, 80.0), np.array([81.0] * 9 + [100.0])
    engines = per_engine(y, p, units, y)
    assert engines[1]["mae"] == 1 and engines[2]["mae"] == 20 and engines[1]["n_predictions"] == 9
    assert across_engines(engines)["engine_mean_mae"] == 10.5                # not the pooled 2.9
    assert np.isnan(engines[1]["near_rmse"])
    metrics, _ = dataset_metrics(y, p, units, y)
    assert metrics["mae"] == pytest.approx(2.9) and metrics["engine_mean_mae"] == 10.5 and metrics["n_engines"] == 2
    rows = stage_metrics(y, p, units, y, STAGES)
    assert [r["n_engines"] for r in rows] == [0, 2, 0] and rows[1]["mae"] == pytest.approx(2.9)


@pytest.mark.parametrize("y,p", [([], []), ([1.0], [1.0, 2.0]), ([np.nan], [1.0]), ([1.0], [np.inf])])
def test_empty_or_invalid_errors_are_refused(y, p):
    with pytest.raises(ValueError):
        signed_error(y, p)


# ---- worst engines and statistics --------------------------------------------------------------

def engine(mae, late=0.0, early=0.0, near=float("nan")):
    return {"mae": mae, "late_part": late, "early_part": early, "near_rmse": near}


def test_worst_engine_rule_is_deterministic():
    a = {1: engine(5, late=5, near=1), 2: engine(9), 3: engine(7, early=7, near=3)}
    b = {1: engine(7, late=7, near=1), 2: engine(3), 3: engine(7, early=7, near=5)}
    rows = worst_engines({"a": a, "b": b}, 2)
    pick = lambda c: [(r["unit_id"], r["score"]) for r in rows if r["criterion"] == c]
    assert pick("absolute_error") == [(3, 7.0), (1, 6.0)]                    # tie at 6: lower unit first
    assert pick("late_error") == [(1, 6.0)] and pick("early_error") == [(3, 7.0)]
    assert pick("near_failure") == [(3, 4.0), (1, 1.0)]
    assert rows == worst_engines({"a": a, "b": b}, 2)
    with pytest.raises(ValueError):
        worst_engines({"a": a, "b": {1: engine(1)}}, 2)
    agreement = family_agreement({"a": a, "b": a}, 2)
    assert agreement[0]["spearman_rho"] == pytest.approx(1.0) and agreement[0]["shared_worst"] == 2


def test_paired_tests_pair_by_engine_and_correct_within_the_family():
    settings = {"alpha": 0.05, "bootstrap_resamples": 200, "bootstrap_seed": 1}
    a = {u: 10.0 + u for u in range(20)}
    b = {u: 9.0 + u for u in range(20)}
    same = dict(a)
    rows = paired_family("f", [({"a": "x"}, a, b), ({"a": "y"}, a, same)], settings)
    assert rows[0]["median_difference"] == 1 and rows[0]["engines_worse"] == 20 and rows[0]["family_size"] == 2
    assert rows[0]["p_holm"] == pytest.approx(min(1.0, 2 * rows[0]["p_value"])) and rows[0]["significant_after_correction"]
    assert rows[1]["n_nonzero"] == 0 and rows[1]["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_family("f", [({}, a, {u: 1.0 for u in range(19)})], settings)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_is_deterministic_and_needs_engines():
    values = [1.0, 2.0, 3.0, 4.0]
    assert bootstrap_ci(values, 500, 3) == bootstrap_ci(values, 500, 3)
    low, high = bootstrap_ci(values, 500, 3, "mean")
    assert 1 <= low < 2.5 < high <= 4
    for bad in ([1.0], [1.0, np.nan]):
        with pytest.raises(ValueError):
            bootstrap_ci(bad, 100, 1)


# ---- provenance and resume ---------------------------------------------------------------------

def test_stored_predictions_are_reused_only_for_the_same_identity(tmp_path):
    path = tmp_path / "p.npz"
    identity = {"experiment_id": "a", "inputs": "1"}
    assert inference.stored_prediction(path, identity) is None
    inference.save_prediction(path, identity, np.array([1.0, 2.0]))
    assert inference.stored_prediction(path, identity).tolist() == [1, 2]
    with pytest.raises(RuntimeError, match="other inputs"):
        inference.stored_prediction(path, {**identity, "inputs": "2"})
    np.savez(path, prediction=np.array([1.0, 9.0]))
    with pytest.raises(RuntimeError, match="digest"):
        inference.stored_prediction(path, identity)


def test_a_changed_model_file_is_refused_before_loading(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model/gru.pt").write_bytes(b"weights")
    record = {"files": {"model/gru.pt": "0" * 64}}
    with pytest.raises(ValueError, match="differs from the canonical record"):
        inference.load_model({"experiment_id": "x", "model": "gru", "sensors": FEATURES}, tmp_path, record, {}, authorization=AUTH)


def test_tables_round_trip_deterministically(tmp_path):
    rows = [{"a": 1, "b": 0.1, "c": True, "d": "x"}, {"a": 2, "b": "", "c": False, "d": "y", "e": 3.5}]
    write_rows(tmp_path / "t.csv", rows)
    first = (tmp_path / "t.csv").read_bytes()
    back = read_rows(tmp_path / "t.csv")
    write_rows(tmp_path / "t.csv", rows)
    assert (tmp_path / "t.csv").read_bytes() == first and back[0]["b"] == 0.1 and back[1]["c"] is False
    assert back[1]["e"] == 3.5 and back[0]["e"] == ""
    with pytest.raises(ValueError):
        write_rows(tmp_path / "t.csv", [])


def test_session8_writes_outside_the_frozen_outputs():
    outputs = CFG["outputs"]
    s7 = yaml.safe_load((PROJECT_ROOT / "configs/session7_main_experiment.yaml").read_text())["outputs"]
    assert not set(outputs.values()) & {s7["frozen_dir"], s7["working_dir"], s7["report"]}
    assert CFG["sensitivity"]["experiment_name"] not in ("cmapss_rul_benchmark", "cmapss_rul_session7")
    assert outputs["working_dir"].startswith("results/")                       # git-ignored


def test_training_caps_change_only_targets_and_target_scale():
    from src.session7_plan import DEEP
    from src.session8_sensitivity import sources_with_cap, with_cap
    from tests.session7_helpers import synthetic

    data = synthetic()
    raw = {k: v.astype("float64") * 10 for k, v in data.y.items()}
    capped = with_cap(data, raw, 100)
    assert capped.X is data.X and capped.rul_cap == 100 and capped.y["train"].max() == 100
    sources = {m: {"training": {"target_scale": 125.0, "seed": 42}} for m in DEEP}
    out = sources_with_cap(sources, 150)
    assert out["gru"]["training"] == {"target_scale": 150.0, "seed": 42}
    assert sources["gru"]["training"]["target_scale"] == 125.0


# ---- published results reconcile (skipped when Session 8 has not been run) ----------------------

ARTIFACTS = PROJECT_ROOT / CFG["outputs"]["artifacts_dir"]
published = pytest.mark.skipif(not (ARTIFACTS / "manifest.json").exists(), reason="Session 8 outputs are absent")


@published
def test_artifact_figure_and_report_hashes_match_the_manifest():
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert all(sha256_file(ARTIFACTS / n) == d for n, d in manifest["files"].items())
    assert all(sha256_file(PROJECT_ROOT / n) == d for n, d in manifest["figures"].items())
    assert sha256_file(PROJECT_ROOT / CFG["outputs"]["report"]) == manifest["report_sha256"]
    assert manifest["fd001_official_test_arrays_opened"] is False and len(manifest["checkpoints"]) == 174
    assert manifest["canonical"] is False and "not canonical" in manifest["evidence_class"]
    assert manifest["evidence_manifest"]["sha256"] == sha256_file(ARTIFACTS / "evidence_manifest.json")
    assert not any(f"/fig{n}_" in p for p in manifest["figures"] for n in range(1, 22))


@published
def test_master_table_agrees_with_the_canonical_session7_summary():
    master = {r["experiment_id"]: r for r in read_rows(ARTIFACTS / "master_results.csv")}
    canonical = read_rows(PROJECT_ROOT / "artifacts/session7/master_summary.csv")
    assert len(master) == len(canonical) == 174
    for row in canonical:
        mine = master[row["experiment_id"]]
        assert mine["mlflow_run_id"] == row["mlflow_run_id"]
        assert mine["fd001_validation_rmse"] == pytest.approx(row["val_rmse"], rel=1e-9)


@published
def test_report_states_the_saved_numbers():
    report = (PROJECT_ROOT / CFG["outputs"]["report"]).read_text()
    for r in read_rows(ARTIFACTS / "candidate_summary.csv"):
        if r["dataset"] == "fd003_test":
            assert f"| {r['candidate']} |" in report and f"{r['rmse']:.2f}" in report
    stats = read_rows(ARTIFACTS / "statistics.csv")
    assert {r["family"]: r["family_size"] for r in stats} == {
        "A_reduced_vs_full": 5, "B_full_vs_reference": 5, "C_ranked_vs_random_mean": 48}
    assert all(r["n_engines"] == 100 for r in stats)
    for section in range(1, 16):
        assert f"\n## {section}. " in report
    rows = read_rows(ARTIFACTS / "stage_metrics.csv")
    terminal = [r for r in rows if r["dataset"] == "fd003_test"]
    assert terminal and all(r["engine_mean_rmse"] == "" and r["rmse"] >= r["mae"] - 1e-9 for r in terminal)
    assert any(r["rmse"] > r["mae"] + 1 for r in terminal)
    worst = read_rows(ARTIFACTS / "worst_engines.csv")
    for view in ("all_full_models", "without_tcn"):
        mine = [r["unit_id"] for r in worst if (r["view"], r["dataset"], r["criterion"]) ==
                (view, "fd003_test", "absolute_error")]
        assert ", ".join(map(str, mine)) in report


@published
def test_test_set_metrics_recompute_from_the_saved_predictions():
    path = PROJECT_ROOT / CFG["outputs"]["working_dir"] / "fd003_test_predictions.csv"
    if not path.exists():
        pytest.skip("detailed predictions are not present")
    rows = read_rows(path)
    master = {r["experiment_id"]: r for r in read_rows(ARTIFACTS / "master_results.csv")}
    by_id = {}
    for r in rows:
        assert r["error"] == pytest.approx(r["prediction"] - r["target"])
        assert r["target"] == min(r["true_rul_raw"], 125)
        by_id.setdefault(r["experiment_id"], []).append(r)
    assert len(by_id) == 174
    for key, group in by_id.items():
        assert sorted(g["unit_id"] for g in group) == list(range(1, 101))
        rmse = np.sqrt(np.mean([g["error"] ** 2 for g in group]))
        assert rmse == pytest.approx(master[key]["fd003_test_rmse"], rel=1e-9)


# ---- audit remediation: named aggregations -----------------------------------------------------

def test_terminal_stage_columns_are_what_their_names_say():
    """One prediction per engine with errors 1, 2, 12: RMSE, MAE and median absolute error all differ."""
    y, p = np.array([10.0, 20.0, 30.0]), np.array([11.0, 22.0, 42.0])
    row = stage_metrics(y, p, np.array([1, 2, 3]), y, STAGES)[0]
    assert row["rmse"] == pytest.approx(np.sqrt((1 + 4 + 144) / 3))
    assert row["mae"] == pytest.approx(5.0) and row["median_abs"] == pytest.approx(2.0)
    assert row["engine_mean_mae"] == pytest.approx(5.0)
    for name in ("engine_mean_rmse", "engine_mean_median_abs", "engine_mean_median_signed",
                 "engine_mean_nasa_late_share"):
        assert row[name] == "", name                       # a mean of one-sample RMSEs would be the MAE
    metrics, _ = dataset_metrics(y, p, np.array([1, 2, 3]), y)
    assert metrics["engine_mean_rmse"] == "" and metrics["rmse"] > metrics["mae"]


def test_pooled_and_per_engine_aggregations_are_both_reported_and_differ():
    units = np.array([1] * 9 + [2])
    y = np.full(10, 40.0)
    p = y + np.array([-1.0] * 9 + [20.0])                  # one long conservative engine, one optimistic
    row = stage_metrics(y, p, units, y, STAGES)[0]
    assert row["over_fraction"] == pytest.approx(0.1) and row["engine_mean_over_fraction"] == pytest.approx(0.5)
    assert row["mean_signed"] == pytest.approx(1.1) and row["engine_mean_signed"] == pytest.approx(9.5)
    assert row["rmse"] == pytest.approx(np.sqrt((9 + 400) / 10)) and row["engine_mean_rmse"] == pytest.approx(10.5)
    assert row["nasa_late_share"] != pytest.approx(row["engine_mean_nasa_late_share"])
    assert row["nasa_sum"] == pytest.approx(nasa_score(y, p))


def test_the_interval_has_the_same_estimand_as_the_point_value():
    from src.session8_analysis import bootstrap_mean_rmse

    rng = np.random.default_rng(0)
    one = np.abs(rng.normal(10, 5, 100))
    assert bootstrap_mean_rmse([one], 2000, 7) == bootstrap_ci(one, 2000, 7)        # one model: unchanged
    two = [np.full(50, 2.0), np.full(50, 10.0)]            # mean of RMSEs is 6; RMSE of mean errors too
    assert bootstrap_mean_rmse(two, 100, 1) == (6.0, 6.0)
    opposed = [np.tile([0.0, 10.0], 50), np.tile([10.0, 0.0], 50)]
    low, high = bootstrap_mean_rmse(opposed, 4000, 1)
    point = float(np.mean([np.sqrt(np.mean(e ** 2)) for e in opposed]))             # 7.07, mean of RMSEs
    assert low < point <= high + 1e-9                      # the interval surrounds the point shown
    assert bootstrap_ci(np.mean(opposed, axis=0), 4000, 1)[1] < point
    for bad in ([[1.0]], [[1.0, np.nan]]):
        with pytest.raises(ValueError):
            bootstrap_mean_rmse(bad, 10, 1)


def test_worst_engine_views_keep_the_predeclared_rule_and_label_the_other():
    from src.session8_analysis import worst_engine_views

    good = {1: engine(5, late=5, near=5), 2: engine(9, late=9, near=9), 3: engine(1, late=1, near=1)}
    tcn = {1: engine(1, late=1, near=1), 2: engine(1, late=1, near=1), 3: engine(500, late=500, near=500)}
    rows = worst_engine_views({"gru": good, "lstm": good, "tcn": tcn}, 1)
    top = {(r["view"], r["status"], r["models"]): r["unit_id"] for r in rows if r["criterion"] == "absolute_error"}
    assert top == {("all_full_models", "predeclared", "gru|lstm|tcn"): 3,
                   ("without_tcn", "post_hoc", "gru|lstm"): 2}


# ---- audit remediation: evidence and provenance ------------------------------------------------

def evidence_project(tmp_path, register=True):
    """A synthetic completed execution in a temporary folder: registered once, then entered into its own
    reference, as a reviewer would."""
    import sqlite3

    from src.session8_evidence import register_evidence

    work, artifacts, tracking = tmp_path / "work", tmp_path / "artifacts", tmp_path / "mlruns"
    for folder in (work / "predictions/fd003_test", work / "sensitivity", artifacts, tracking, tmp_path / "figures"):
        folder.mkdir(parents=True)
    dirty = {"git_commit": "a", "git_dirty": True}
    inference.save_prediction(work / "predictions/fd003_test/x.npz", {"experiment_id": "x"}, np.array([1.0, 2.0]))
    (work / "sensitivity/f.json").write_text(json.dumps({"git": dirty}))
    (work / "evaluation_run.json").write_text(json.dumps({"git": dirty}))
    (work / "fd003_test_access.json").write_text("[]")
    (work / "dataset_facts.json").write_text('{"engines": 2}')
    (work / "snapshot.json").write_text("{}")
    db = sqlite3.connect(tracking / "mlflow.db")
    db.executescript("create table experiments(experiment_id, name); create table runs(run_uuid, experiment_id, name,"
                     " status, lifecycle_stage); create table params(run_uuid, key, value);"
                     " create table latest_metrics(run_uuid, key, value);"
                     " insert into experiments values (1, 'e8'); insert into runs values ('r1', 1, 'n', 'FINISHED', 'active');"
                     " insert into params values ('r1', 'git_dirty', 'True')")
    db.commit()
    db.close()
    (tmp_path / "s7.json").write_text("{}")
    reference = tmp_path / "reference.json"
    reference.write_text('{"executions": []}')
    cfg = {"_reference": str(reference), "fd003": {"sha256": {"rul": "h"}},
           "outputs": {"working_dir": str(work), "artifacts_dir": str(artifacts),
                       "figures_dir": str(tmp_path / "figures"), "report": str(tmp_path / "report.md")},
           "upstream": {"mlflow_dir": str(tracking), "session7_manifest": str(tmp_path / "s7.json")},
           "sensitivity": {"experiment_name": "e8"}}
    if register:
        seal_path = register_evidence(cfg, "n", authorization=AUTH)
        set_reference(cfg, evidence_manifest_sha256=sha256_file(seal_path), canonical=False,
                      evidence_class="not canonical: the source state of some evidence was not recorded",
                      supplementary_files={"snapshot.json": sha256_file(work / "snapshot.json")})
    return cfg


def set_reference(cfg, **changes):
    from pathlib import Path

    path = Path(cfg["_reference"])
    entries = json.loads(path.read_text())["executions"]
    entry = entries[0] if entries else {**cfg["outputs"], "experiment_name": cfg["sensitivity"]["experiment_name"]}
    path.write_text(json.dumps({"executions": [{**entry, **changes}]}))


def seal_of(cfg):
    from pathlib import Path

    return Path(cfg["outputs"]["artifacts_dir"]) / "evidence_manifest.json"


def forge(cfg):
    """Replace a prediction AND its sidecar, so that the evidence stays self-consistent."""
    from pathlib import Path

    path = Path(cfg["outputs"]["working_dir"]) / "predictions/fd003_test/x.npz"
    inference.save_prediction(path, {"experiment_id": "x"}, np.array([9.0, 9.0]))


@pytest.mark.parametrize("damage,expected", [
    (lambda w: np.savez(w / "predictions/fd003_test/x.npz", prediction=np.array([1.0, 9.0])), "changed predictions"),
    (lambda w: (w / "predictions/fd003_test/x.npz").unlink(), "missing predictions"),
    (lambda w: inference.save_prediction(w / "predictions/fd003_test/y.npz", {}, np.array([1.0])), "not in the sealed"),
    (lambda w: (w / "sensitivity/f.json").write_text(json.dumps({"git": {"git_commit": "b", "git_dirty": False}})),
     "changed sensitivity"),
    (lambda w: (w / "fd003_test_access.json").write_text('[{"purpose": "extra"}]'), "fd003_test_access"),
    (lambda w: (w / "predictions/fd003_test/extra.csv").write_text("x"), "unexpected file"),
    (lambda w: (w / "extra.npz").write_text("x"), "unexpected file"),
    (lambda w: (w / "snapshot.json").write_text('{"a": 1}'), "supplementary file missing or changed"),
    (lambda w: (w / "snapshot.json").unlink(), "supplementary file missing or changed")])
def test_sealed_evidence_detects_altered_added_and_missing_files(tmp_path, damage, expected):
    from src.session8_evidence import check_evidence, require_evidence

    cfg = evidence_project(tmp_path)
    assert check_evidence(cfg) == []
    sealed = seal_of(cfg).read_bytes()
    damage(tmp_path / "work")
    assert any(expected in p for p in check_evidence(cfg))
    with pytest.raises(RuntimeError, match="nothing is derived"):
        require_evidence(cfg)
    assert seal_of(cfg).read_bytes() == sealed                              # never rewritten


def test_the_lock_and_the_pre_audit_copies_are_the_only_exceptions(tmp_path):
    from src.session8_evidence import check_evidence

    cfg = evidence_project(tmp_path)
    (tmp_path / "work/execution.lock").write_text("1")
    (tmp_path / "work/remediation/before").mkdir(parents=True)
    (tmp_path / "work/remediation/before/old.csv").write_text("x")
    assert check_evidence(cfg) == []


@pytest.mark.parametrize("remove", [True, False])
def test_a_missing_or_tampered_seal_fails_closed_and_is_not_recreated(tmp_path, remove, monkeypatch):
    import src.run_session8 as runner
    from src.session8_evidence import check_evidence

    cfg = evidence_project(tmp_path)
    if remove:
        seal_of(cfg).unlink()
    else:
        seal_of(cfg).write_text(seal_of(cfg).read_text().replace('"note": "n"', '"note": "m"'))
    before = seal_of(cfg).read_bytes() if not remove else None
    expected = "missing" if remove else "differs from the trusted reference"
    assert any(expected in p for p in check_evidence(cfg))
    monkeypatch.setattr(runner, "load_canonical", lambda *a: pytest.fail("analysis continued"))
    for stage in (runner.stage_derive, runner.stage_report, runner.derived_tables):
        with pytest.raises(RuntimeError, match="nothing is derived"):
            stage(cfg)
    assert seal_of(cfg).exists() is not remove and (remove or seal_of(cfg).read_bytes() == before)


@pytest.mark.parametrize("remove_reference_entry", [False, True])
def test_forged_predictions_are_rejected_even_after_the_seal_is_removed(tmp_path, remove_reference_entry):
    from src.session7_execution import atomic_write_json
    from src.session8_evidence import build_evidence, check_evidence, register_evidence

    cfg = evidence_project(tmp_path)
    forge(cfg)
    seal_of(cfg).unlink()
    assert check_evidence(cfg) == ["the evidence manifest is missing"]
    with pytest.raises(RuntimeError, match="never run into or registered again"):
        register_evidence(cfg, "again", authorization=AUTH)                 # the reference covers the location
    assert not seal_of(cfg).exists()
    atomic_write_json(seal_of(cfg), {"note": "n", **build_evidence(cfg)})   # a seal written by other means
    assert "the evidence manifest differs from the trusted reference" in check_evidence(cfg)
    if remove_reference_entry:                  # without any reviewed reference nothing is accepted either
        (tmp_path / "reference.json").write_text('{"executions": []}')
        assert check_evidence(cfg) == ["there is no trusted reference for this execution"]


def test_seal_creation_is_not_reachable_from_the_safe_command_line():
    import src.run_session8 as runner

    assert "seal" not in runner.STAGES and "register" in runner.EXECUTION_STAGES
    assert not set(runner.SAFE_STAGES) & {"seal", "register", "all"}
    with pytest.raises(SystemExit):
        runner.main(["--stage", "seal"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "all", "--authorize"])                      # no abbreviated authorization


def test_a_changed_mlflow_run_is_detected(tmp_path):
    import sqlite3

    from src.session8_evidence import check_evidence

    cfg = evidence_project(tmp_path)
    db = sqlite3.connect(tmp_path / "mlruns/mlflow.db")
    db.execute("update params set value = 'False'")
    db.commit()
    db.close()
    assert "mlflow_runs: changed r1" in check_evidence(cfg)


# ---- second remediation: the published manifest and the sources ---------------------------------

FIGURE_NAMES = ("fig_session8_a.png",)


def published_project(tmp_path):
    from src.session8_evidence import evidence_state

    cfg = evidence_project(tmp_path)
    artifacts, work = tmp_path / "artifacts", tmp_path / "work"
    sealed = json.loads(seal_of(cfg).read_text())
    (artifacts / "table.csv").write_text("a\n1\n")
    (tmp_path / "figures" / FIGURE_NAMES[0]).write_bytes(b"png")
    (tmp_path / "report.md").write_text("report")
    derivation = {"git": {"git_commit": "a", "git_dirty": True}, "source_hashes": {"src/x.py": "1"}}
    canonical, evidence_class = evidence_state(sealed["execution"], derivation)
    records = {"x": {"mlflow_run_id": "r", "files": {}}}
    manifest = {
        "schema_version": 2, "session": 8, "config_sha256": "c", "protocol_sha256": "p",
        "evidence_manifest": {"file": "evidence_manifest.json", "sha256": sha256_file(seal_of(cfg))},
        "execution": sealed["execution"], "derivation": derivation, "canonical": canonical,
        "evidence_class": evidence_class, "session7_manifest_sha256": sealed["session7_manifest_sha256"],
        "datasets": json.loads((work / "dataset_facts.json").read_text()), "fd003_sha256": cfg["fd003"]["sha256"],
        "fd003_test_label_accesses": [], "fd001_official_test_arrays_opened": False,
        "models_trained_for_main_results": 0, "sensitivity_fits": 1, "checkpoints": records,
        "files": {n: sha256_file(artifacts / n) for n in ("evidence_manifest.json", "table.csv")},
        "figures": {str(tmp_path / "figures" / n): sha256_file(tmp_path / "figures" / n) for n in FIGURE_NAMES},
        "report_sha256": sha256_file(tmp_path / "report.md")}
    write_manifest(cfg, manifest)
    return cfg, records, manifest


def write_manifest(cfg, manifest, reviewed=True):
    path = seal_of(cfg).with_name("manifest.json")
    path.write_text(json.dumps(manifest))
    if reviewed:
        set_reference(cfg, artifacts_manifest_sha256=sha256_file(path))


@pytest.mark.parametrize("field,value", [
    ("canonical", True), ("canonical", None), ("canonical", 0), ("evidence_class", "canonical"),
    ("execution", {}), ("fd003_test_label_accesses", [{"purpose": "x"}]), ("evidence_manifest", {"sha256": "0"}),
    ("session7_manifest_sha256", "0"), ("sensitivity_fits", 0), ("models_trained_for_main_results", 1),
    ("fd001_official_test_arrays_opened", True), ("checkpoints", {}), ("datasets", {}), ("fd003_sha256", {}),
    ("report_sha256", "0"), ("figures", {}), ("files", {}), ("schema_version", 1), ("published_commit", "abc")])
def test_a_forged_manifest_field_is_rejected_even_if_its_hash_were_accepted(tmp_path, field, value):
    from src.session8_evidence import check_manifest

    cfg, records, manifest = published_project(tmp_path)
    assert check_manifest(cfg, records, FIGURE_NAMES) == []
    write_manifest(cfg, {**manifest, field: value}, reviewed=False)
    assert "manifest.json differs from the trusted reference" in check_manifest(cfg, records, FIGURE_NAMES)
    write_manifest(cfg, {**manifest, field: value})                 # the field checks stand on their own
    assert check_manifest(cfg, records, FIGURE_NAMES)


def test_a_canonical_claim_is_rejected_even_when_the_reference_agrees(tmp_path):
    from src.session8_evidence import check_manifest

    cfg, records, manifest = published_project(tmp_path)
    set_reference(cfg, canonical=True, evidence_class="canonical")
    write_manifest(cfg, {**manifest, "canonical": True, "evidence_class": "canonical"})
    assert any("canonical" in p for p in check_manifest(cfg, records, FIGURE_NAMES))   # the evidence says no


def test_a_later_clean_commit_does_not_change_the_classification(tmp_path, monkeypatch):
    from src import session7_integrity
    from src.session8_evidence import check_manifest

    cfg, records, manifest = published_project(tmp_path)
    monkeypatch.setattr(session7_integrity, "git_state", lambda *a: pytest.fail("the current HEAD was consulted"))
    assert check_manifest(cfg, records, FIGURE_NAMES) == []
    later = {**manifest["derivation"], "git": {"git_commit": "later", "git_dirty": False}}
    write_manifest(cfg, {**manifest, "derivation": later})
    assert check_manifest(cfg, records, FIGURE_NAMES) == []          # still not canonical, and accepted as such
    write_manifest(cfg, {**manifest, "derivation": later, "canonical": None,
                         "evidence_class": "executed and derived from one clean commit"})
    assert check_manifest(cfg, records, FIGURE_NAMES)


def test_added_published_files_are_rejected(tmp_path):
    from src.session8_evidence import check_manifest

    cfg, records, _ = published_project(tmp_path)
    (tmp_path / "artifacts/extra.csv").write_text("x")
    (tmp_path / "figures/fig99_session8_extra.png").write_bytes(b"x")
    problems = check_manifest(cfg, records, FIGURE_NAMES)
    assert any("extra.csv" in p for p in problems) and any("fig99" in p for p in problems)


def test_source_changes_after_derivation_must_be_the_reviewed_ones(tmp_path):
    from src.session8_evidence import check_sources

    cfg, _, _ = published_project(tmp_path)
    assert check_sources(cfg, {"src/x.py": "1"}) == []
    assert check_sources(cfg, {"src/x.py": "2"}) and check_sources(cfg, {"src/x.py": "1", "src/new.py": "3"})
    assert check_sources(cfg, {})                                    # a listed source disappeared
    set_reference(cfg, sources_changed_after_derivation={"src/x.py": {"derived_with": "1", "current": "2"},
                                                         "src/new.py": {"derived_with": None, "current": "3"}})
    assert check_sources(cfg, {"src/x.py": "2", "src/new.py": "3"}) == []
    assert check_sources(cfg, {"src/x.py": "4", "src/new.py": "3"})


# ---- second remediation: authorization and isolation --------------------------------------------

def unauthorized_calls():
    import src.run_session8 as runner
    from src.session8_evidence import register_evidence
    from src.session8_sensitivity import run_fits

    return {"labels": lambda cfg, **k: s8.load_test_labels(cfg, "x", **k),
            "load_model": lambda cfg, **k: inference.load_model({}, ".", {}, {}, **k),
            "inference": lambda cfg, **k: inference.predict_all(cfg, None, None, **k),
            "training": lambda cfg, **k: run_fits(cfg, None, **k),
            "mlflow": lambda cfg, **k: runner.Tracking(cfg, **k),
            "register": lambda cfg, **k: register_evidence(cfg, "n", **k),
            "predict": runner.stage_predict, "evaluate": runner.stage_evaluate,
            "sensitivity": runner.stage_sensitivity}


OPERATIONS = ("labels", "load_model", "inference", "training", "mlflow", "register", "predict", "evaluate",
              "sensitivity")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("authorization", [None, True, "--authorize-execution", ExecutionAuthorization("")])
def test_direct_calls_are_refused_before_any_side_effect(operation, authorization):
    call = unauthorized_calls()[operation]
    # cfg is None: any use of it, before the authorization check, would fail differently
    with pytest.raises(PermissionError, match="ExecutionAuthorization"):
        call(None, authorization=authorization)
    if authorization is None:
        with pytest.raises(PermissionError):
            call(None)


@pytest.mark.parametrize("operation", [o for o in OPERATIONS if o != "load_model"])
def test_authorized_calls_still_cannot_run_into_a_completed_execution(operation, tmp_path):
    cfg = evidence_project(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="never run into or registered again"):
        unauthorized_calls()[operation](cfg, authorization=AUTH)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_the_command_line_refuses_a_completed_execution_before_writing(tmp_path, monkeypatch, capsys):
    import src.run_session8 as runner

    cfg = evidence_project(tmp_path)
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "execution_lock", lambda *a: pytest.fail("a lock was taken"))
    for stage in runner.EXECUTION_STAGES:
        assert runner.main(["--stage", stage, "--authorize-execution"]) == 1
        assert "never run into" in capsys.readouterr().err


@pytest.mark.parametrize("shared", ["working_dir", "artifacts_dir", "figures_dir", "report", "experiment_name",
                                    "inside", "around"])
def test_a_new_execution_needs_its_own_identity_and_locations(tmp_path, shared):
    old = evidence_project(tmp_path / "old")
    new = {"_reference": old["_reference"], "sensitivity": {"experiment_name": "rerun"},
           "outputs": {k: str(tmp_path / "new" / k) for k in ("working_dir", "artifacts_dir", "figures_dir", "report")}}
    require_new_location(new)
    assert not (tmp_path / "new").exists()                                   # the check writes nothing
    if shared == "experiment_name":
        new["sensitivity"]["experiment_name"] = "e8"
    elif shared == "inside":
        new["outputs"]["working_dir"] = old["outputs"]["working_dir"] + "/rerun"
    elif shared == "around":
        new["outputs"]["working_dir"] = str(tmp_path / "old")
    else:
        new["outputs"][shared] = old["outputs"][shared].upper() if shared == "report" else old["outputs"][shared]
    with pytest.raises(RuntimeError, match="its own"):
        require_new_location(new)


def test_a_new_execution_registers_once_and_is_not_accepted_without_review(tmp_path):
    from src.session8_evidence import check_evidence, register_evidence

    cfg = evidence_project(tmp_path, register=False)
    register_evidence(cfg, "n", authorization=AUTH)
    assert check_evidence(cfg) == ["there is no trusted reference for this execution"]
    with pytest.raises(RuntimeError, match="exists"):
        register_evidence(cfg, "n", authorization=AUTH)


def test_a_missing_reference_opens_nothing(tmp_path):
    cfg = evidence_project(tmp_path)
    (tmp_path / "reference.json").unlink()
    with pytest.raises(RuntimeError, match="trusted reference"):
        require_new_location(cfg)


def test_canonical_status_needs_recorded_clean_matching_execution():
    from src.session8_evidence import evidence_state

    clean = {"git_commit": "a", "git_dirty": False}
    sources = {"src/x.py": "1"}
    execution = {"predictions": {"files": 2, "with_recorded_producer": 2,
                                 "producers": [{**clean, "source_hashes": sources}]},
                 "fits": {"git_states": [clean]}, "evaluation_marker_git": clean}
    derivation = {"git": clean, "source_hashes": sources}
    assert evidence_state(execution, derivation)[0] is None
    spoilers = [
        {**execution, "predictions": {"files": 2, "with_recorded_producer": 0, "producers": []}},    # cached
        {**execution, "predictions": {**execution["predictions"], "with_recorded_producer": 1}},
        {**execution, "evaluation_marker_git": None},
        {**execution, "fits": {"git_states": [{**clean, "git_dirty": True}]}},
        {**execution, "fits": {"git_states": [{**clean, "git_commit": "b"}]}}]
    for bad in spoilers:
        assert evidence_state(bad, derivation)[0] is False
    later = {"git": {"git_commit": "b", "git_dirty": False}, "source_hashes": sources}        # a later commit
    assert evidence_state(execution, later)[0] is False
    assert evidence_state(execution, {**derivation, "source_hashes": {"src/x.py": "2"}})[0] is False


def test_saved_predictions_are_read_without_loading_any_model(tmp_path, monkeypatch):
    from src.session8_evidence import load_saved_predictions

    monkeypatch.setattr(inference, "load_model", lambda *a: pytest.fail("a model was loaded"))
    ws = s8.Windows(np.zeros((2, 30, 14), "float32"), np.array([1, 2]), np.array([30, 30]))

    class Canonical:
        conditions = {"x": {"experiment_id": "x", "identity_sha256": "i"}}
        records = {"x": {"files": {}, "predictions_sha256": "0"}}

    cfg = {"outputs": {"working_dir": str(tmp_path)}}
    with pytest.raises(RuntimeError, match="not recomputed"):
        load_saved_predictions(cfg, Canonical(), {k: ws for k in s8.DATASETS})


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(), lambda rows: rows[0].update(prediction=99.0), lambda rows: rows[3].update(true_rul_raw=1.0)])
def test_saved_test_labels_must_agree_with_the_stored_predictions(tmp_path, change):
    from src.session8_evidence import saved_test_labels

    predictions = {"fd003_test": {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}}
    rows = [{"experiment_id": k, "unit_id": u + 1, "true_rul_raw": 10.0 + u, "prediction": float(v)}
            for k, p in predictions["fd003_test"].items() for u, v in enumerate(p)]
    cfg = {"outputs": {"working_dir": str(tmp_path)}}
    write_rows(tmp_path / "fd003_test_predictions.csv", rows)
    assert saved_test_labels(cfg, predictions).tolist() == [10, 11]
    change(rows)
    write_rows(tmp_path / "fd003_test_predictions.csv", rows)
    with pytest.raises(ValueError):
        saved_test_labels(cfg, predictions)


@pytest.mark.parametrize("stage", ["predict", "evaluate", "sensitivity", "register", "all"])
def test_execution_stages_need_deliberate_authorization(stage, monkeypatch, capsys):
    import src.run_session8 as runner

    monkeypatch.setattr(runner, "load_config", lambda path: pytest.fail("work was started"))
    assert runner.main(["--stage", stage]) == 1 and "--authorize-execution" in capsys.readouterr().err


def test_a_control_refit_that_differs_is_refused_even_when_reused():
    from src.session8_sensitivity import require_control

    require_control("k", {"predictions_sha256": "a"}, {"predictions_sha256": "a"})
    with pytest.raises(RuntimeError, match="does not reproduce"):
        require_control("k", {"predictions_sha256": "a"}, {"predictions_sha256": "b"})


def test_the_published_notebook_was_executed_without_errors_and_is_portable():
    text = (PROJECT_ROOT / "notebooks/session8_error_analysis.ipynb").read_text()
    code = [c for c in json.loads(text)["cells"] if c["cell_type"] == "code"]
    assert [c["execution_count"] for c in code] == list(range(1, len(code) + 1))    # one clean top-to-bottom run
    assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])
    assert "/Users/" not in text and "/home/" not in text


# ---- second remediation: the reviewed execution itself (read only) ------------------------------

@published
def test_the_project_reference_protects_the_reviewed_execution():
    from src.session8_evidence import trusted_reference

    cfg = s8.load_config(CONFIG)
    entry = trusted_reference(cfg)
    assert entry["canonical"] is False
    assert sha256_file(ARTIFACTS / "evidence_manifest.json") == entry["evidence_manifest_sha256"]
    assert sha256_file(ARTIFACTS / "manifest.json") == entry["artifacts_manifest_sha256"]
    with pytest.raises(RuntimeError, match="never run into or registered again"):
        require_new_location(cfg)


@published
def test_saved_evidence_is_analysed_without_labels_inference_or_training(monkeypatch):
    import src.run_session8 as runner
    import src.session8_sensitivity as sensitivity

    if not (PROJECT_ROOT / CFG["outputs"]["working_dir"] / "fd003_test_predictions.csv").exists():
        pytest.skip("local evidence is not present")
    for module, name in ((s8, "load_test_labels"), (s8, "read_rul_file"), (runner, "load_test_labels"),
                         (runner, "predict_all"), (runner, "run_fits"), (runner, "register_evidence"),
                         (inference, "load_model"), (sensitivity, "run_fits")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("an execution operation was called"))
    ledger = PROJECT_ROOT / CFG["outputs"]["working_dir"] / "fd003_test_access.json"
    before = ledger.read_bytes()
    assert len(runner.derived_tables(s8.load_config(CONFIG))["master_results"]) == 174
    assert ledger.read_bytes() == before
