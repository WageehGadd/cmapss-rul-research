"""Selection logic, config validation and an end-to-end Session 5 run on synthetic windows."""
import hashlib
import json

import numpy as np
import pytest
import yaml

import src.run_session5 as run_session5
from src.benchmark_config import PROJECT_ROOT, resolve_path
from src.convolutional_study import (candidate_grid, load_historical_representatives, select_candidate,
                                     threshold_near_miss)
from src.session5_outputs import latency_ratio_notes, near_miss_sentence, plot_rmse_vs_latency
from src.session5_config import check_protocol_matches_reference, load_session5_config
from tests.test_session4_pipeline import synthetic_sequence_data

PROCESSED = "data/processed/fd001"
needs_session1 = pytest.mark.skipif(not (resolve_path(PROCESSED) / "manifest.json").exists(),
                                    reason="Session 1 artifacts not generated")


def rec(candidate, rmse, params, latency, status="ok", family="cnn"):
    return {"candidate": candidate, "validation": {"rmse": rmse}, "n_parameters": params, "status": status,
            "latency_ms_per_batch": latency, "family": family}


# ---- selection rule ----------------------------------------------------------------------------------
def test_selection_lowest_rmse_when_nothing_is_equivalent():
    chosen, rule = select_candidate([rec(0, 10.0, 900, 1.0), rec(1, 12.0, 10, 0.1)], "rmse", 0.01)
    assert chosen["candidate"] == 0 and rule["n_equivalent"] == 1 and rule["decided_by"] == "validation metric"


def test_selection_ties_prefer_fewer_parameters_then_lower_latency_then_grid_order():
    by_params, r1 = select_candidate([rec(0, 10.0, 900, 1.0), rec(1, 10.05, 100, 5.0), rec(2, 11.0, 5, 0.1)], "rmse", 0.01)
    assert by_params["candidate"] == 1 and r1["decided_by"] == "parameter count"
    by_latency, r2 = select_candidate([rec(0, 10.0, 100, 3.0), rec(1, 10.02, 100, 1.0)], "rmse", 0.01)
    assert by_latency["candidate"] == 1 and r2["decided_by"] == "inference latency or grid order"
    by_order, _ = select_candidate([rec(0, 10.0, 100, 1.0), rec(1, 10.02, 100, 1.0)], "rmse", 0.01)
    assert by_order["candidate"] == 0


def test_selection_skips_unstable_runs_and_uses_only_validation_values():
    chosen, _ = select_candidate([rec(0, 1.0, 1, 0.1, status="unstable"), rec(1, 10.0, 50, 1.0)], "rmse", 0.01)
    assert chosen["candidate"] == 1
    with pytest.raises(ValueError, match="no stable"):
        select_candidate([rec(0, 1.0, 1, 0.1, status="unstable")])
    with_test = [{**rec(0, 10.0, 100, 1.0), "test": {"rmse": 99.0}}, {**rec(1, 10.0, 200, 1.0), "test": {"rmse": 1.0}}]
    assert select_candidate(with_test)[0]["candidate"] == 0               # test values are ignored


def test_candidate_grids_are_the_declared_eight_per_family():
    cfg, _, _ = load_session5_config(PROJECT_ROOT / "configs/session5_convolutional.yaml")
    cnn, tcn = candidate_grid(cfg, "cnn"), candidate_grid(cfg, "tcn")
    assert len(cnn) == len(tcn) == 8
    assert {(a["channels"], a["kernel_size"], a["depth"]) for a in cnn} == \
        {(c, k, d) for c in (32, 64) for k in (3, 5) for d in (2, 3)}
    assert {(a["channels"], a["kernel_size"], tuple(a["dilations"])) for a in tcn} == \
        {(c, k, d) for c in (32, 64) for k in (3, 5) for d in ((1, 2, 4), (1, 2, 4, 8))}
    assert cnn[0] == {"channels": 32, "kernel_size": 3, "depth": 2}       # channels outermost, declared order


# ---- config ------------------------------------------------------------------------------------------------
def test_real_config_is_valid_and_matches_the_session4_protocol():
    cfg, _, _ = load_session5_config(PROJECT_ROOT / "configs/session5_convolutional.yaml")
    check_protocol_matches_reference(cfg)
    assert cfg["input"] == {"n_sensors": 14, "sequence_length": 30} and cfg["benchmark"]["seed"] == 42
    assert cfg["training"]["batch_size"] == 256 and cfg["training"]["max_epochs"] == 60


def test_config_validation_errors(tmp_path):
    src = PROJECT_ROOT / "configs/session5_convolutional.yaml"
    base = yaml.safe_load(src.read_text())

    def write(mutate):
        cfg = yaml.safe_load(src.read_text())
        mutate(cfg)
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    with pytest.raises(ValueError, match="missing section"):
        write(lambda c: c.pop("selection")) and load_session5_config(tmp_path / "c.yaml")
    with pytest.raises(ValueError, match="adam"):
        load_session5_config(write(lambda c: c["training"].update(optimizer="sgd")))
    with pytest.raises(ValueError, match="does not fit"):
        load_session5_config(write(lambda c: c["architecture"]["cnn"].update(depth=[9], kernel_size=[5])))
    with pytest.raises(ValueError, match="positive integers"):
        load_session5_config(write(lambda c: c["architecture"]["tcn"].update(dilations=[[1, 0]])))
    cfg, _, _ = load_session5_config(write(lambda c: c["training"].update(learning_rate=0.5)))
    with pytest.raises(ValueError, match="differs from"):
        check_protocol_matches_reference(cfg)                              # protocol drift is rejected
    assert base["protocol_reference"]["require_identical_training"] is True


def test_historical_representatives_are_read_only_and_canonical_rows_only(tmp_path):
    rows = [{"model": "lstm", "eval_set": "validation", "git_dirty": False, "rmse": 1.0},
            {"model": "lstm", "eval_set": "test", "git_dirty": False, "rmse": 2.0},
            {"model": "gru", "eval_set": "validation", "git_dirty": True, "rmse": 3.0}]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"results": rows}))
    before = path.read_bytes()
    out = load_historical_representatives(path)
    assert set(out) == {"lstm"} and out["lstm"]["test"]["rmse"] == 2.0
    assert path.read_bytes() == before and load_historical_representatives(tmp_path / "none.json") == {}


# ---- end to end on synthetic windows ----------------------------------------------------------------------
def write_config(tmp_path):
    cfg = yaml.safe_load((PROJECT_ROOT / "configs/session5_convolutional.yaml").read_text())
    cfg["session2"] = {"config": str(tmp_path / "session2.yaml")}
    cfg["benchmark"]["seed"] = 5
    cfg["training"].update(max_epochs=3, early_stopping_patience=2, batch_size=64, learning_rate=0.01)
    cfg["protocol_reference"]["require_identical_training"] = False
    cfg["architecture"]["cnn"] = {"channels": [4], "kernel_size": [3], "depth": [1, 2]}
    cfg["architecture"]["tcn"] = {"channels": [4, 8], "kernel_size": [3], "dilations": [[1, 2]]}
    cfg["stability_gate"] = {"samples": 32, "epochs": 150, "learning_rate": 0.02, "min_loss_drop": 0.2}
    cfg["outputs"] = {"frozen_dir": str(tmp_path / "frozen"), "checkpoint_dir": str(tmp_path / "ckpt"),
                      "working_dir": str(tmp_path / "work"), "report": str(tmp_path / "report.md"),
                      "figures_dir": str(tmp_path / "figs")}
    s2 = {"session1": cfg["session1"], "benchmark": cfg["benchmark"],
          "baselines": [{"name": "mean_rul", "type": "mean_rul"}],
          "timing": {"warmup_runs": 1, "timed_runs": 2, "batch_size": 16, "latency_unit": "ms",
                     "timing_set": "validation", "hardware_description": "auto"},
          "tracking": {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_s5"},
          "outputs": {"results_csv": str(tmp_path / "results.csv"), "results_json": str(tmp_path / "results.json"),
                      "runs_dir": str(tmp_path / "runs")},
          "environment": {"library_names": ["numpy"]}}
    (tmp_path / "session2.yaml").write_text(yaml.safe_dump(s2))
    # a frozen earlier-family row so the comparison has something to read
    rows = [{"model": "lstm", "eval_set": s, "git_dirty": False, "rmse": 12.0, "mae": 9.0, "r2": 0.9, "nasa_score": 1.0,
             "inference_latency_ms_per_batch": 5.0, "train_time_s": 10.0, "n_parameters": 100, "run_id": "x",
             **{k: "" for k in ()}} for s in ("validation", "test")]
    (tmp_path / "results.json").write_text(json.dumps({"schema_version": 1, "columns": [], "results": rows}))
    path = tmp_path / "session5.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run(tmp_path, monkeypatch, test_shift=False):
    path = write_config(tmp_path)

    def loader(processed_dir, session1_config, target_type, L):
        d = synthetic_sequence_data(L)
        if test_shift:
            d.X["test"][:] = 7.0
            d.y["test"][:] = 111.0
        return d

    monkeypatch.setattr(run_session5, "load_sequence_data", loader)
    assert run_session5.main(["--config", str(path)]) == 0
    return tmp_path


def _flat_table_cols(tmp_path):
    return json.loads((tmp_path / "results.json").read_text())


@needs_session1
def test_end_to_end_freezes_artifacts_logs_runs_and_extends_the_table(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "Gate cnn" in out and "Final convolutional architecture" in out
    frozen = tmp_path / "frozen"
    names = {"selected_cnn.json", "selected_tcn.json", "selected_convolutional.json", "architecture_study.csv",
             "receptive_field.json", "manifest.json"}
    assert names <= {p.name for p in frozen.iterdir()}
    manifest = json.loads((frozen / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest
    for kind, ck in manifest["checkpoints"].items():
        assert hashlib.sha256((tmp_path / "ckpt" / f"selected_{kind}.pt").read_bytes()).hexdigest() == ck["file_sha256"]
    assert manifest["session1"]["sequence_length"] == 30 and manifest["sequence_length"] == 30
    final = json.loads((frozen / "selected_convolutional.json").read_text())
    assert final["final_family"] in ("cnn", "tcn") and final["selection"]["data"] == "validation only"
    assert final["retrain_matches_candidate_weights"] is True              # CPU training is repeatable
    for name in ("selected_cnn.json", "selected_tcn.json", "selected_convolutional.json"):
        assert "test" not in json.dumps(json.loads((frozen / name).read_text())["selection"])
    rf = json.loads((frozen / "receptive_field.json").read_text())
    assert rf["sequence_length"] == 30 and all("receptive_field" in c for c in rf["configurations"])

    table = _flat_table_cols(tmp_path)
    new = [r for r in table["results"] if r["model"] in ("cnn", "tcn")]
    assert [r["eval_set"] for r in new] == ["validation", "test"] and len({r["run_id"] for r in new}) == 1
    assert table["results"][:2][0]["model"] == "lstm"                      # earlier rows untouched, in front
    assert {"model_family", "channels", "kernel_size", "receptive_field", "sequence_length", "best_epoch",
            "epochs_run", "device", "n_parameters"} <= set(table["columns"])

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    runs = mlflow.search_runs([mlflow.get_experiment_by_name("pytest_s5").experiment_id], output_format="list")
    stages = [r.data.tags.get("stage", "final") for r in runs]
    assert len(runs) == 2 + 2 + 1 and sorted(set(stages)) == ["ablation_cnn", "ablation_tcn", "final"]
    assert not any(k.startswith("test_") for r in runs if r.data.tags.get("stage") for k in r.data.metrics)
    for r in runs:
        if r.data.tags.get("stage"):
            assert {"validation_rmse", "n_parameters", "best_epoch", "epochs_run", "train_time_s",
                    "latency_ms_per_batch", "throughput_samples_per_s"} <= set(r.data.metrics)

    report = (tmp_path / "report.md").read_text()
    for needle in ("Scientific caveats and interpretation", "Single seed", "zero padding", "did not participate",
                   "environment dependent", "Receptive field", "dropout -> `Linear(channels, 1)`", "Near-miss note",
                   "Timing interpretation", "Sensitivity to the 1% threshold", "Plain CNN receptive field"):
        assert needle in report                                            # caveats are generated, not patched in
    for fig in ("fig10_session5_training_curves.png", "fig11_session5_validation_prediction_vs_target.png",
                "fig12_session5_validation_rmse_vs_latency.png"):
        assert (tmp_path / "figs" / fig).exists()


@needs_session1
def test_test_split_cannot_influence_any_selection(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run(a, monkeypatch)
    run(b, monkeypatch, test_shift=True)
    for name in ("selected_cnn.json", "selected_tcn.json", "selected_convolutional.json"):
        x, y = (json.loads((r / "frozen" / name).read_text()) for r in (a, b))
        for doc in (x, y):
            doc.pop("checkpoint", None)
        assert x["architecture"] == y["architecture"]
        assert x.get("selection", {}).get("best_validation_value") == y.get("selection", {}).get("best_validation_value")
    ca = {p: json.loads((a / "frozen" / "manifest.json").read_text())["checkpoints"][p]["weights_sha256"] for p in ("cnn", "tcn")}
    cb = {p: json.loads((b / "frozen" / "manifest.json").read_text())["checkpoints"][p]["weights_sha256"] for p in ("cnn", "tcn")}
    assert ca == cb
    ra = [r for r in json.loads((a / "results.json").read_text())["results"] if r["model"] in ("cnn", "tcn")]
    rb = [r for r in json.loads((b / "results.json").read_text())["results"] if r["model"] in ("cnn", "tcn")]
    val = lambda rows: [r["predictions_sha256"] for r in rows if r["eval_set"] == "validation"]
    tst = lambda rows: [r["rmse"] for r in rows if r["eval_set"] == "test"]
    assert val(ra) == val(rb) and tst(ra) != tst(rb)


def test_stability_gate_failure_stops_the_study(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["stability_gate"]["min_loss_drop"] = 0.999999
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_session5, "load_sequence_data", lambda *a, **k: synthetic_sequence_data(30))
    assert run_session5.main(["--config", str(path)]) == 1


def test_protocol_guard_also_checks_the_execution_device():
    cfg, _, _ = load_session5_config(PROJECT_ROOT / "configs/session5_convolutional.yaml")
    check_protocol_matches_reference(cfg)
    cfg["execution"]["device"] = "mps"
    with pytest.raises(ValueError, match="execution.device differs"):
        check_protocol_matches_reference(cfg)


def test_threshold_near_miss_uses_unrounded_values_and_reports_the_smaller_candidate():
    recs = [rec(0, 17.451172808, 12609, 1.0), rec(1, 17.627003376, 7457, 1.0), rec(2, 17.7886, 25153, 1.0),
            rec(3, 18.0, 4513, 1.0)]
    for i, r in enumerate(recs):
        r["run_name"] = f"c{i}"
    chosen, rule = select_candidate(recs, "rmse", 0.01)
    assert chosen["candidate"] == 0 and rule["n_equivalent"] == 1            # 17.627 misses 17.62568
    miss = threshold_near_miss(recs, chosen, rule)
    assert miss["run_name"] == "c1" and miss["n_parameters"] == 7457
    assert miss["threshold"] == pytest.approx(17.451172808 * 1.01) and 0 < miss["margin"] < 0.0014
    text = near_miss_sentence("cnn", miss, rule)
    assert "sensitive to the exact threshold" in text and "7,457" in text and "smaller model would have won" in text
    # a candidate that is inside the threshold is equivalent, not a near miss
    inside = [rec(0, 10.0, 900, 1.0), rec(1, 10.05, 100, 1.0)]
    chosen2, rule2 = select_candidate(inside, "rmse", 0.01)
    assert chosen2["candidate"] == 1 and threshold_near_miss(inside, chosen2, rule2) is None
    assert "did not exclude" in near_miss_sentence("tcn", None, rule2)


def test_latency_notes_are_computed_from_the_rows_not_hard_coded():
    rows = [{"model": "gru", "latency_ms": 3.10}, {"model": "lstm", "latency_ms": 5.06},
            {"model": "tcn (tcn_x)", "latency_ms": 47.84}, {"model": "cnn (cnn_x)", "latency_ms": 2.0}]
    tcn = latency_ratio_notes(rows, "tcn")
    assert tcn[0].startswith("TCN latency is 15.4x that of GRU") and tcn[1].startswith("TCN latency is 9.5x that of LSTM")
    cnn = latency_ratio_notes(rows, "cnn")
    assert "lower than that of GRU" in cnn[0] and "1.6x" in cnn[0]
    rows[2]["latency_ms"] = 31.0
    assert "10.0x that of GRU" in latency_ratio_notes(rows, "tcn")[0]       # follows the data
    assert latency_ratio_notes([], "tcn") == []


def test_latency_figure_is_deterministic_and_keeps_every_point(tmp_path):
    points = [{"label": "lstm", "family": "recurrent", "latency": 5.06, "rmse": 12.99, "n_parameters": 14625},
              {"label": "random_forest", "family": "classical", "latency": 4.59, "rmse": 12.95, "n_parameters": 117656},
              {"label": "gradient_boosting", "family": "classical", "latency": 4.14, "rmse": 12.79, "n_parameters": 12196},
              {"label": "gru", "family": "recurrent", "latency": 3.10, "rmse": 12.44, "n_parameters": 15425}]
    plot_rmse_vs_latency(tmp_path / "a.png", points)
    plot_rmse_vs_latency(tmp_path / "b.png", list(reversed(points)))        # input order must not matter
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
