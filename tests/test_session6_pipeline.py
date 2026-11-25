"""Config validation, selection, report/manifest generation and an end-to-end Session 6 run on
synthetic windows."""
import hashlib
import json

import numpy as np
import pytest
import yaml

import src.run_session6 as run_session6
from src.benchmark_config import PROJECT_ROOT, resolve_path
from src.convolutional_study import select_candidate
from src.session6_config import check_protocol_matches_reference, load_session6_config
from src.session6_outputs import (comparison_note, early_stopping_text, head_count_spread, interpretation_notes,
                                  latency_notes, latency_ticks, plot_family_rmse_vs_latency)
from src.transformer_study import stability_gate, stage_a_grid
from tests.test_session4_pipeline import synthetic_sequence_data

CONFIG = PROJECT_ROOT / "configs/session6_transformer.yaml"
needs_session1 = pytest.mark.skipif(not (resolve_path("data/processed/fd001") / "manifest.json").exists(),
                                    reason="Session 1 artifacts not generated")


def test_declared_grid_has_eight_candidates_and_protocol_matches_session4():
    cfg, _, _ = load_session6_config(CONFIG)
    check_protocol_matches_reference(cfg)
    grid = stage_a_grid(cfg)
    assert len(grid) == 8 and {g["pooling"] for g in grid} == {"last"}
    assert {(g["d_model"], g["num_heads"], g["num_layers"]) for g in grid} == \
        {(d, h, n) for d in (32, 64) for h in (2, 4) for n in (1, 2)}
    assert cfg["study"]["stage_b"]["pooling"] == ["last", "mean"]
    assert cfg["input"] == {"n_sensors": 14, "sequence_length": 30} and cfg["benchmark"]["seed"] == 42


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c.pop("model"), "missing section"),
    (lambda c: c["study"]["stage_a"].update(num_heads=[3]), "divisible"),
    (lambda c: c["study"]["stage_a"].update(num_heads=[4.5]), "num_heads must be an integer"),
    (lambda c: c["study"]["stage_a"].update(d_model=[32.5]), "d_model must be an integer"),
    (lambda c: c["model"].update(ff_multiplier=1.5), "ff_multiplier must be an integer"),
    (lambda c: c["input"].update(sequence_length=30.5), "sequence_length must be an integer"),
    (lambda c: c["stability_gate"].update(session7_sensor_counts=[7.5]), "must be an integer"),
    (lambda c: c["study"]["stage_a"].update(pooling="max"), "unsupported pooling"),
    (lambda c: c["study"]["stage_b"].update(pooling=["last"]), "at least two"),
    (lambda c: c["model"].update(attention_mask="causal"), "attention_mask"),
    (lambda c: c["model"].update(positional_encoding="learned"), "positional_encoding"),
    (lambda c: c["training"].update(optimizer="sgd"), "adam"),
    (lambda c: c["study"].update(sensor_aware_variant=True), "not implemented"),
])
def test_config_validation(tmp_path, mutate, match):
    cfg = yaml.safe_load(CONFIG.read_text())
    mutate(cfg)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=match):
        load_session6_config(path)


@pytest.mark.parametrize("section,key,value,match", [("training", "learning_rate", 0.5, "training protocol differs"),
                                                     ("benchmark", "seed", 1, "seed differs"),
                                                     ("execution", "device", "mps", "execution.device differs")])
def test_protocol_drift_is_rejected(section, key, value, match):
    cfg, _, _ = load_session6_config(CONFIG)
    cfg[section][key] = value
    with pytest.raises(ValueError, match=match):
        check_protocol_matches_reference(cfg)


def _rec(candidate, rmse, params, latency):
    return {"candidate": candidate, "validation": {"rmse": rmse}, "n_parameters": params, "status": "ok",
            "latency_ms_per_batch": latency}


def test_head_count_ties_are_decided_by_latency_and_recorded():
    # 2 vs 4 heads: identical parameter count, within 1% -> latency decides, and says so
    chosen, rule = select_candidate([_rec(0, 10.0, 500, 3.0), _rec(1, 10.05, 500, 2.0)], "rmse", 0.01)
    assert chosen["candidate"] == 1 and rule["decided_by"] == "inference latency or grid order"
    chosen, _ = select_candidate([_rec(0, 10.0, 500, 3.0), _rec(1, 10.2, 100, 1.0)], "rmse", 0.01)
    assert chosen["candidate"] == 0                                     # outside 1%: RMSE decides


def test_generated_interpretation_follows_the_data():
    rows = [{"model": "gru", "val_rmse": 12.0, "latency_ms": 3.0}, {"model": "transformer", "val_rmse": 13.2, "latency_ms": 9.0}]
    assert "ranks 2 of 2" in comparison_note(rows, 0.01) and "10.0% higher" in comparison_note(rows, 0.01)
    assert latency_notes(rows) == ["3.0x the latency of gru"]
    rows[1].update(val_rmse=11.95, latency_ms=1.5)
    note = comparison_note(rows, 0.01)
    assert "ranks 1 of 2" in note and "within the 1% equivalence band" in note
    assert latency_notes(rows) == ["2.0x faster than gru"]


def test_stability_gate_passes_on_learnable_data_and_fails_when_it_should():
    cfg, _, _ = load_session6_config(CONFIG)
    cfg["stability_gate"].update(samples=32, epochs=150, min_loss_drop=0.5)
    data = synthetic_sequence_data(30)
    arch = stage_a_grid(cfg)[0]
    gate = stability_gate(cfg, arch, data)
    assert gate["passed"], gate["checks"]
    for key in ("forward_shape_is_batch", "loss_and_gradients_finite", "tiny_batch_loss_decreases",
                "target_scaling_inverted_in_predict", "save_load_identical_predictions", "deterministic_initialisation",
                "deterministic_training", "device_as_configured", "train_and_validation_engines_disjoint",
                "predictions_finite"):
        assert key in gate["checks"]
    cfg["stability_gate"]["min_loss_drop"] = 0.999999
    assert not stability_gate(cfg, arch, data)["passed"]
    data.unit_ids["validation"][:] = data.unit_ids["train"][0]            # leakage between splits is detected
    assert not stability_gate(cfg, arch, data)["checks"]["train_and_validation_engines_disjoint"]


# ---- end to end ----------------------------------------------------------------------------------------
def write_config(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["session2"] = {"config": str(tmp_path / "session2.yaml")}
    cfg["training"].update(max_epochs=3, early_stopping_patience=2, batch_size=64, learning_rate=0.01)
    cfg["protocol_reference"]["require_identical_training"] = False
    cfg["study"]["stage_a"] = {"d_model": [8, 16], "num_heads": [2], "num_layers": [1], "pooling": "last"}
    cfg["stability_gate"] = {"samples": 32, "epochs": 150, "learning_rate": 0.02, "min_loss_drop": 0.2,
                             "session7_sensor_counts": [14, 10, 7, 4, 2]}
    cfg["outputs"] = {"frozen_dir": str(tmp_path / "frozen"), "checkpoint_dir": str(tmp_path / "ckpt"),
                      "working_dir": str(tmp_path / "work"), "report": str(tmp_path / "report.md"),
                      "figures_dir": str(tmp_path / "figs")}
    s2 = {"session1": cfg["session1"], "benchmark": cfg["benchmark"],
          "baselines": [{"name": "mean_rul", "type": "mean_rul"}],
          "timing": {"warmup_runs": 1, "timed_runs": 2, "batch_size": 16, "latency_unit": "ms",
                     "timing_set": "validation", "hardware_description": "auto"},
          "tracking": {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_s6"},
          "outputs": {"results_csv": str(tmp_path / "results.csv"), "results_json": str(tmp_path / "results.json"),
                      "runs_dir": str(tmp_path / "runs")},
          "environment": {"library_names": ["numpy"]}}
    (tmp_path / "session2.yaml").write_text(yaml.safe_dump(s2))
    rows = [{"model": m, "eval_set": "validation", "git_dirty": False, "rmse": 12.0 + i, "mae": 9.0, "r2": 0.9,
             "nasa_score": 1.0, "inference_latency_ms_per_batch": 4.0 + i, "train_time_s": 10.0, "n_parameters": 100,
             "run_id": "x"} for i, m in enumerate(run_session6.HISTORICAL)]
    (tmp_path / "results.json").write_text(json.dumps({"schema_version": 1, "columns": [], "results": rows}))
    path = tmp_path / "session6.yaml"
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

    monkeypatch.setattr(run_session6, "load_sequence_data", loader)
    assert run_session6.main(["--config", str(path)]) == 0
    return tmp_path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_session1
def test_end_to_end_artifacts_manifest_report_table_and_mlflow(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "Gate tf_" in out and "Stage A leader" in out and "Selected Transformer" in out
    frozen = tmp_path / "frozen"
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert set(manifest["files"]) == {"selected_transformer.json", "architecture_study.csv", "pooling_study.json"}
    for name, digest in manifest["files"].items():
        assert sha(frozen / name) == digest
    assert sha(tmp_path / "ckpt" / "selected_transformer.pt") == manifest["checkpoint"]["file_sha256"]
    for key in ("config_sha256", "session1", "session5_manifest_sha256", "protocol_reference_sha256", "seed", "device",
                "n_parameters", "final_run_id", "candidate_run_ids", "git_commit", "git_dirty"):
        assert key in manifest
    definition = json.loads((frozen / "selected_transformer.json").read_text())
    assert definition["retrain_matches_candidate_weights"] is True
    assert definition["selection"]["data"] == "validation only" and "test" not in json.dumps(definition["selection"])
    for key in ("d_model", "num_heads", "num_layers", "ff_dim", "dropout", "pooling", "positional_encoding",
                "normalization", "attention_mask"):
        assert key in definition["architecture"]

    table = json.loads((tmp_path / "results.json").read_text())
    new = [r for r in table["results"] if r["model"] == "transformer"]
    assert [r["eval_set"] for r in new] == ["validation", "test"] and len(table["results"]) == 5 + 2
    assert {"d_model", "num_heads", "pooling", "ff_dim", "best_epoch", "n_parameters"} <= set(table["columns"])

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    runs = mlflow.search_runs([mlflow.get_experiment_by_name("pytest_s6").experiment_id], output_format="list")
    stages = sorted(r.data.tags.get("stage", "final") for r in runs)
    assert stages == ["ablation_transformer", "ablation_transformer", "final", "pooling_transformer"]
    for r in runs:
        if r.data.tags.get("stage"):
            assert not any(k.startswith("test_") for k in r.data.metrics)
            assert {"validation_rmse", "validation_nasa_score", "n_parameters", "best_epoch", "epochs_run",
                    "train_time_s", "latency_ms_per_batch", "throughput_samples_per_s"} <= set(r.data.metrics)

    report = (tmp_path / "report.md").read_text()
    for needle in ("Positional encoding", "Attention-mask policy", "Stability gate", "Pooling comparison",
                   "Single seed", "Head count does not change", "Test data did not participate",
                   "Session 7 compatibility", "Not attempted", "Timing interpretation", "verification run",
                   "ranks", "Early-stopping semantics", "min_delta", "no statistical superiority",
                   "Validation RMSE: Transformer",
                   "one scalar prediction for each of the three samples", "never truncated"):
        assert (needle in report) or (needle == "verification run" and manifest["git_dirty"] is False)
    for fig in ("fig13_session6_training_curves.png", "fig14_session6_validation_prediction_vs_target.png",
                "fig15_session6_architecture_study.png", "fig16_session6_validation_rmse_vs_latency.png"):
        assert (tmp_path / "figs" / fig).exists()


@needs_session1
def test_test_split_cannot_influence_selection(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run(a, monkeypatch)
    run(b, monkeypatch, test_shift=True)
    da, db = (json.loads((r / "frozen" / "selected_transformer.json").read_text()) for r in (a, b))
    assert da["architecture"] == db["architecture"]
    assert da["checkpoint"]["weights_sha256"] == db["checkpoint"]["weights_sha256"]
    assert da["validation_metrics"] == db["validation_metrics"]
    pa, pb = (json.loads((r / "frozen" / "pooling_study.json").read_text()) for r in (a, b))
    assert [x["rmse"] for x in pa["results"]] == [x["rmse"] for x in pb["results"]]
    rows = lambda root: [r for r in json.loads((root / "results.json").read_text())["results"] if r["model"] == "transformer"]
    assert rows(a)[0]["predictions_sha256"] == rows(b)[0]["predictions_sha256"]
    assert rows(a)[1]["rmse"] != rows(b)[1]["rmse"]                        # test only changes its own score


def test_failing_gate_stops_before_any_candidate(tmp_path, monkeypatch):
    path = write_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["stability_gate"]["min_loss_drop"] = 0.999999
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_session6, "load_sequence_data", lambda *a, **k: synthetic_sequence_data(30))
    assert run_session6.main(["--config", str(path)]) == 1
    assert not (tmp_path / "mlruns").exists() and not (tmp_path / "frozen").exists()


def test_early_stopping_summary_follows_the_min_delta_rule():
    history = [{"epoch": 1, "val_loss": 0.5}, {"epoch": 2, "val_loss": 0.011884}, {"epoch": 3, "val_loss": 0.011877}]
    other = [{"epoch": 1, "val_loss": 0.3}, {"epoch": 2, "val_loss": 0.2}]
    records = [{"run_name": "a", "best_epoch": 2, "history": history}, {"run_name": "b", "best_epoch": 2, "history": other}]
    summary = run_session6.early_stopping_summary(records, "b", 1e-5)
    assert [d["run_name"] for d in summary["differing"]] == ["a"]
    assert 0 < summary["differing"][0]["gap"] < 1e-5 and summary["selected"]["gap"] == 0
    text = " ".join(early_stopping_text(summary))
    assert "min_delta" in text and "also its absolute validation-loss minimum" in text and "`a`" in text
    assert "differs from" in " ".join(early_stopping_text(run_session6.early_stopping_summary(records, "a", 1e-5)))


def test_interpretation_notes_are_derived_from_the_records():
    comparison = [{"model": "gru", "val_rmse": 12.44, "val_nasa": 9000.0, "n_parameters": 15000, "latency_ms": 3.0},
                  {"model": "random_forest", "val_rmse": 12.95, "val_nasa": 12000.0, "n_parameters": 100000, "latency_ms": 4.5},
                  {"model": "transformer", "val_rmse": 12.32, "val_nasa": 10600.0, "n_parameters": 68000, "latency_ms": 10.0}]
    stage_a = [{"d_model": 32, "num_layers": 1, "validation": {"rmse": 13.16}},
               {"d_model": 32, "num_layers": 1, "validation": {"rmse": 13.63}}]
    assert head_count_spread(stage_a) == pytest.approx(0.47)
    text = " ".join(interpretation_notes(comparison, {"random_forest": 11.59, "gru": 14.3}, 12.9, stage_a))
    for needle in ("less than 1%", "Only one seed", "gru has a lower validation NASA score",
                   "gru has a lower trainable parameter count", "gru has a lower recorded CPU latency",
                   "random_forest 11.590 (classical)", "no Session 6 choice used test results", "larger than"):
        assert needle in text, needle
    assert "gru 14.300" not in text
    worse = " ".join(interpretation_notes(comparison, {"gru": 20.0}, 12.9, []))
    assert "No historical model has a lower" in worse


def test_latency_axis_ticks_are_sparse_and_the_figure_is_deterministic(tmp_path):
    assert latency_ticks(2.4, 62.0) == [5, 10, 20, 50]
    points = [{"model": "gru", "family": "recurrent", "latency": 3.1, "rmse": 12.4, "n_parameters": 1},
              {"model": "tcn", "family": "convolutional", "latency": 47.6, "rmse": 14.6, "n_parameters": 2}]
    plot_family_rmse_vs_latency(tmp_path / "a.png", points)
    plot_family_rmse_vs_latency(tmp_path / "b.png", points[::-1])
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
