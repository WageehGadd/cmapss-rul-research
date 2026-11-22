"""Sequence-length handling, selection logic and an end-to-end Session 4 run on synthetic windows."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

import src.run_session4 as run_session4
from src.benchmark_config import PROJECT_ROOT, resolve_path
from src.recurrent_study import common_endpoint_metrics, select_sequence_length, select_stage_a, stage_a_grid
from src.sequence_data import SequenceData, load_sequence_data, regenerate_windows, window_counts
from src.session4_config import load_session4_config
from src.windows import make_test_windows, make_training_windows

PROCESSED = "data/processed/fd001"
S1_CONFIG = "configs/session1_fd001.yaml"
needs_session1 = pytest.mark.skipif(not (resolve_path(PROCESSED) / "manifest.json").exists(),
                                    reason="Session 1 artifacts not generated")
SENSORS = [f"sensor_{i}" for i in range(1, 15)]


def synthetic_sequence_data(L, seed=0):
    """Windows built with the Session 1 windowing functions from a small synthetic fleet."""
    rng = np.random.default_rng(seed)
    frames = {}
    for split, units in (("train", range(1, 9)), ("validation", range(9, 13))):
        parts = []
        for u in units:
            n = int(rng.integers(55, 75))
            df = pd.DataFrame(rng.normal(size=(n, 14)), columns=SENSORS)
            df["unit_id"], df["cycle"] = u, np.arange(1, n + 1)
            df["rul_raw"] = n - df["cycle"]
            df["rul"] = df["rul_raw"].clip(upper=125)
            df[SENSORS[0]] += (1.0 - df["rul"] / 125.0) * 3              # a learnable signal
            parts.append(df)
        frames[split] = pd.concat(parts, ignore_index=True)
    test = pd.concat([pd.DataFrame({**{s: rng.normal(size=60) for s in SENSORS}, "unit_id": u,
                                    "cycle": np.arange(1, 61)}) for u in range(13, 18)], ignore_index=True)
    sets = {k: make_training_windows(v, SENSORS, L) for k, v in frames.items()}
    sets["test"] = make_test_windows(test, SENSORS, L, np.array([30, 80, 140, 10, 55]), 125)
    return SequenceData(X={k: w.X for k, w in sets.items()}, y={k: w.y for k, w in sets.items()},
                        unit_ids={k: w.unit_ids for k, w in sets.items()},
                        end_cycles={k: w.end_cycles for k, w in sets.items()}, sensors=list(SENSORS),
                        target_type="capped", rul_cap=125, sequence_length=L)


@needs_session1
def test_regenerating_the_canonical_length_reproduces_the_frozen_arrays():
    frozen = load_sequence_data(PROCESSED, S1_CONFIG, "capped", 30)
    again = regenerate_windows(PROCESSED, S1_CONFIG, "capped", 30)
    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(frozen.X[split], again.X[split])
        np.testing.assert_array_equal(frozen.y[split], again.y[split])
        np.testing.assert_array_equal(frozen.end_cycles[split], again.end_cycles[split])
    assert frozen.X["train"].shape == (14459, 30, 14) and frozen.X["test"].shape == (100, 30, 14)


@needs_session1
def test_other_lengths_keep_split_and_scaler_and_only_change_window_counts():
    c, d = load_sequence_data(PROCESSED, S1_CONFIG, "capped", 30), load_sequence_data(PROCESSED, S1_CONFIG, "capped", 20)
    assert d.X["train"].shape[1:] == (20, 14) and window_counts(d)["test"] == 100
    assert window_counts(d)["train"] > window_counts(c)["train"]
    for split in ("train", "validation", "test"):
        assert set(d.unit_ids[split]) == set(c.unit_ids[split])               # same engines per split
    assert not set(d.unit_ids["train"]) & set(d.unit_ids["validation"])
    # the last 20 cycles of an L=30 window are exactly the L=20 window with the same endpoint
    key = lambda x: {(u, e): i for i, (u, e) in enumerate(zip(x.unit_ids["validation"], x.end_cycles["validation"]))}
    kc, kd = key(c), key(d)
    for k in list(kc)[:50]:
        np.testing.assert_array_equal(c.X["validation"][kc[k]][-20:], d.X["validation"][kd[k]])
    with pytest.raises(Exception, match="fewer than sequence_length"):
        load_sequence_data(PROCESSED, S1_CONFIG, "capped", 50)                # test engines with 31 cycles


def _rec(L, family="lstm", rmse=10.0, params=100, status="ok"):
    return {"candidate": 0, "validation": {"rmse": rmse}, "n_parameters": params, "status": status,
            "family": family, "sequence_length": L, "hidden_size": 8, "num_layers": 1, "dropout": 0.0}


def test_stage_a_selection_uses_validation_rmse_then_fewest_parameters_and_skips_unstable():
    cfg = {"selection": {"metric": "rmse", "tie_tolerance_rel": 0.01}}
    recs = {"lstm": [{**_rec(30, rmse=10.0, params=900), "candidate": 0},
                     {**_rec(30, rmse=10.05, params=100), "candidate": 1},
                     {**_rec(30, rmse=11.0, params=10), "candidate": 2},
                     {**_rec(30, rmse=1.0, params=1, status="unstable"), "candidate": 3}]}
    chosen, rule = select_stage_a(cfg, recs)["lstm"]
    assert chosen["candidate"] == 1 and rule["n_equivalent"] == 2


def test_stage_a_grid_size():
    cfg = {"ablation": {"stage_a": {"hidden_size": [32, 64, 128], "num_layers": [1, 2], "dropout": [0.0, 0.2]}}}
    grid = stage_a_grid(cfg)
    assert len(grid) == 12 and len({tuple(g.values()) for g in grid}) == 12


def test_sequence_lengths_are_compared_on_common_endpoints():
    d20, d30 = synthetic_sequence_data(20), synthetic_sequence_data(30)
    assert len(d20.y["validation"]) > len(d30.y["validation"])
    datasets = {20: d20, 30: d30}
    perfect30 = {**_rec(30), "predictions": d30.y["validation"].astype(float)}
    biased20 = {**_rec(20), "predictions": d20.y["validation"].astype(float) + 5.0}
    cfg = {"selection": {"metric": "rmse", "tie_tolerance_rel": 0.01}}
    chosen, rule, rows = select_sequence_length(cfg, "lstm", [biased20, perfect30], datasets, 30)
    assert chosen["sequence_length"] == 30 and rows[0]["n_common_windows"] == rows[1]["n_common_windows"]
    assert rows[0]["rmse"] == pytest.approx(5.0) and rows[1]["rmse"] == pytest.approx(0.0, abs=1e-6)
    metrics, mask = common_endpoint_metrics(biased20, d20, 30)
    assert mask.sum() == len(d30.y["validation"])


# ---------------------------------------------------------------------------------------------
def write_config(tmp_path):
    base = {
        "session1": {"config": S1_CONFIG, "processed_dir": PROCESSED},
        "session2": {"config": str(tmp_path / "session2.yaml")},
        "benchmark": {"target_type": "capped", "seed": 5, "evaluation_sets": ["validation", "test"],
                      "sensor_set_name": "synthetic_14"},
        "training": {"optimizer": "adam", "learning_rate": 0.01, "weight_decay": 0.0, "batch_size": 64,
                     "max_epochs": 4, "early_stopping_patience": 2, "min_delta": 1e-6, "grad_clip": 1.0,
                     "scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 1, "min_lr": 1e-5},
                     "target_scale": 125},
        "execution": {"device": "cpu"},
        "reference": {"family": "lstm", "hidden_size": 8, "num_layers": 1, "dropout": 0.0, "sequence_length": 30,
                      "tiny_batch_samples": 32, "tiny_batch_epochs": 100, "tiny_batch_learning_rate": 0.02,
                      "min_loss_drop": 0.3},
        "ablation": {"families": ["lstm", "gru"],
                     "stage_a": {"sequence_length": 30, "hidden_size": [4, 8], "num_layers": [1], "dropout": [0.0]},
                     "stage_b": {"sequence_lengths": [20]}},
        "selection": {"metric": "rmse", "tie_tolerance_rel": 0.01, "stage_b_comparison": "common_endpoints"},
        "outputs": {"frozen_dir": str(tmp_path / "frozen"), "checkpoint_dir": str(tmp_path / "ckpt"),
                    "working_dir": str(tmp_path / "work"), "report": str(tmp_path / "report.md"),
                    "figures_dir": str(tmp_path / "figs")},
    }
    s2 = {"session1": base["session1"], "benchmark": base["benchmark"],
          "baselines": [{"name": "mean_rul", "type": "mean_rul"}],
          "timing": {"warmup_runs": 1, "timed_runs": 2, "batch_size": 16, "latency_unit": "ms",
                     "timing_set": "validation", "hardware_description": "auto"},
          "tracking": {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_s4"},
          "outputs": {"results_csv": str(tmp_path / "results.csv"), "results_json": str(tmp_path / "results.json"),
                      "runs_dir": str(tmp_path / "runs")},
          "environment": {"library_names": ["numpy"]}}
    (tmp_path / "session2.yaml").write_text(yaml.safe_dump(s2))
    path = tmp_path / "session4.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return path


def run(tmp_path, monkeypatch, test_shift=False):
    path = write_config(tmp_path)

    def loader(processed_dir, session1_config, target_type, L):
        d = synthetic_sequence_data(L)
        if test_shift:
            d.X["test"][:] = 7.0
            d.y["test"][:] = 111.0
        return d

    monkeypatch.setattr(run_session4, "load_sequence_data", loader)
    assert run_session4.main(["--config", str(path)]) == 0
    return tmp_path


@needs_session1
def test_end_to_end_run_freezes_selected_models_and_extends_the_table(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "Reference lstm" in out and "Stage A" in out
    frozen = tmp_path / "frozen"
    for f in ("selected_lstm.json", "selected_gru.json", "manifest.json"):
        assert (frozen / f).exists()
    lstm = json.loads((frozen / "selected_lstm.json").read_text())
    assert lstm["selection"]["data"] == "validation only" and "test" not in json.dumps(lstm["validation_metrics"])
    assert lstm["retrain_matches_ablation_weights"] is True              # CPU training is repeatable
    manifest = json.loads((frozen / "manifest.json").read_text())
    import hashlib
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest
    for family, ck in manifest["checkpoints"].items():
        assert hashlib.sha256((tmp_path / "ckpt" / f"selected_{family}.pt").read_bytes()).hexdigest() == ck["file_sha256"]
    assert manifest["session1"]["sequence_length"] == 30

    table = json.loads((tmp_path / "results.json").read_text())
    assert [r["model"] for r in table["results"]] == ["lstm", "lstm", "gru", "gru"]
    assert {"model_family", "hidden_size", "num_layers", "dropout", "sequence_length", "best_epoch", "epochs_run",
            "n_parameters"} <= set(table["columns"])

    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    exp = mlflow.get_experiment_by_name("pytest_s4")
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    stages = sorted(r.data.tags.get("stage", "table") for r in runs)
    assert len(runs) == 8                                                 # 4 stage A + 2 stage B + 2 selected
    assert any(s.startswith("ablation") for s in stages) and any(s.startswith("sequence_length") for s in stages)
    ablation = next(r for r in runs if r.data.tags.get("stage", "").startswith("ablation"))
    for key in ("validation_rmse", "best_epoch", "epochs_run", "n_parameters", "train_time_s"):
        assert key in ablation.data.metrics
    assert "test_rmse" not in ablation.data.metrics                       # ablation runs never see the test set

    # checkpoints reload and reproduce the table's predictions
    from src.recurrent_models import RecurrentRegressor
    run_id = table["results"][0]["run_id"]
    model = RecurrentRegressor.load(tmp_path / "runs" / run_id / "model")
    d = synthetic_sequence_data(model.sequence_length)
    assert np.isfinite(model.predict(d.X["validation"])).all()


@needs_session1
def test_test_split_cannot_influence_selection_or_validation_results(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run(a, monkeypatch)
    run(b, monkeypatch, test_shift=True)
    for f in ("selected_lstm.json", "selected_gru.json"):
        x, y = (json.loads((r / "frozen" / f).read_text()) for r in (a, b))
        assert x["architecture"] == y["architecture"] and x["checkpoint"]["weights_sha256"] == y["checkpoint"]["weights_sha256"]
        assert x["selection"]["stage_b_common_endpoint_rows"] == y["selection"]["stage_b_common_endpoint_rows"]
    ra = json.loads((a / "results.json").read_text())["results"]
    rb = json.loads((b / "results.json").read_text())["results"]
    val = lambda rows: [r["predictions_sha256"] for r in rows if r["eval_set"] == "validation"]
    tst = lambda rows: [r["rmse"] for r in rows if r["eval_set"] == "test"]
    assert val(ra) == val(rb) and tst(ra) != tst(rb)


def test_config_validation(tmp_path):
    path = write_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["training"]["optimizer"] = "sgd"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="adam"):
        load_session4_config(path)
