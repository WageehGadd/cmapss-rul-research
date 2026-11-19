"""End-to-end Session 3 run on synthetic data: Session 2 engine reuse and test-set isolation."""
import csv
import json

import numpy as np
import pytest
import yaml

import src.run_session3 as run_session3
from src.session3_config import load_session3_config
from tests.conftest import synthetic_data


def write_config(tmp_path):
    cfg = {
        "session1": {"config": "configs/session1_fd001.yaml", "processed_dir": "data/processed/fd001"},
        "session2": {"config": str(tmp_path / "session2.yaml")},
        "benchmark": {"target_type": "capped", "seed": 3, "evaluation_sets": ["validation", "test"],
                      "sensor_set_name": "synthetic_14"},
        "features": {"temporal": ["last", "mean", "std", "min", "max", "slope"]},
        "execution": {"n_jobs_train": 1, "n_jobs_predict": 1},
        "models": {
            "random_forest": {"name": "random_forest",
                              "search_space": {"n_estimators": [3, 5], "max_depth": [3, 5]}},
            "gradient_boosting": {"name": "gradient_boosting",
                                  "search_space": {"max_iter": [5, 10], "max_depth": [2, 3]},
                                  "fixed": {"min_samples_leaf": 5}}},
        "selection": {"metric": "rmse", "tie_tolerance_rel": 0.01},
        "ranking": {"correlation": {"feature": "last"},
                    "permutation": {"source_model": "random_forest", "n_repeats": 2, "seed": 1,
                                    "aggregation": "sum"},
                    "top_disagreements": 3},
        "budgets": {"sensor_counts": [14, 10, 7, 4, 2], "random_control_seeds": [11, 12]},
        "outputs": {"frozen_dir": str(tmp_path / "frozen"), "working_dir": str(tmp_path / "work"),
                    "report": str(tmp_path / "report.md"), "figures_dir": str(tmp_path / "figs")},
    }
    s2 = {
        "session1": cfg["session1"], "benchmark": cfg["benchmark"],
        "baselines": [{"name": "mean_rul", "type": "mean_rul"}],
        "timing": {"warmup_runs": 1, "timed_runs": 2, "batch_size": 16, "latency_unit": "ms",
                   "timing_set": "validation", "hardware_description": "auto"},
        "tracking": {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_s3"},
        "outputs": {"results_csv": str(tmp_path / "results.csv"), "results_json": str(tmp_path / "results.json"),
                    "runs_dir": str(tmp_path / "runs")},
        "environment": {"library_names": ["numpy"]},
    }
    (tmp_path / "session2.yaml").write_text(yaml.safe_dump(s2))
    path = tmp_path / "session3.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run(tmp_path, monkeypatch, data):
    path = write_config(tmp_path)
    monkeypatch.setattr(run_session3, "load_benchmark_data", lambda *a, **k: data)
    assert run_session3.main(["--config", str(path)]) == 0
    return tmp_path


def frozen_content(root):
    """Frozen files with the location-dependent config reference removed (temp dirs differ)."""
    out = {}
    for p in sorted((root / "frozen").iterdir()):
        if p.name == "manifest.json":
            continue
        text = p.read_text()
        if p.suffix == ".json":
            doc = json.loads(text)
            if isinstance(doc, dict):
                doc.pop("config_ref", None)
                doc.pop("config_sha256", None)
            text = json.dumps(doc, sort_keys=True)
        out[p.name] = text
    return out


def test_end_to_end_outputs_and_session2_table(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, synthetic_data(seed=0))
    frozen = tmp_path / "frozen"
    expected = {"sensor_ranking_correlation.csv", "sensor_ranking_permutation.csv",
                "sensor_ranking_comparison.json", "sensor_subsets_topk.json",
                "sensor_subsets_randomk.json", "manifest.json"}
    assert expected <= {p.name for p in frozen.iterdir()}

    with open(frozen / "sensor_ranking_correlation.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["rank"]) for r in rows] == list(range(1, 15))
    with open(frozen / "sensor_ranking_permutation.csv", newline="") as f:
        perm = list(csv.DictReader(f))
    assert {"importance_last", "importance_slope"} <= set(perm[0])

    topk = json.loads((frozen / "sensor_subsets_topk.json").read_text())["subsets"]
    randomk = json.loads((frozen / "sensor_subsets_randomk.json").read_text())["subsets"]
    for method in ("correlation", "permutation"):
        assert [len(topk[method][f"top_{k}"]["sensors"]) for k in (14, 10, 7, 4, 2)] == [14, 10, 7, 4, 2]
    assert all(len(v["draws"]) == 2 for v in randomk.values())

    # Same results table as Session 2, extended: required columns plus optional model fields.
    table = json.loads((tmp_path / "results.json").read_text())
    models = [r["model"] for r in table["results"]]
    assert models == ["random_forest"] * 2 + ["gradient_boosting"] * 2
    assert {r["eval_set"] for r in table["results"]} == {"validation", "test"}
    assert {"model_family", "n_parameters", "selected_params"} <= set(table["columns"])
    assert all(r["config_ref"].endswith("session3.yaml") or r["config_ref"] == str(tmp_path / "session3.yaml")
               for r in table["results"])
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert set(manifest["files"]) == (expected - {"manifest.json"}) | {"selected_models.json",
                                                                      "permutation_feature_importance.json"}
    assert "Spearman" in (tmp_path / "report.md").read_text()
    assert "Top-k subsets" in capsys.readouterr().out


def test_rerun_reproduces_features_selection_rankings_and_metrics(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run(a, monkeypatch, synthetic_data(seed=0))
    run(b, monkeypatch, synthetic_data(seed=0))
    assert frozen_content(a) == frozen_content(b)
    ta = json.loads((a / "results.json").read_text())["results"]
    tb = json.loads((b / "results.json").read_text())["results"]
    assert [r["predictions_sha256"] for r in ta] == [r["predictions_sha256"] for r in tb]
    assert [r["rmse"] for r in ta] == [r["rmse"] for r in tb]
    assert [r["run_id"] for r in ta] != [r["run_id"] for r in tb]


def test_test_split_changes_cannot_alter_selection_or_rankings(tmp_path, monkeypatch):
    base, changed = tmp_path / "base", tmp_path / "changed"
    base.mkdir(), changed.mkdir()
    run(base, monkeypatch, synthetic_data(seed=0))
    data = synthetic_data(seed=0)
    data.X["test"][:] = np.random.default_rng(5).normal(size=data.X["test"].shape).astype("float32")
    data.y["test"][:] = 999.0
    run(changed, monkeypatch, data)
    assert frozen_content(base) == frozen_content(changed)                 # rankings and subsets
    sel = lambda root: json.loads((root / "work" / "selection.json").read_text())
    for family in ("random_forest", "gradient_boosting"):
        a, b = sel(base)[family], sel(changed)[family]
        assert a["selected_params"] == b["selected_params"] and a["rule"] == b["rule"]
    rows = lambda root: json.loads((root / "results.json").read_text())["results"]
    val = lambda root: [r["rmse"] for r in rows(root) if r["eval_set"] == "validation"]
    test = lambda root: [r["rmse"] for r in rows(root) if r["eval_set"] == "test"]
    assert val(base) == val(changed) and test(base) != test(changed)    # test only affects its own score


def test_missing_config_section_is_reported(tmp_path):
    path = write_config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    del cfg["budgets"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="budgets"):
        load_session3_config(path)


def test_frozen_package_has_selected_models_session1_identity_and_consistent_manifest(tmp_path, monkeypatch):
    import hashlib
    from src.benchmark_config import PROJECT_ROOT
    run(tmp_path, monkeypatch, synthetic_data(seed=0))
    frozen = tmp_path / "frozen"

    selected = json.loads((frozen / "selected_models.json").read_text())["models"]
    selection = json.loads((tmp_path / "work" / "selection.json").read_text())
    assert set(selected) == {"random_forest", "gradient_boosting"}
    for family, doc in selected.items():
        assert doc["params"] == selection[family]["selected_params"]          # same as the real selection
        assert doc["selected_candidate"] == selection[family]["selected_candidate"]
        assert doc["seed"] == 3 and doc["selection"]["metric"] == "rmse"
        assert doc["selection"]["tie_tolerance_rel"] == 0.01 and "n_parameters" in doc["selection"]["tiebreak"]
    assert selected["random_forest"]["estimator"].endswith("RandomForestRegressor")
    assert selected["gradient_boosting"]["estimator"].endswith("HistGradientBoostingRegressor")
    assert selected["gradient_boosting"]["params"]["min_samples_leaf"] == 5      # fixed param is frozen too

    manifest = json.loads((frozen / "manifest.json").read_text())
    on_disk = {p.name for p in frozen.iterdir() if p.name != "manifest.json"}
    assert set(manifest["files"]) == on_disk
    assert {"selected_models.json", "permutation_feature_importance.json"} <= on_disk
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest

    s1 = manifest["session1"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    real_config = PROJECT_ROOT / "configs" / "session1_fd001.yaml"
    assert s1["config_sha256"] == sha(real_config)
    assert s1["sequence_length"] == 30 and s1["rul_cap"] == 125 and s1["target_type"] == "capped"
    assert len(s1["sensors"]) == 14 and "train_units_sha256" in s1 and "split_sha256" in s1


def test_committed_session1_identity_matches_the_actual_files():
    """Skipped when the processed Session 1 data has not been generated."""
    import hashlib
    from src.benchmark_config import PROJECT_ROOT
    processed = PROJECT_ROOT / "data/processed/fd001"
    manifest_path = PROJECT_ROOT / "artifacts/session3/manifest.json"
    if not (processed / "manifest.json").exists() or not manifest_path.exists():
        pytest.skip("Session 1 artifacts or frozen Session 3 artifacts not present")
    s1 = json.loads(manifest_path.read_text())["session1"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert s1["manifest_sha256"] == sha(processed / "manifest.json")
    assert s1["split_sha256"] == sha(processed / "split.json")
    assert s1["config_sha256"] == sha(PROJECT_ROOT / "configs/session1_fd001.yaml")
    frozen = manifest_path.parent
    for name, digest in json.loads(manifest_path.read_text())["files"].items():
        assert sha(frozen / name) == digest
    assert (frozen / "selected_models.json").exists()
