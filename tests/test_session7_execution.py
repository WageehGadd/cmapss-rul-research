"""Resumable execution on synthetic data: persistence, resume, interruption, corruption, MLflow
identity, isolation from the historical table and the test set, report and manifest."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

import src.run_session7 as run_session7
import src.session7_execution as execution
from src.benchmark_config import PROJECT_ROOT
from src.session7_integrity import execution_lock
from tests.session7_helpers import CONFIG, IDENTITY, synthetic

N = 14  # 2 models x (1 full + 2 budgets x (2 ranked + 1 random))
HISTORICAL = PROJECT_ROOT / "results/benchmark_results.json"


def write_config(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    s2 = yaml.safe_load((PROJECT_ROOT / "configs/session2_benchmark.yaml").read_text())
    s2["timing"].update(warmup_runs=1, timed_runs=2, batch_size=16)
    s2["tracking"] = {"tracking_dir": str(tmp_path / "mlruns"), "experiment_name": "pytest_history"}
    s2["environment"] = {"library_names": ["numpy"]}
    (tmp_path / "session2.yaml").write_text(yaml.safe_dump(s2))
    cfg["session2"] = {"config": str(tmp_path / "session2.yaml")}
    cfg["matrix"].update(models=["gradient_boosting", "lstm"], budgets=[4, 2], random_subset_seeds=[101])
    cfg["analysis"]["statistics"]["bootstrap_resamples"] = 200
    cfg["outputs"] = {"frozen_dir": str(tmp_path / "frozen"), "working_dir": str(tmp_path / "work"),
                      "report": str(tmp_path / "report.md"), "figures_dir": str(tmp_path / "figs")}
    path = tmp_path / "session7.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(run_session7, "load_train_validation", lambda cfg: (synthetic(), dict(IDENTITY)))
    monkeypatch.setattr(run_session7, "reproducibility_gate", lambda cfg, sources, records: {
        "checks": [{"model": r["model"], "passed": True, "rmse": r["metrics"]["rmse"],
                    "canonical_rmse": r["metrics"]["rmse"]} for r in records], "passed": True})
    for name in ("INCIDENT_PATH", "CONDITIONS_PATH"):                     # synthetic runs have no incident
        monkeypatch.setattr(run_session7, name, tmp_path / f"{name}.json")
    path = write_config(tmp_path)

    def run(*extra):
        return run_session7.main(["--config", str(path), *extra])

    return tmp_path, path, run


def records(tmp_path):
    return {p.stem: json.loads(p.read_text()) for p in (tmp_path / "work/conditions").glob("*.json")}


def lock_is_free(tmp_path):
    with execution_lock(tmp_path / "work"):
        return True


def tree(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "execution.lock"}


def mlflow_runs(tmp_path):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    e = mlflow.get_experiment_by_name("cmapss_rul_session7")
    return mlflow.search_runs([e.experiment_id], output_format="list")


def test_partial_run_resume_and_idempotence(env, capsys):
    tmp_path, _, run = env
    before = HISTORICAL.read_bytes() if HISTORICAL.exists() else None
    assert run("--stage", "run", "--limit", "3") == 3                       # incomplete
    first = records(tmp_path)
    assert len(first) == 3 and all(r["status"] == "completed" for r in first.values())
    assert "INCOMPLETE" in capsys.readouterr().out and lock_is_free(tmp_path)
    assert run("--stage", "resume") == 0
    second = records(tmp_path)
    assert len(second) == N and all(r["status"] == "completed" for r in second.values())
    for key, record in first.items():                                       # completed work is not redone
        assert second[key]["mlflow_run_id"] == record["mlflow_run_id"] and second[key]["attempt"] == 1
    assert len(mlflow_runs(tmp_path)) == N
    assert run("--stage", "all") == 0
    assert "Executed 0 conditions, reused 14" in capsys.readouterr().out and len(mlflow_runs(tmp_path)) == N
    if before is not None:
        assert HISTORICAL.read_bytes() == before                            # historical table untouched


def test_mlflow_identity_no_test_leakage_and_isolated_tables(env):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    recs = records(tmp_path)
    runs = {r.info.run_id: r for r in mlflow_runs(tmp_path)}
    for record in recs.values():
        mine = runs[record["mlflow_run_id"]]
        assert mine.data.tags["s7.experiment_id"] == record["experiment_id"] == mine.info.run_name
        assert mine.data.tags["session"] == "7" and mine.data.tags["s7.status"] == "completed"
        assert mine.data.params["s7.sensor_indices"] == ",".join(map(str, record["sensor_indices"]))
        assert mine.data.params["sensors"] == ",".join(record["sensors"])
        assert mine.data.params["s7.training_seed"] == "42"
        assert not any(k.startswith("test_") for k in mine.data.metrics)
        assert record["evaluation_sets"] == ["validation"]
    table = json.loads((tmp_path / "work/engine_results.json").read_text())["results"]
    assert len(table) == N and {r["eval_set"] for r in table} == {"validation"}


def test_interrupted_and_corrupted_and_tampered_records_are_rerun(env):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    recs = records(tmp_path)
    ids = sorted(recs)
    folder = tmp_path / "work/conditions"
    # 1) interrupted: left 'running' with a live-looking MLflow run
    import mlflow
    stale = recs[ids[0]]
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    exp = mlflow.get_experiment_by_name("cmapss_rul_session7").experiment_id
    dangling = client.create_run(exp).info.run_id
    (folder / f"{ids[0]}.json").write_text(json.dumps({**stale, "status": "running", "mlflow_run_id": dangling}))
    # 2) half-written record   3) tampered predictions
    (folder / f"{ids[1]}.json").write_text('{"status": "completed", ')
    np.savez(f"{recs[ids[2]]['run_dir']}/predictions.npz", validation=np.zeros(3))
    assert run("--stage", "run") == 0
    after = records(tmp_path)
    for key in ids[:3]:
        assert after[key]["status"] == "completed" and after[key]["mlflow_run_id"] != recs[key]["mlflow_run_id"]
    assert after[ids[0]]["attempt"] == 2 and after[ids[0]]["previous_attempts"][-1]["status"] == "interrupted"
    assert client.get_run(dangling).info.status == "KILLED"
    for key in ids[3:]:
        assert after[key]["mlflow_run_id"] == recs[key]["mlflow_run_id"]


def test_keyboard_interrupt_marks_the_condition_and_keeps_completed_work(env, monkeypatch):
    tmp_path, _, run = env
    real, calls = execution.run_experiment, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(execution, "run_experiment", flaky)
    with pytest.raises(KeyboardInterrupt):
        run("--stage", "run")
    status = sorted(r["status"] for r in records(tmp_path).values())
    assert status == ["completed", "completed", "interrupted"]
    assert lock_is_free(tmp_path)
    monkeypatch.setattr(execution, "run_experiment", real)
    assert run("--stage", "run") == 0


def test_failed_condition_is_recorded_and_the_rest_continue(env, monkeypatch):
    tmp_path, _, run = env
    real = execution.build_condition_model

    def broken(model, sources, data, **kwargs):
        if model == "lstm" and len(data.sensors) == 2:
            raise FloatingPointError("diverged")
        return real(model, sources, data, **kwargs)

    monkeypatch.setattr(execution, "build_condition_model", broken)
    assert run("--stage", "all") == 3
    recs = records(tmp_path)
    failed = [r for r in recs.values() if r["status"] == "failed"]
    assert len(failed) == 3 and len(recs) == N and "diverged" in failed[0]["error"]
    assert not any((tmp_path / name).exists() for name in ("report.md", "frozen", "figs"))   # nothing versioned
    partial = tmp_path / "work/incomplete_outputs"
    report = (partial / "report.md").read_text()
    assert "INCOMPLETE EXPERIMENT: 11 of 14" in report and "diverged" in report
    manifest = json.loads((partial / "artifacts/manifest.json").read_text())
    assert len(manifest["failed_conditions"]) == 3 and manifest["n_completed"] == 11


def test_changed_execution_settings_are_refused_and_the_plan_is_immutable(env, capsys):
    tmp_path, path, run = env
    assert run("--stage", "run", "--limit", "1") == 3
    cfg = yaml.safe_load(path.read_text())
    cfg["execution"]["classical_n_jobs_train"] = 1
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    before = tree(tmp_path / "work")
    assert run("--stage", "run", "--limit", "1") == 1
    assert "execution settings changed" in capsys.readouterr().err and tree(tmp_path / "work") == before
    cfg["execution"]["classical_n_jobs_train"] = -1
    cfg["matrix"]["budgets"] = [4]
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert run("--stage", "plan") == 1
    assert "differs from the predeclared one" in capsys.readouterr().err


def test_second_concurrent_execution_is_refused(env, capsys):
    tmp_path, _, run = env
    with execution_lock(tmp_path / "work"):                               # a live holder
        for stage in ("run", "plan", "summarize", "report"):              # every writing stage
            assert run("--stage", stage) == 1 and "another Session 7 execution" in capsys.readouterr().err
    assert not records(tmp_path)


def test_failing_gate_stops_before_the_reduced_matrix(env, monkeypatch):
    tmp_path, _, run = env
    monkeypatch.setattr(run_session7, "reproducibility_gate",
                        lambda *a: {"checks": [{"model": "x", "passed": False}], "passed": False})
    assert run("--stage", "all") == 2
    assert len(records(tmp_path)) == 2 and not (tmp_path / "report.md").exists()


def test_master_table_manifest_figures_and_report(env):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    master = json.loads((tmp_path / "work/master_experiments.json").read_text())["results"]
    assert len(master) == N and len({r["experiment_id"] for r in master}) == N
    for column in ("model", "family", "budget", "strategy", "subset_seed", "training_seed", "sensors",
                   "sensor_indices", "n_parameters", "val_rmse", "val_mae", "val_r2", "val_nasa_score",
                   "train_time_s", "inference_latency_ms_per_batch", "throughput_samples_per_s", "model_size_bytes",
                   "mlflow_run_id", "run_dir", "status", "error"):
        assert column in master[0], column
    frozen = tmp_path / "frozen"
    manifest = json.loads((frozen / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == digest
    for name in ("experiment_plan.json", "master_summary.csv", "degradation.csv", "random_controls.csv",
                 "ranked_vs_random.csv", "robustness.csv", "statistics.csv", "pareto.json", "conclusions.json",
                 "reproducibility_gate.json"):
        assert name in manifest["files"], name
    assert manifest["n_completed"] == manifest["n_planned"] == N and not manifest["missing_conditions"]
    assert len(manifest["conditions"]) == N and "upstream_files" in manifest and manifest["subset_seeds"] == [101]
    degradation = (frozen / "degradation.csv").read_text().splitlines()
    assert len(degradation) == N + 1
    report = (tmp_path / "report.md").read_text()
    for section in range(1, 21):
        assert f"\n## {section}. " in report, section
    assert "INCOMPLETE" not in report and "official test arrays are never opened" in report
    assert len(list((tmp_path / "figs").glob("fig*.png"))) == 5
    assert (tmp_path / "work/per_engine_metrics.csv").exists() and (tmp_path / "work/validation_index.npz").exists()


# ---- regression tests for the audit findings ---------------------------------------------------

def with_git(monkeypatch, **git):
    real = run_session7.git_state
    monkeypatch.setattr(run_session7, "git_state", lambda root: {**real(root), **git})


@pytest.mark.parametrize("git,reason", [({"git_dirty": False}, "clean/dirty"), ({"git_commit": "f" * 40}, "commit")])
def test_records_of_another_source_state_are_never_reused_or_overwritten(env, monkeypatch, capsys, git, reason):
    tmp_path, _, run = env
    with_git(monkeypatch, git_dirty=True)
    assert run("--stage", "all") == 0
    before, n_runs = tree(tmp_path / "work"), len(mlflow_runs(tmp_path))
    published = tree(tmp_path / "frozen")
    with_git(monkeypatch, **git)                                          # e.g. a later clean-commit execution
    for stage in ("run", "resume", "all", "plan", "summarize", "report"):
        assert run("--stage", stage) == 1, stage
        assert reason in capsys.readouterr().err
    assert tree(tmp_path / "work") == before and tree(tmp_path / "frozen") == published
    assert len(mlflow_runs(tmp_path)) == n_runs
    assert run("--stage", "verify") == 0                                  # still inspectable
    assert "never reused" in capsys.readouterr().out and tree(tmp_path / "work") == before


def edit(tmp_path, key, change):
    path = tmp_path / "work/conditions" / f"{key}.json"
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize("model,change,reason", [
    ("gradient_boosting", lambda r: r["metrics"].update(mae=0.1), "metrics do not match the predictions: mae"),
    ("gradient_boosting", lambda r: r["metrics"].update(nasa_score=1.0), "nasa_score"),
    ("gradient_boosting", lambda r: r["metrics"].update(r2=0.999), "r2"),
    ("gradient_boosting", lambda r: r.update(inference_latency_ms_per_batch=1e-3), "inference_latency_ms_per_batch"),
    ("gradient_boosting", lambda r: r.update(train_time_s=0.0), "train_time_s"),
    ("gradient_boosting", lambda r: r.update(n_parameters=1), "n_parameters"),
    ("gradient_boosting", lambda r: r.update(git_dirty=not r["git_dirty"]), "git_dirty"),
    ("gradient_boosting", lambda r: r["files"].pop("predictions.npz"), "does not reference"),
    ("lstm", lambda r: r.update(best_epoch=99), "best_epoch"),
    ("lstm", lambda r: r.update(weights_sha256="0" * 64), "weights"),
])
def test_edited_record_values_are_detected_before_aggregation(env, capsys, model, change, reason):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    key = next(k for k, r in records(tmp_path).items() if r["model"] == model and r["strategy"] == "permutation")
    edit(tmp_path, key, change)
    before = tree(tmp_path / "work")
    capsys.readouterr()
    assert run("--stage", "verify") == 3
    out = capsys.readouterr().out
    assert f"INVALID {key}" in out and reason in out and tree(tmp_path / "work") == before
    if reason == "git_dirty":                                             # edited provenance: refused outright
        assert run("--stage", "report") == 1
        return
    assert run("--stage", "report") == 3                                  # not counted, nothing versioned
    assert not (tmp_path / "frozen").exists() and not (tmp_path / "report.md").exists()
    master = json.loads((tmp_path / "work/master_experiments.json").read_text())["results"]
    row = next(r for r in master if r["experiment_id"] == key)
    assert row["status"] == "invalid" and row["val_rmse"] == ""


def test_deleted_mlflow_run_is_not_reused_and_the_old_record_is_archived(env, capsys):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    key, record = sorted(records(tmp_path).items())[0]
    mlflow_runs(tmp_path)
    import mlflow
    mlflow.MlflowClient().delete_run(record["mlflow_run_id"])             # soft delete: still readable
    assert run("--stage", "verify") == 3 and "MLflow run is deleted" in capsys.readouterr().out
    assert run("--stage", "run") == 0
    assert records(tmp_path)[key]["mlflow_run_id"] != record["mlflow_run_id"]
    archived = json.loads((tmp_path / f"work/conditions/archive/{key}.1.json").read_text())
    assert archived["mlflow_run_id"] == record["mlflow_run_id"]


def test_publish_writes_only_a_new_folder_and_states_the_source_difference(env, monkeypatch, capsys):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    work, frozen = tree(tmp_path / "work"), tree(tmp_path / "frozen")
    original_report = (tmp_path / "report.md").read_bytes()
    monkeypatch.setattr(run_session7, "code_fingerprint", lambda: "changed-source")
    target = tmp_path / "published"
    assert run("--stage", "publish", "--output-dir", str(target)) == 0
    assert tree(tmp_path / "work") == work and tree(tmp_path / "frozen") == frozen
    assert (tmp_path / "report.md").read_bytes() == original_report
    manifest = json.loads((target / "artifacts/manifest.json").read_text())
    provenance = manifest["execution_provenance"]
    assert provenance["source_matches_current"] is False and len(provenance["code_fingerprints"]) == 1
    assert "src/session7_outputs.py" in manifest["report_source_hashes"]
    assert manifest["official_test_evaluation"] == "not performed" and len(manifest["record_files_sha256"]) == N
    if manifest["git_dirty"]:
        assert manifest["canonical"] is False and "not canonical" in manifest["evidence_class"]
    report = (target / "report.md").read_text()
    for text in ("Selection-contaminated", "No official-test evaluation was performed",
                 "mean of the random draws", "Statistical conclusions (20 engines)", "Exploratory and post-hoc",
                 "Numerical versus practical", "Hardware-dependent latency", "code that trained them"):
        assert text in report, text
    for section in (8, 9, 10, 11, 15):                                    # caveat attached to the findings
        body = report.split(f"\n## {section}. ")[1].split("\n## ")[0]
        assert "Selection-contaminated" in body, section
    assert len(list((target / "figures").glob("fig*.png"))) == 5
    assert run("--stage", "publish", "--output-dir", str(target)) == 1
    assert "never overwrites" in capsys.readouterr().err
    assert run("--stage", "publish", "--output-dir", str(tmp_path / "work/inside")) == 1


def test_publish_refuses_incomplete_evidence(env, capsys):
    tmp_path, _, run = env
    assert run("--stage", "run", "--limit", "2") == 3
    assert run("--stage", "publish", "--output-dir", str(tmp_path / "published")) == 3
    assert not (tmp_path / "published").exists() and "nothing published" in capsys.readouterr().out


def test_outputs_of_another_execution_are_not_replaced(env, capsys):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    old = tree(tmp_path / "frozen")
    (tmp_path / "work").rename(tmp_path / "work_old")                     # a fresh execution, outputs left behind
    assert run("--stage", "all") == 1 and "another execution" in capsys.readouterr().err
    assert tree(tmp_path / "frozen") == old


def test_lock_is_released_by_the_kernel_when_its_process_dies(tmp_path):
    code = ("import sys, time; from src.session7_integrity import execution_lock\n"
            "with execution_lock(sys.argv[1]):\n    print('held', flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=PROJECT_ROOT,
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(RuntimeError, match=str(holder.pid)):
            with execution_lock(tmp_path):
                pass
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()
    assert (tmp_path / "execution.lock").exists()                         # left behind, and harmless
    with execution_lock(tmp_path):
        assert (tmp_path / "execution.lock").read_text() == str(os.getpid())


@pytest.mark.parametrize("same_execution", [True, False])
def test_operator_observations_are_printed_only_for_the_execution_they_describe(env, same_execution):
    tmp_path, _, run = env
    fingerprint = execution.code_fingerprint() if same_execution else "another-execution"
    (tmp_path / "CONDITIONS_PATH.json").write_text(json.dumps({
        "applies_to_code_fingerprint": fingerprint,
        "operator_observations": {"power": "OBSERVED-BATTERY", "load": "busy"},
        "latency_repeatability": {"max_relative_difference_pct": 12.0}}))
    assert run("--stage", "all") == 0
    report = (tmp_path / "report.md").read_text()
    assert ("OBSERVED-BATTERY" in report) is same_execution
    assert ("were not recorded for these runs" in report) is not same_execution
