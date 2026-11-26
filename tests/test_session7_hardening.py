"""Regression tests for the post-audit hardening. Synthetic data, temporary folders and stores only."""
import ast
import json
import math
import subprocess

import pytest

import src.run_session7 as run_session7
from src.benchmark_config import PROJECT_ROOT
from src.session7_execution import FINGERPRINT_MODULES, SOURCE_MODULES
from src.session7_integrity import UNKNOWN, git_state, schema_problem
from tests.test_session7_execution import N, edit, env, records, tree, with_git  # noqa: F401  (env is a fixture)

# ---- required record fields --------------------------------------------------------------------

BASE = {"metrics": {"rmse": 1.0, "mae": 1.0, "r2": -0.5, "nasa_score": 2.0}, "train_time_s": 0.1,
        "inference_latency_ms_per_batch": 2.0, "inference_latency_ms_per_sample": 0.01,
        "throughput_samples_per_s": 100.0, "model_size_bytes": 10, "n_parameters": 5, "code_fingerprint": "c",
        "started_at": "s", "ended_at": "e", "git_commit": "g", "git_dirty": False, "mlflow_run_id": "m",
        "run_dir": "d"}
DEEP_FIELDS = {"weights_sha256": "a" * 64, "best_epoch": 3, "epochs_run": 5, "stop_reason": "patience"}


def test_valid_classical_and_deep_records_pass():
    assert schema_problem({"model": "random_forest"}, BASE) is None
    assert schema_problem({"model": "gru"}, {**BASE, **DEEP_FIELDS}) is None


@pytest.mark.parametrize("field", [k for k in BASE if k != "metrics"] + list(DEEP_FIELDS))
@pytest.mark.parametrize("bad", ["<missing>", None, ""])
def test_missing_null_or_empty_fields_are_rejected(field, bad):
    record = {**BASE, **DEEP_FIELDS, field: bad}
    if bad == "<missing>":
        del record[field]
    assert field in schema_problem({"model": "lstm"}, record)


@pytest.mark.parametrize("field,bad", [
    ("inference_latency_ms_per_batch", "2.0"), ("inference_latency_ms_per_batch", 0.0),
    ("inference_latency_ms_per_batch", -1.0), ("inference_latency_ms_per_batch", math.nan),
    ("inference_latency_ms_per_batch", math.inf), ("inference_latency_ms_per_batch", True),
    ("model_size_bytes", 1.5), ("n_parameters", 0), ("weights_sha256", "A" * 64), ("weights_sha256", "a" * 63),
    ("best_epoch", 1.5), ("git_dirty", UNKNOWN), ("git_dirty", 0)])
def test_malformed_values_are_rejected(field, bad):
    assert field in schema_problem({"model": "lstm"}, {**BASE, **DEEP_FIELDS, field: bad})


@pytest.mark.parametrize("bad", [None, {}, {"rmse": 1.0}, {**BASE["metrics"], "mae": math.nan}])
def test_incomplete_metrics_are_rejected(bad):
    assert "metric" in schema_problem({"model": "random_forest"}, {**BASE, "metrics": bad})


def test_a_classical_record_does_not_define_a_weight_digest():
    assert "classical" in schema_problem({"model": "random_forest"}, {**BASE, "weights_sha256": "a" * 64})


@pytest.mark.parametrize("model,change,field", [
    ("gradient_boosting", lambda r: r.pop("inference_latency_ms_per_batch"), "inference_latency_ms_per_batch"),
    ("gradient_boosting", lambda r: r.update(inference_latency_ms_per_batch=None), "inference_latency_ms_per_batch"),
    ("lstm", lambda r: r.pop("weights_sha256"), "weights_sha256"),
    ("lstm", lambda r: r.update(weights_sha256=None), "weights_sha256")])
def test_removed_fields_invalidate_the_record_before_any_pareto_step(env, monkeypatch, capsys, model, change, field):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    key = next(k for k, r in records(tmp_path).items() if r["model"] == model and r["strategy"] == "permutation")
    edit(tmp_path, key, change)
    seen = []
    real = run_session7.pareto_front
    monkeypatch.setattr(run_session7, "pareto_front", lambda rows, objectives: seen.append(
        [r["experiment_id"] for r in rows if r["status"] == "completed"]) or real(rows, objectives))
    capsys.readouterr()
    assert run("--stage", "report") == 3
    out = capsys.readouterr().out
    assert f"INVALID {key}: {field}" in out and seen and all(key not in ids for ids in seen)


# ---- git state ---------------------------------------------------------------------------------

def test_git_state_reports_clean_dirty_and_unavailable(tmp_path, monkeypatch):
    assert git_state(tmp_path) == {"git_commit": UNKNOWN, "git_dirty": UNKNOWN}          # not a repository
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path,
                                    check=True, capture_output=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    assert git_state(tmp_path)["git_dirty"] is False
    (tmp_path / "new").write_text("x")
    assert git_state(tmp_path)["git_dirty"] is True

    real = subprocess.run

    def status_fails(command, *args, **kwargs):
        if command[1] == "status":
            raise subprocess.CalledProcessError(128, command)
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", status_fails)
    state = git_state(tmp_path)
    assert state["git_dirty"] == UNKNOWN and state["git_commit"] != UNKNOWN


def test_nothing_is_written_when_the_source_state_is_unknown(env, monkeypatch, capsys):
    tmp_path, _, run = env
    with_git(monkeypatch, git_dirty=UNKNOWN)
    for stage in ("run", "plan", "summarize", "report", "all"):
        assert run("--stage", stage) == 1 and "cannot be determined" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


# ---- source fingerprint ------------------------------------------------------------------------

def imported(module):
    found = set()
    for node in ast.walk(ast.parse((PROJECT_ROOT / f"src/{module}.py").read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            found.update([node.module] if node.module else [a.name for a in node.names])
    return found


def closure(start):
    seen, todo = set(), [start]
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo += imported(module)
    return seen


def test_every_imported_source_module_is_hashed():
    assert closure("run_session7") <= set(SOURCE_MODULES)
    # record checking is hashed with the report sources but does not change how anything is trained
    assert closure("session7_execution") - {"session7_integrity"} <= set(FINGERPRINT_MODULES)


# ---- publication gate --------------------------------------------------------------------------

@pytest.mark.parametrize("damage", ["missing", "failed", "partial"])
def test_report_and_publish_refuse_without_a_passed_gate(env, capsys, damage):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    path = tmp_path / "work/reproducibility_gate.json"
    gate = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(json.dumps({**gate, "passed": False} if damage == "failed"
                                   else {**gate, "checks": gate["checks"][:1]}))
    capsys.readouterr()
    assert run("--stage", "report") == 2 and "NOT PUBLISHED" in capsys.readouterr().out
    assert run("--stage", "publish", "--output-dir", str(tmp_path / "published")) == 2
    assert not any((tmp_path / name).exists() for name in ("report.md", "frozen", "figs", "published"))
    assert not (tmp_path / "work/incomplete_outputs").exists()


def test_a_stored_pass_does_not_override_failing_records(env, monkeypatch):
    tmp_path, _, run = env
    assert run("--stage", "run") == 0
    monkeypatch.setattr(run_session7, "reproducibility_gate", lambda *a: {"checks": [], "passed": False})
    assert run("--stage", "report") == 2 and not (tmp_path / "report.md").exists()


# ---- experiment identity of published outputs --------------------------------------------------

def published(tmp_path):
    return {**tree(tmp_path / "frozen"), **tree(tmp_path / "figs"), "report": (tmp_path / "report.md").read_bytes()}


@pytest.mark.parametrize("change", [
    lambda m: m.update(conditions={f"other_{i}": v for i, v in enumerate(m["conditions"].values())}),   # no overlap
    lambda m: m.update(data={"manifest_sha256": "another"}),
    lambda m: m.update(config_sha256="0" * 64)])
def test_outputs_of_a_different_experiment_block_every_writing_stage(env, capsys, change):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    path = tmp_path / "frozen/manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    before, work = published(tmp_path), tree(tmp_path / "work")
    capsys.readouterr()
    for stage in ("plan", "run", "resume", "summarize", "report", "all"):
        assert run("--stage", stage) == 1, stage
        assert "different experiment" in capsys.readouterr().err
    assert published(tmp_path) == before and tree(tmp_path / "work") == work


@pytest.mark.parametrize("damage", ["unreadable", "deleted"])
def test_outputs_without_a_usable_manifest_are_not_replaced(env, capsys, damage):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    path = tmp_path / "frozen/manifest.json"
    path.write_text("{") if damage == "unreadable" else path.unlink()
    before = published(tmp_path)
    assert run("--stage", "all") == 1 and "manifest" in capsys.readouterr().err
    assert published(tmp_path) == before


def test_a_fresh_work_folder_is_refused_before_training(env, capsys):
    tmp_path, _, run = env
    assert run("--stage", "all") == 0
    (tmp_path / "work").rename(tmp_path / "work_old")
    assert run("--stage", "all") == 1 and "another execution" in capsys.readouterr().err
    assert not (tmp_path / "work").exists()


# ---- canonical designation ---------------------------------------------------------

@pytest.mark.parametrize("dirty,expected", [(True, ("verification", False)), (False, ("clean_commit", None))])
def test_manifest_states_the_execution_class(env, monkeypatch, dirty, expected):
    tmp_path, _, run = env
    with_git(monkeypatch, git_dirty=dirty)
    assert run("--stage", "all") == 0
    manifest = json.loads((tmp_path / "frozen/manifest.json").read_text())
    assert (manifest["execution_class"], manifest["canonical"]) == expected and manifest["canonical_note"]


@pytest.mark.parametrize("dirty,text", [(True, "it was not committed"), (False, "it is the source of commit")])
def test_a_republished_report_says_whether_the_training_source_was_committed(env, monkeypatch, dirty, text):
    tmp_path, _, run = env
    with_git(monkeypatch, git_dirty=dirty)
    assert run("--stage", "all") == 0
    monkeypatch.setattr(run_session7, "code_fingerprint", lambda: "changed-source")
    assert run("--stage", "publish", "--output-dir", str(tmp_path / "published")) == 0
    assert text in (tmp_path / "published/report.md").read_text()
