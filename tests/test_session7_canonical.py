"""The canonical-run loader: only the frozen matrix and its published evidence qualify.

One synthetic execution is made per module; every test tampers with its own copy of it."""
import hashlib
import json
import shutil
from dataclasses import replace

import pytest

import src.run_session7 as run_session7
from src.benchmark_config import PROJECT_ROOT
from src.session7_canonical import CANONICAL, Frozen, canonical_runs, matrix_digest
from src.session7_integrity import RunStore
from tests.session7_helpers import IDENTITY, synthetic
from tests.test_session7_execution import N, write_config

COMMIT = "a" * 40


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    root = tmp_path_factory.mktemp("canonical")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_session7, "load_train_validation", lambda cfg: (synthetic(), dict(IDENTITY)))
        patch.setattr(run_session7, "git_state", lambda r: {"git_commit": COMMIT, "git_dirty": False})
        patch.setattr(run_session7, "reproducibility_gate", lambda cfg, sources, records: {
            "checks": [{"model": r["model"], "passed": True, "rmse": 1.0, "canonical_rmse": 1.0} for r in records],
            "passed": True})
        for name in ("INCIDENT_PATH", "CONDITIONS_PATH"):
            patch.setattr(run_session7, name, root / f"{name}.json")
        assert run_session7.main(["--config", str(write_config(root))]) == 0
    return root


class Evidence:
    """A private copy of the execution, with the expectation it satisfies when untouched."""

    def __init__(self, source, root):
        shutil.copytree(source, root)
        self.manifest = root / "frozen/manifest.json"
        self.plan = root / "frozen/experiment_plan.json"
        self.work, self.store = root / "work", root / "mlruns"
        self.expected = Frozen(n_conditions=N, matrix_sha256=matrix_digest(load(self.plan)["conditions"]),
                               git_commit=COMMIT, manifest_sha256=digest(self.manifest))
        self.ids = sorted(load(self.manifest)["conditions"])

    def edit(self, change):
        """Change the manifest and pin the result, so that only the other checks can reject it."""
        manifest = load(self.manifest)
        change(manifest)
        self.manifest.write_text(json.dumps(manifest))
        self.expected = replace(self.expected, manifest_sha256=digest(self.manifest))

    def load(self):
        return canonical_runs(self.manifest, self.store, self.work, expected=self.expected)


@pytest.fixture
def evidence(execution, tmp_path):
    return Evidence(execution, tmp_path / "copy")


def test_the_untouched_evidence_loads_without_being_modified(evidence):
    root = evidence.work.parent
    before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
    runs = evidence.load()
    assert len(runs) == N and sorted(runs) == evidence.ids
    assert runs == {k: v["mlflow_run_id"] for k, v in load(evidence.manifest)["conditions"].items()}
    assert {p: digest(p) for p in root.rglob("*") if p.is_file()} == before


def drop(manifest, ids):
    """A smaller manifest whose counts still agree with itself."""
    del manifest["conditions"][ids[0]], manifest["record_files_sha256"][ids[0]]
    manifest.update(n_planned=N - 1, n_completed=N - 1)


def rename(manifest, ids):
    for key in ("conditions", "record_files_sha256"):
        manifest[key]["s7_substituted"] = manifest[key].pop(ids[0])


def swap(manifest, ids):
    c = manifest["conditions"]
    c[ids[0]], c[ids[1]] = c[ids[1]], c[ids[0]]


def entry(**values):
    return lambda m, ids: m["conditions"][ids[0]].update(values)


@pytest.mark.parametrize("change,message", [
    (drop, "completed conditions"),
    (lambda m, ids: m["conditions"].pop(ids[0]), "1 missing"),
    (lambda m, ids: m["record_files_sha256"].pop(ids[0]), "record hashes"),
    (lambda m, ids: m["conditions"].update(extra=m["conditions"][ids[0]]), "1 unknown"),
    (rename, "1 missing"),
    (lambda m, ids: m["conditions"].update({ids[1]: m["conditions"][ids[0]]}), "more than one condition"),
    (swap, "not a canonical run"),
    (entry(mlflow_run_id="f" * 32), "not a canonical run"),
    (entry(predictions_sha256="0" * 64), "unexpected predictions_sha256"),
    (entry(status="failed"), "not completed"),
    (lambda m, ids: m["record_files_sha256"].update({ids[0]: "0" * 64}), "manifest hash"),
    (lambda m, ids: m.update(git_commit="b" * 40), "clean-commit"),
    (lambda m, ids: m.update(git_dirty=True), "clean-commit"),
    (lambda m, ids: m.update(mlflow_experiment="another"), "not a canonical run"),
    (lambda m, ids: m["files"].update({"experiment_plan.json": "0" * 64}), "plan differs"),
    (lambda m, ids: m.pop("conditions"), "incomplete or malformed"),
    (lambda m, ids: m.update(conditions=None), "incomplete or malformed"),
])
def test_altered_manifests_are_rejected(evidence, change, message):
    evidence.edit(lambda m: change(m, evidence.ids))
    with pytest.raises(ValueError, match=message):
        evidence.load()


def test_a_condition_listed_twice_in_the_file_is_rejected(evidence):
    text = evidence.manifest.read_text()
    key = json.dumps(evidence.ids[0])
    first = json.dumps(load(evidence.manifest)["conditions"][evidence.ids[0]])
    evidence.manifest.write_text(text.replace('"conditions": {', f'"conditions": {{ {key}: {first},', 1))
    assert len(load(evidence.manifest)["conditions"]) == N                # plain json would not notice
    evidence.expected = replace(evidence.expected, manifest_sha256=digest(evidence.manifest))
    with pytest.raises(ValueError, match="duplicated entries"):
        evidence.load()


@pytest.mark.parametrize("change", [
    lambda c: c.pop(),                                                    # smaller matrix
    lambda c: c.append(dict(c[0])),                                       # duplicated condition
    lambda c: c[0].update(identity_sha256="0" * 64),                      # substituted identity
    lambda c: c[0].update(experiment_id="s7_substituted")])
def test_a_plan_other_than_the_frozen_matrix_is_rejected_even_if_the_manifest_agrees(evidence, change):
    plan = load(evidence.plan)
    change(plan["conditions"])
    evidence.plan.write_text(json.dumps(plan))
    evidence.edit(lambda m: m["files"].update({"experiment_plan.json": digest(evidence.plan)}))
    with pytest.raises(ValueError, match="frozen"):
        evidence.load()


def test_any_other_change_to_the_published_manifest_is_rejected(evidence):
    evidence.manifest.write_text(evidence.manifest.read_text() + " ")
    with pytest.raises(ValueError, match="not the published manifest"):
        evidence.load()


@pytest.mark.parametrize("damage,message", [
    (lambda record, run: record.unlink(), "incomplete or malformed"),
    (lambda record, run: record.write_text(record.read_text() + " "), "manifest hash"),
    (lambda record, run: (run / "predictions.npz").write_bytes(b"x"), "changed: predictions.npz"),
    (lambda record, run: (run / next(f for f in load(record)["files"] if f.startswith("model/"))).write_bytes(b"x"),
     "changed: model/"),
    (lambda record, run: shutil.rmtree(run), "incomplete or malformed")])
def test_missing_or_changed_stored_evidence_is_rejected(evidence, damage, message):
    key = evidence.ids[0]
    run_id = load(evidence.manifest)["conditions"][key]["mlflow_run_id"]
    damage(evidence.work / f"conditions/{key}.json", evidence.work / f"runs/{run_id}")
    with pytest.raises(ValueError, match=message):
        evidence.load()


def test_a_record_with_another_identity_is_rejected_even_if_its_hash_is_listed(evidence):
    key = evidence.ids[0]
    path = evidence.work / f"conditions/{key}.json"
    path.write_text(json.dumps({**load(path), "identity_sha256": "0" * 64}))
    evidence.edit(lambda m: m["record_files_sha256"].update({key: digest(path)}))
    with pytest.raises(ValueError, match="unexpected identity_sha256"):
        evidence.load()


@pytest.mark.parametrize("field,key,value", [
    ("status", None, "FAILED"), ("lifecycle_stage", None, "deleted"), ("experiment", None, "another"),
    ("tags", "s7.experiment_id", "another"), ("tags", "s7.verification", "True"), ("tags", "s7.status", "superseded"),
    ("params", "s7.identity_sha256", "0" * 64), ("params", "git_commit", "b" * 40), ("params", "git_dirty", "True")])
def test_runs_that_do_not_qualify_are_rejected(evidence, monkeypatch, field, key, value):
    real = RunStore.get

    def altered(self, run_id):
        run = real(self, run_id)
        return {**run, field: value if key is None else {**run[field], key: value}}

    monkeypatch.setattr(RunStore, "get", altered)
    with pytest.raises(ValueError, match="not a canonical run"):
        evidence.load()


# ---- the frozen constants and the real evidence -------------------------------------------------

PUBLISHED = PROJECT_ROOT / "artifacts/session7"


def test_the_pinned_values_are_those_of_the_published_artifacts():
    conditions = load(PUBLISHED / "experiment_plan.json")["conditions"]
    assert CANONICAL.n_conditions == len(conditions) == 174
    assert CANONICAL.matrix_sha256 == matrix_digest(conditions)
    assert CANONICAL.manifest_sha256 == digest(PUBLISHED / "manifest.json")
    assert CANONICAL.git_commit == load(PUBLISHED / "manifest.json")["git_commit"]


@pytest.mark.skipif(not (PROJECT_ROOT / "results/session7/conditions").is_dir()
                    or not (PROJECT_ROOT / "mlruns/mlflow.db").exists(), reason="local Session 7 evidence absent")
def test_the_real_canonical_evidence_loads():
    runs = canonical_runs("artifacts/session7/manifest.json", "mlruns")
    assert len(runs) == len(set(runs.values())) == 174
