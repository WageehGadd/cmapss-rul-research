"""Session 8 primary evidence: what it is, whether it is unchanged, how it is read without producing
anything, and what may truthfully be said about its origin.

Primary evidence is what execution produced: stored predictions, refits, the label-access ledger,
the saved test labels beside their predictions. Derived results (tables, figures, report) are
computed from it. Nothing in this module runs a model or reads the FD003 label file.

Trust boundary. `evidence_manifest.json` lists the evidence; the reviewed, versioned reference
(`configs/session8_trusted_reference.json`) states which manifest is the right one. Hashes show change
relative to that reference, nothing more: whoever can rewrite both the files and the reference (before
it is committed, or in history) is not detected. Both were made after execution.
"""
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .benchmark_config import PROJECT_ROOT, resolve_path
from .evaluation import predictions_digest
from .session7_execution import atomic_write_json
from .session7_plan import sha256_file, sha256_json
from .session8_data import DATASETS, TEST_SET, windows_digest
from .session8_guard import (EVIDENCE_NAME, REFERENCE_PATH, require_authorization,
                             require_new_location)
from .session8_evaluate import read_rows
from .session8_inference import identity_of, prediction_path, stored_prediction

EVIDENCE_PATTERNS = ("predictions/*/*.npz", "predictions/*/*.json", "sensitivity/*.json", "sensitivity/*.npz",
                     "sensitivity/models/*/*", "fd003_test_predictions.csv", "per_engine_all.csv",
                     "fd003_test_access.json", "evaluation_run.json", "dataset_facts.json")
# The only other things a working directory may hold. Neither is evidence.
NOT_EVIDENCE = ("execution.lock", ".DS_Store")           # writer lock; macOS folder metadata
NOT_EVIDENCE_FOLDERS = ("remediation",)                  # local copies of files from before the audit fixes
MANIFEST_KEYS = {"schema_version", "session", "config_sha256", "protocol_sha256", "evidence_manifest", "execution",
                 "derivation", "canonical", "evidence_class", "session7_manifest_sha256", "datasets", "fd003_sha256",
                 "fd003_test_label_accesses", "fd001_official_test_arrays_opened", "models_trained_for_main_results",
                 "sensitivity_fits", "checkpoints", "files", "figures", "report_sha256"}


def working_dir(cfg):
    return resolve_path(cfg["outputs"]["working_dir"])


def evidence_files(cfg):
    work = working_dir(cfg)
    return {str(p.relative_to(work)): sha256_file(p)
            for pattern in EVIDENCE_PATTERNS for p in sorted(work.glob(pattern)) if p.is_file()}


def mlflow_runs(cfg):
    """{run_id: digest of its parameters, metrics and status} of the Session 8 experiment (read only)."""
    db = sqlite3.connect(f"file:{resolve_path(cfg['upstream']['mlflow_dir']) / 'mlflow.db'}?mode=ro", uri=True)
    try:
        runs = db.execute("select r.run_uuid, r.name, r.status, r.lifecycle_stage from runs r join experiments e "
                          "using(experiment_id) where e.name = ?", (cfg["sensitivity"]["experiment_name"],)).fetchall()
        out = {}
        for run_id, name, status, stage in runs:
            rows = lambda t: sorted(db.execute(f"select key, value from {t} where run_uuid = ?", (run_id,)))
            out[run_id] = {"name": name, "digest": sha256_json([status, stage, rows("params"), rows("latest_metrics")]),
                           "git_commit": dict(rows("params")).get("git_commit"),
                           "git_dirty": dict(rows("params")).get("git_dirty")}
        return out
    finally:
        db.close()


def execution_record(cfg):
    """The git states recorded WHEN the evidence was produced. Absent stays absent."""
    work = working_dir(cfg)
    predictions = [json.loads(p.read_text()).get("produced_by") for p in sorted(work.glob("predictions/*/*.json"))]
    fits = [json.loads(p.read_text()).get("git") for p in sorted(work.glob("sensitivity/*.json"))]
    marker = work / "evaluation_run.json"
    evaluation = json.loads(marker.read_text()).get("git") if marker.exists() else None
    runs = mlflow_runs(cfg)
    logged = sorted({(r["git_commit"], r["git_dirty"]) for r in runs.values()}, key=str)
    return {"predictions": {"files": len(predictions), "with_recorded_producer": sum(p is not None for p in predictions),
                            "producers": [json.loads(s) for s in sorted({json.dumps(p, sort_keys=True)
                                                                        for p in predictions if p})]},
            "fits": {"count": len(fits), "git_states": [json.loads(s) for s in
                                                        sorted({json.dumps(g, sort_keys=True) for g in fits})]},
            "evaluation_marker_git": evaluation,
            "mlflow_logged_git": [{"git_commit": c, "git_dirty": d} for c, d in logged]}


def evidence_state(execution, derivation):
    """(canonical, evidence class). Canonical status is never granted here: it can only be left open
    (None) when every piece of evidence records one clean commit AND the same source files that derive
    the results. A later commit or a rebuilt report upgrades nothing."""
    producers = execution.get("predictions", {})
    states = [*producers.get("producers", []), *execution.get("fits", {}).get("git_states", []),
              execution.get("evaluation_marker_git")]
    complete = (producers.get("files", 0) > 0 and producers.get("with_recorded_producer") == producers.get("files")
                and states and all(isinstance(s, dict) for s in states))
    if complete:
        git = derivation.get("git", {})
        clean = all(s.get("git_dirty") is False and s.get("git_commit") == git.get("git_commit") for s in states)
        sources = derivation.get("source_hashes") or {}
        same = all(s.get("source_hashes", {}).items() <= sources.items() for s in producers["producers"])
        if clean and git.get("git_dirty") is False and same:
            return None, "executed and derived from one clean commit; canonical status needs a review"
    reason = ("the source state of some evidence was not recorded" if not complete
              else "some evidence comes from an uncommitted or different source state")
    return False, f"not canonical: {reason}"


def build_evidence(cfg):
    ledger = json.loads((working_dir(cfg) / "fd003_test_access.json").read_text())
    return {"files": evidence_files(cfg), "mlflow_runs": mlflow_runs(cfg), "execution": execution_record(cfg),
            "fd003_test_label_accesses": ledger,
            "session7_manifest_sha256": sha256_file(resolve_path(cfg["upstream"]["session7_manifest"]))}


def trusted_reference(cfg):
    """The reviewed entry for exactly this execution's locations, or None."""
    outputs = cfg["outputs"]
    path = resolve_path(cfg.get("_reference", REFERENCE_PATH))
    if not path.exists():
        return None
    for entry in json.loads(path.read_text())["executions"]:
        same = resolve_path(entry["artifacts_dir"]) == resolve_path(outputs["artifacts_dir"])
        if same and resolve_path(entry["working_dir"]) == resolve_path(outputs["working_dir"]) \
                and entry["experiment_name"] == cfg["sensitivity"]["experiment_name"]:
            return entry
    return None


def register_evidence(cfg, note, *, authorization=None):
    """INITIAL registration of a new execution's evidence, once. It is part of execution, not of
    verification: it needs authorization, and it refuses any location that a reviewed reference
    covers or that already holds a seal or an artifacts manifest. Removing a seal does not reopen it."""
    require_authorization(authorization, "registering evidence")
    require_new_location(cfg)
    current = build_evidence(cfg)
    if not current["files"]:
        raise RuntimeError("there is no evidence to register")
    path = resolve_path(cfg["outputs"]["artifacts_dir"]) / EVIDENCE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"schema_version": 1, "sealed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "note": note, **current})
    return path


def unexpected_files(cfg, sealed_files, supplementary):
    """Files in the working directory that are neither evidence nor on the short list of exceptions."""
    work = working_dir(cfg)
    known = set(sealed_files) | set(supplementary) | set(evidence_files(cfg))
    out = []
    for p in sorted(work.rglob("*")):
        relative = p.relative_to(work)
        if (p.is_file() or p.is_symlink()) and relative.parts[0] not in NOT_EVIDENCE_FOLDERS \
                and p.name not in NOT_EVIDENCE and str(relative) not in known:
            out.append(str(relative))
    return out


def check_evidence(cfg):
    """Differences between what is present and what the reviewed reference vouches for (empty when
    identical). A missing reference, a missing seal or a seal other than the reviewed one is a
    difference. Nothing is written."""
    entry = trusted_reference(cfg)
    if entry is None:
        return ["there is no trusted reference for this execution"]
    path = resolve_path(cfg["outputs"]["artifacts_dir"]) / EVIDENCE_NAME
    if not path.exists():
        return ["the evidence manifest is missing"]
    problems = []
    if sha256_file(path) != entry["evidence_manifest_sha256"]:
        problems.append("the evidence manifest differs from the trusted reference")
    sealed = json.loads(path.read_text())
    work = working_dir(cfg)
    supplementary = entry.get("supplementary_files", {})
    problems += [f"supplementary file missing or changed: {name}" for name, digest in sorted(supplementary.items())
                 if not (work / name).is_file() or sha256_file(work / name) != digest]
    problems += [f"unexpected file in the working directory: {name}"
                 for name in unexpected_files(cfg, sealed["files"], supplementary)]
    try:
        current = build_evidence(cfg)
    except (OSError, ValueError) as error:
        return [*problems, f"the evidence cannot be read: {error}"]
    for key in ("files", "mlflow_runs"):
        old, new = sealed[key], current[key]
        problems += [f"{key}: missing {k}" for k in sorted(set(old) - set(new))]
        problems += [f"{key}: not in the sealed manifest {k}" for k in sorted(set(new) - set(old))]
        problems += [f"{key}: changed {k}" for k in sorted(set(old) & set(new)) if old[k] != new[k]]
    problems += [f"{k} differs from the sealed manifest" for k in
                 ("execution", "fd003_test_label_accesses", "session7_manifest_sha256") if sealed[k] != current[k]]
    return problems


def check_manifest(cfg, records, expected_figures):
    """The published artifacts manifest against the reviewed reference, the sealed evidence and the
    present files. Execution provenance comes from the seal and the reference only: the current Git
    HEAD plays no part, so a later commit changes nothing."""
    entry = trusted_reference(cfg)
    if entry is None:
        return ["there is no trusted reference for this execution"]
    artifacts = resolve_path(cfg["outputs"]["artifacts_dir"])
    path, seal_path = artifacts / "manifest.json", artifacts / EVIDENCE_NAME
    if not path.exists() or not seal_path.exists():
        return [f"{p.name} is missing" for p in (path, seal_path) if not p.exists()]
    problems = []
    if sha256_file(path) != entry.get("artifacts_manifest_sha256"):
        problems.append("manifest.json differs from the trusted reference")
    manifest, sealed = json.loads(path.read_text()), json.loads(seal_path.read_text())
    if set(manifest) != MANIFEST_KEYS:
        return [*problems, f"manifest.json fields differ: {sorted(set(manifest) ^ MANIFEST_KEYS)}"]
    canonical, evidence_class = evidence_state(sealed["execution"], manifest["derivation"])
    if manifest["canonical"] is not entry["canonical"] or manifest["canonical"] is not canonical:
        problems.append(f"manifest.json claims canonical={manifest['canonical']!r}; the reviewed classification "
                        f"is {entry['canonical']!r}")
    if not manifest["evidence_class"] == entry["evidence_class"] == evidence_class:
        problems.append("manifest.json evidence_class differs from the reviewed execution classification")
    expected = {
        "schema_version": 2, "session": 8,
        "evidence_manifest": {"file": EVIDENCE_NAME, "sha256": entry["evidence_manifest_sha256"]},
        "execution": sealed["execution"], "fd003_test_label_accesses": sealed["fd003_test_label_accesses"],
        "session7_manifest_sha256": sealed["session7_manifest_sha256"],
        "sensitivity_fits": sealed["execution"]["fits"]["count"], "fd003_sha256": cfg["fd003"]["sha256"],
        "fd001_official_test_arrays_opened": False, "models_trained_for_main_results": 0,
        "checkpoints": {k: {"mlflow_run_id": r["mlflow_run_id"], "files": r["files"]} for k, r in records.items()},
    }
    facts = working_dir(cfg) / "dataset_facts.json"
    expected["datasets"] = json.loads(facts.read_text()) if facts.exists() else None
    problems += [f"manifest.json field {k} is not what the evidence shows" for k, v in expected.items()
                 if manifest[k] != v]
    present = {p.name for p in artifacts.iterdir() if p.name not in ("manifest.json", ".DS_Store")}
    problems += [f"unlisted file in the artifacts folder: {n}" for n in sorted(present - set(manifest["files"]))]
    problems += [f"artifact changed or missing: {n}" for n, d in sorted(manifest["files"].items())
                 if not (artifacts / n).is_file() or sha256_file(artifacts / n) != d]
    figures = resolve_path(cfg["outputs"]["figures_dir"])
    listed = {resolve_path(n) for n in manifest["figures"]}
    if listed != {figures / n for n in expected_figures}:
        problems.append("manifest.json does not list exactly the Session 8 figures")
    problems += [f"unlisted Session 8 figure: {p.name}" for p in sorted(figures.glob("*session8*"))
                 if p not in listed]
    problems += [f"figure changed or missing: {n}" for n, d in sorted(manifest["figures"].items())
                 if not resolve_path(n).is_file() or sha256_file(resolve_path(n)) != d]
    report = resolve_path(cfg["outputs"]["report"])
    if not report.is_file() or sha256_file(report) != manifest["report_sha256"]:
        problems.append("report changed or missing")
    return problems


def check_sources(cfg, current):
    """Each source file is the one the manifest says derived the results, or exactly the later change
    recorded in the reviewed reference. `current` is {path: sha256} of the source files now."""
    entry = trusted_reference(cfg) or {}
    path = resolve_path(cfg["outputs"]["artifacts_dir"]) / "manifest.json"
    if not path.exists():
        return ["manifest.json is missing"]
    derived = json.loads(path.read_text()).get("derivation", {}).get("source_hashes", {})
    reviewed = entry.get("sources_changed_after_derivation", {})
    problems = [f"source listed by the manifest is not a Session 8 source: {n}" for n in sorted(set(derived) - set(current))]
    for name, digest in sorted(current.items()):
        change = {"derived_with": derived.get(name), "current": digest}
        if derived.get(name) != digest and reviewed.get(name) != change:
            problems.append(f"source differs from the derivation and from the reviewed changes: {name}")
    return problems


def require_evidence(cfg):
    problems = check_evidence(cfg)
    if problems:
        raise RuntimeError(f"the saved evidence is not the sealed evidence ({len(problems)} differences, e.g. "
                           f"{problems[0]}); nothing is derived from it")


def load_saved_predictions(cfg, canonical, inputs):
    """Stored predictions only. A missing or foreign file is an error: no model is ever loaded."""
    digests = {k: windows_digest(w) for k, w in inputs.items()}
    out = {dataset: {} for dataset in DATASETS}
    for key, condition in sorted(canonical.conditions.items()):
        for dataset in DATASETS:
            identity = identity_of(condition, canonical.records[key], digests[dataset])
            prediction = stored_prediction(prediction_path(cfg, dataset, key), identity)
            if prediction is None:
                raise RuntimeError(f"no saved prediction of {key} on {dataset}; it is not recomputed here")
            if dataset == "fd001_validation" and predictions_digest(prediction) != \
                    canonical.records[key]["predictions_sha256"]:
                raise RuntimeError(f"{key}: saved FD001 predictions differ from the canonical record")
            out[dataset][key] = prediction
    return out


def saved_test_labels(cfg, predictions):
    """The raw FD003 test RUL as saved beside the predictions by the evaluation (the label file is not
    opened). Every condition's rows must agree with each other and with its stored predictions."""
    rows = read_rows(working_dir(cfg) / "fd003_test_predictions.csv")
    by_condition = {}
    for r in rows:
        by_condition.setdefault(r["experiment_id"], {})[r["unit_id"]] = r
    if set(by_condition) != set(predictions[TEST_SET]):
        raise ValueError("the saved test records do not cover exactly the stored predictions")
    labels = None
    for key, units in by_condition.items():
        order = sorted(units)
        mine = np.array([units[u]["true_rul_raw"] for u in order], dtype=float)
        labels = mine if labels is None else labels
        if order != list(range(1, len(labels) + 1)) or not np.array_equal(mine, labels) \
                or not np.array_equal([units[u]["prediction"] for u in order], predictions[TEST_SET][key]):
            raise ValueError(f"{key}: saved test records are inconsistent with the stored predictions")
    return labels
