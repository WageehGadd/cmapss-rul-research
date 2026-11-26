"""Session 8 entry point: reliability, FD001 -> FD003 generalisation and error analysis.

    python -m src.run_session8 --config configs/session8_evaluation.yaml --stage STAGE

Safe stages (no inference, no training, no label file, no MLflow writes):
  validate     canonical Session 7 evidence and inputs (writes nothing)
  verify       evidence and published files against the trusted reference, and that the tables
               re-derive (writes nothing)
  derive       rebuild the tables from the saved predictions and saved label records
  report       rebuild figures, report and artifacts manifest from saved results

Execution stages, each requiring --authorize-execution (a deliberate, separate operation) AND output
locations that hold no completed execution (a new configuration; the reviewed one is refused):
  predict      apply the 174 frozen models (inference)
  evaluate     read the FD003 test labels (logged, protocol-locked) and build every table
  sensitivity  21 small refits (trains)
  register     initial registration of the new execution's evidence (once)
  all          predict, evaluate, sensitivity, register
No stage that reads saved results creates, replaces or updates the evidence manifest.
Exit codes: 0 ok, 1 error, 3 something is missing or differs.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from .benchmark_config import PROJECT_ROOT, resolve_path
from .evaluation import predictions_digest
from .session7_execution import atomic_write_json
from .session7_integrity import execution_lock, git_state
from .session7_plan import load_session7_config, load_train_validation, sha256_file, sha256_json
from .session8_data import (dataset_facts, input_shift, load_config, load_inputs, load_test_labels, protocol_identity,
                            sensor_list)
from .session8_evaluate import evaluate_all, read_rows, write_rows
from .session8_evidence import (check_evidence, check_manifest, check_sources, evidence_state, load_saved_predictions,
                                register_evidence, require_evidence, saved_test_labels, trusted_reference)
from .session8_guard import (EVIDENCE_NAME, ExecutionAuthorization, require_authorization,
                             require_new_location)
from .session8_inference import load_canonical, predict_all
from .session8_sensitivity import raw_targets, run_fits, sensitivity_rows

SAFE_STAGES = ("validate", "verify", "derive", "report")
EXECUTION_STAGES = ("predict", "evaluate", "sensitivity", "register", "all")
STAGES = (*SAFE_STAGES, *EXECUTION_STAGES)
LOCAL_ONLY = ("per_engine_all", "fd003_test_predictions")        # detailed tables kept in the working directory
SOURCES = ("session8_guard", "session8_evidence", "session8_data", "session8_inference", "session8_analysis", "session8_evaluate", "session8_sensitivity",
           "session8_reporting", "session8_report_text", "run_session8", "metrics", "windows", "scaling", "rul", "data_loading",
           "session7_analysis", "session7_canonical", "session7_models", "session7_plan")


SEAL_NOTE = ("Sealed after execution. It shows the evidence is unchanged since this time; it cannot show what the "
             "source files were when the evidence was produced, which was not recorded for predictions.")


def say(text):
    print(text, flush=True)


def source_hashes():
    return {f"src/{m}.py": sha256_file(PROJECT_ROOT / f"src/{m}.py") for m in SOURCES}


def paths(cfg):
    return resolve_path(cfg["outputs"]["working_dir"]), resolve_path(cfg["outputs"]["artifacts_dir"])


class Tracking:
    """Session 8's own MLflow experiment; Sessions 1-7 runs are never opened for writing."""

    def __init__(self, cfg, *, authorization=None):
        require_authorization(authorization, "writing MLflow runs")
        require_new_location(cfg)
        self.dir = resolve_path(cfg["upstream"]["mlflow_dir"])
        self.name = cfg["sensitivity"]["experiment_name"]
        if self.name in ("cmapss_rul_benchmark", "cmapss_rul_session7"):
            raise ValueError("Session 8 needs its own MLflow experiment")

    def log(self, run_name, tags, params, metrics):
        from .tracking import MlflowTracker, mlflow

        with MlflowTracker(self.dir, self.name).start_run(run_name, tags={"session": "8", **tags}) as run_id:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        return run_id


def stage_predict(cfg, *, authorization=None):
    require_authorization(authorization, "stage predict")
    require_new_location(cfg)
    canonical = load_canonical(cfg)
    inputs = load_inputs(cfg)
    work, _ = paths(cfg)
    say(f"Canonical evidence verified: {len(canonical.conditions)} conditions. Inference only.")
    predictions = predict_all(cfg, canonical, inputs, say,
                              {**git_state(PROJECT_ROOT), "source_hashes": source_hashes()},
                              authorization=authorization)
    say("All models reproduced their canonical FD001 validation predictions.")
    atomic_write_json(work / "dataset_facts.json", dataset_facts(cfg, inputs))
    return canonical, inputs, predictions


def stage_evaluate(cfg, *, authorization=None):
    require_authorization(authorization, "stage evaluate")
    require_new_location(cfg)
    canonical, inputs, predictions = stage_predict(cfg, authorization=authorization)
    work, artifacts = paths(cfg)
    labels = load_test_labels(cfg, "evaluate", authorization=authorization)
    tables = evaluate_all(cfg, canonical, inputs, predictions, labels)
    for name, rows in tables.items():
        write_rows((work if name in LOCAL_ONLY else artifacts) / f"{name}.csv", rows)
    write_rows(artifacts / "input_shift.csv", input_shift(cfg))
    tracking = Tracking(cfg, authorization=authorization)
    reference = {(r["candidate"], r["dataset"]): r for r in tables["candidate_summary"]}
    marker = work / "evaluation_run.json"
    git = git_state(PROJECT_ROOT)
    identity = sha256_json({**protocol_identity(cfg), "master": tables["master_results"], "git": git})
    if not marker.exists() or json.loads(marker.read_text())["identity"] != identity:
        run_id = tracking.log("s8_evaluation", {"s8.kind": "evaluation"},
                              {**protocol_identity(cfg), "git_commit": git["git_commit"],
                               "git_dirty": git["git_dirty"], "n_conditions": len(canonical.conditions),
                               "training": "none"},
                              {f"{k}.{d}.rmse": v["rmse"] for (k, d), v in reference.items()})
        atomic_write_json(marker, {"identity": identity, "mlflow_run_id": run_id, "git": git})
    say(f"Evaluation tables written to {artifacts}")


def stage_sensitivity(cfg, *, authorization=None):
    require_authorization(authorization, "stage sensitivity")
    require_new_location(cfg)
    canonical = load_canonical(cfg)
    _, artifacts = paths(cfg)
    tracking = Tracking(cfg, authorization=authorization)

    def track(key, cap, record):
        return tracking.log(f"s8_cap{cap}_{key}", {"s8.kind": "cap_sensitivity", "s8.candidate": key},
                            {"training_cap": cap, "model": record["model"], "git_commit": record["git"]["git_commit"],
                             "git_dirty": record["git"]["git_dirty"], "fit_identity": record["identity"]},
                            {**{f"validation_own_cap_{k}": v for k, v in record["own_cap_metrics"].items()},
                             "train_time_s": record["train_time_s"]})

    records, data, raw = run_fits(cfg, canonical, say, track, authorization=authorization)
    write_rows(artifacts / "sensitivity_cap.csv", sensitivity_rows(cfg, canonical, records, data, raw))
    say(f"Cap sensitivity: {len(records)} fits recorded.")


def derived_tables(cfg):
    """Every table, recomputed in memory from saved evidence: no model is loaded, the FD003 label file
    is not opened and nothing is written."""
    require_evidence(cfg)
    canonical = load_canonical(cfg)
    inputs = load_inputs(cfg)
    predictions = load_saved_predictions(cfg, canonical, inputs)
    tables = evaluate_all(cfg, canonical, inputs, predictions, saved_test_labels(cfg, predictions))
    tables["input_shift"] = input_shift(cfg)
    records, data, raw = saved_fits(cfg)
    tables["sensitivity_cap"] = sensitivity_rows(cfg, canonical, records, data, raw)
    return tables


def saved_fits(cfg):
    """The stored refit records with what is needed to score them; nothing is fitted."""
    s7, _, _ = load_session7_config(resolve_path(cfg["upstream"]["session7_config"]))
    data, _ = load_train_validation(s7)
    folder = resolve_path(cfg["outputs"]["working_dir"]) / "sensitivity"
    records = []
    for cap in (cfg["sensitivity"]["control_cap"], *cfg["sensitivity"]["caps"]):
        for key in cfg["sensitivity"]["configurations"]:
            record = json.loads((folder / f"{key}_cap{cap}.json").read_text())
            with np.load(folder / f"{key}_cap{cap}.npz") as z:
                if predictions_digest(z["prediction"]) != record["predictions_sha256"]:
                    raise RuntimeError(f"{key} cap {cap}: saved predictions do not match their record")
            records.append(record)
    return records, data, raw_targets(cfg["upstream"]["processed_dir"])


def stage_derive(cfg):
    _, artifacts = paths(cfg)
    tables = derived_tables(cfg)
    for name, rows in tables.items():
        if name not in LOCAL_ONLY:                      # the local tables are evidence and stay as saved
            write_rows(artifacts / f"{name}.csv", rows)
    say(f"Tables derived from saved evidence into {artifacts}")


def stage_report(cfg):
    from .session8_reporting import build_report

    require_evidence(cfg)
    canonical = load_canonical(cfg)
    inputs = load_inputs(cfg)
    work, artifacts = paths(cfg)
    predictions = load_saved_predictions(cfg, canonical, inputs)
    figures = build_report(cfg, canonical, inputs, predictions)
    files = {p.name: sha256_file(p) for p in sorted(artifacts.iterdir()) if p.name != "manifest.json"}
    evidence = json.loads((artifacts / EVIDENCE_NAME).read_text())
    derivation = {"git": git_state(PROJECT_ROOT), "source_hashes": source_hashes()}
    canonical_flag, evidence_class = evidence_state(evidence["execution"], derivation)
    reviewed = trusted_reference(cfg)                   # require_evidence has already demanded it
    if (canonical_flag, evidence_class) != (reviewed["canonical"], reviewed["evidence_class"]):
        raise RuntimeError("the execution classification differs from the reviewed one; no manifest is written")
    atomic_write_json(artifacts / "manifest.json", {
        "schema_version": 2, "session": 8, **protocol_identity(cfg),
        "evidence_manifest": {"file": EVIDENCE_NAME, "sha256": sha256_file(artifacts / EVIDENCE_NAME)},
        "execution": evidence["execution"],               # copied from the seal, never recomputed here
        "derivation": derivation,                         # the sources that built these files, now
        "canonical": canonical_flag, "evidence_class": evidence_class,
        "session7_manifest_sha256": sha256_file(resolve_path(cfg["upstream"]["session7_manifest"])),
        "datasets": json.loads((work / "dataset_facts.json").read_text()), "fd003_sha256": cfg["fd003"]["sha256"],
        "fd003_test_label_accesses": evidence["fd003_test_label_accesses"],
        "fd001_official_test_arrays_opened": False, "models_trained_for_main_results": 0,
        "sensitivity_fits": evidence["execution"]["fits"]["count"],
        "checkpoints": {k: {"mlflow_run_id": r["mlflow_run_id"], "files": r["files"]}
                        for k, r in sorted(canonical.records.items())},
        "files": files,
        "figures": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in figures},
        "report_sha256": sha256_file(resolve_path(cfg["outputs"]["report"]))})
    say(f"Report: {resolve_path(cfg['outputs']['report'])}")


def _same_table(path, rows):
    """Whether writing `rows` would reproduce the published file byte for byte."""
    with tempfile.TemporaryDirectory() as folder:
        write_rows(Path(folder) / path.name, rows)
        return path.exists() and (Path(folder) / path.name).read_bytes() == path.read_bytes()


def stage_verify(cfg):
    """Read-only: canonical evidence, Session 8 evidence and published files against the trusted
    reference, re-derivation and the pre-session snapshot."""
    from .session8_reporting import FIGURES, sensor_exposure

    canonical = load_canonical(cfg)
    work, artifacts = paths(cfg)
    evidence_problems = check_evidence(cfg)
    problems = [*evidence_problems, *check_manifest(cfg, canonical.records, FIGURES.values())]
    problems += check_sources(cfg, source_hashes())
    path = artifacts / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    if any(manifest.get(k) != v for k, v in protocol_identity(cfg).items()):
        problems.append("config or protocol differs from the one that produced the artifacts")
    master = read_rows(artifacts / "master_results.csv")
    if len(master) != 174:
        problems.append(f"master table has {len(master)} rows")
    for row in master:
        if abs(row["fd001_validation_rmse"] - canonical.records[row["experiment_id"]]["metrics"]["rmse"]) > 1e-9:
            problems.append(f"{row['experiment_id']}: FD001 RMSE differs from the canonical record")
    if not evidence_problems:
        tables = derived_tables(cfg)
        tables["exploratory_sensor_exposure"] = sensor_exposure(read_rows(artifacts / "master_results.csv"),
                                                                sensor_list(cfg)[0])
        for name, rows in tables.items():
            target = (work if name in LOCAL_ONLY else artifacts) / f"{name}.csv"
            if not _same_table(target, rows):
                problems.append(f"{name}.csv does not re-derive from the saved evidence")
    snapshot_path = work / "frozen_snapshot.json"
    if not snapshot_path.exists():
        problems.append("the pre-session snapshot is missing; frozen files were not checked")
    else:
        snapshot = json.loads(snapshot_path.read_text())
        reviewed = (trusted_reference(cfg) or {}).get("reviewed_snapshot_changes", {})
        changed, accepted = [], []
        for p, d in snapshot["files"].items():
            now = sha256_file(PROJECT_ROOT / p) if (PROJECT_ROOT / p).exists() else None
            if now != d:
                (accepted if reviewed.get(p) == {"snapshot": d, "current": now} else changed).append(p)
        problems += [f"frozen file changed: {p}" for p in changed]
        say(f"Pre-session snapshot: {len(snapshot['files'])} files, unchanged: "
            f"{len(snapshot['files']) - len(changed) - len(accepted)}, reviewed changes: {accepted or 'none'}, "
            f"other changes: {len(changed)}")
    for p in problems:
        say(f"  PROBLEM {p}")
    say(f"Verified {len(canonical.conditions)} canonical conditions, sealed evidence and "
        f"{len(manifest.get('files', {}))} artifacts: {len(problems)} problems")
    return 3 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Session 8: generalisation and error analysis", allow_abbrev=False)
    parser.add_argument("--config", default="configs/session8_evaluation.yaml")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--authorize-execution", action="store_true",
                        help="required for stages that run inference, read test labels, train or register evidence")
    args = parser.parse_args(argv)
    try:
        if args.stage in EXECUTION_STAGES and not args.authorize_execution:
            raise RuntimeError(f"--stage {args.stage} executes the experiment (inference, FD003 test labels or "
                               "training). Use verify, derive or report to work from saved results, or pass "
                               "--authorize-execution deliberately.")
        cfg = load_config(args.config)
        authorization = ExecutionAuthorization("--authorize-execution") if args.authorize_execution else None
        if args.stage in EXECUTION_STAGES:
            require_new_location(cfg)                   # before the lock file or anything else is written
        if args.stage == "validate":
            canonical, inputs = load_canonical(cfg), load_inputs(cfg)
            say(f"{len(canonical.conditions)} canonical conditions; windows "
                f"{ {k: len(v.X) for k, v in inputs.items()} }")
            return 0
        if args.stage == "verify":
            return stage_verify(cfg)
        with execution_lock(resolve_path(cfg["outputs"]["working_dir"])):
            if args.stage == "predict":
                stage_predict(cfg, authorization=authorization)
            if args.stage in ("evaluate", "all"):
                stage_evaluate(cfg, authorization=authorization)
            if args.stage in ("sensitivity", "all"):
                stage_sensitivity(cfg, authorization=authorization)
            if args.stage in ("register", "all"):
                say(f"Evidence manifest: {register_evidence(cfg, SEAL_NOTE, authorization=authorization)}")
                say("Review it and add it to the trusted reference before deriving or reporting.")
            if args.stage == "derive":
                stage_derive(cfg)
            if args.stage == "report":
                stage_report(cfg)
        return 0
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
