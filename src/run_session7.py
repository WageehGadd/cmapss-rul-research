"""Session 7 entry point: sensor budget x model architecture.

    python -m src.run_session7 --config configs/session7_main_experiment.yaml [--stage STAGE]

STAGE: validate  check the configuration, frozen sources and data (writes nothing)
       verify    check every stored record and its evidence (writes nothing)
       publish   write report, figures and summaries of the stored records, as they are, into a new
                 --output-dir (training records and the original outputs are not touched)
       plan      write / verify the immutable experiment plan
       run       execute every condition that has no valid result (stage 1 full sensor + gate,
                 stage 2 ranked subsets, stage 3 random controls); `resume` is the same operation
       summarize build the master and aggregate tables from the records
       report    figures, report and artifacts/session7
       all       plan, run, summarize, report (default)
The configured report, figures and artifacts are written only when every record is complete and
valid; otherwise they go to <working_dir>/incomplete_outputs. Stages that write refuse records
produced by another source state.
Exit codes: 0 complete, 1 error, 2 reproducibility gate failed, 3 conditions missing or invalid.
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .benchmark_config import PROJECT_ROOT, load_benchmark_config, resolve_path
from .environment import capture_environment
from .session3_outputs import write_csv, write_json
from .session7_analysis import (OK, degradation_rows, load_predictions, master_rows, pareto_front,
                                per_engine_metrics, planned_comparisons, random_control_rows, ranked_vs_random_rows,
                                robustness_rows)
from .session7_execution import (Executor, atomic_write_json, code_fingerprint, execution_fingerprint, read_record,
                                 reproducibility_gate, source_hashes)
from .session7_integrity import RunStore, execution_lock, provenance_mismatch, validate_record
from .session7_outputs import (FIGURES, conclusions, figure_heatmap, figure_pareto, figure_per_engine,
                               figure_ranked_vs_random, figure_rmse_vs_budget, write_package, write_report)
from .session7_plan import (build_plan, load_frozen_sources, load_session7_config, load_train_validation, sha256_file)

STAGES = ("validate", "verify", "publish", "plan", "run", "resume", "summarize", "report", "all")
# versioned evidence documents, read (never written) by the report
INCIDENT_PATH = PROJECT_ROOT / "reports/session7_incidents/gradient_boosting_memory_layout.json"
CONDITIONS_PATH = PROJECT_ROOT / "reports/session7_incidents/execution_conditions.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Session 7: sensor budget x architecture")
    parser.add_argument("--config", default="configs/session7_main_experiment.yaml")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="execute at most this many new conditions, then stop (resume later)")
    parser.add_argument("--output-dir", default=None, help="new or empty folder for --stage publish")
    args = parser.parse_args(argv)
    try:
        return _run(args.config, args.stage, args.limit, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"ERROR: {err}", file=sys.stderr, flush=True)
        return 1


def say(text):
    print(text, flush=True)


class Session:
    def __init__(self, config_path):
        self.cfg, self.text, self.config_path = load_session7_config(config_path)
        self.sources = load_frozen_sources(self.cfg)
        self.data, self.data_identity = load_train_validation(self.cfg)
        self.bench = load_benchmark_config(resolve_path(self.cfg["session2"]["config"]))
        self.work = resolve_path(self.cfg["outputs"]["working_dir"])
        self.plan = build_plan(self.cfg, self.sources, self.data, self.data_identity, self.bench["timing"])
        try:
            self.config_ref = str(self.config_path.relative_to(PROJECT_ROOT))
        except ValueError:
            self.config_ref = str(self.config_path)

    def environment(self):
        env = capture_environment(PROJECT_ROOT, [*self.bench["environment"]["library_names"], "torch"],
                                  {"session7_config": self.config_ref})
        env["torch"] = {"version": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
        return env

    def write_plan(self):
        """The plan is written once; a later run must rebuild exactly the same conditions."""
        path = self.work / "experiment_plan.json"
        stored = read_record(path)
        if stored is None:
            atomic_write_json(path, self.plan)
        else:
            self.check_plan()
        self.work.mkdir(parents=True, exist_ok=True)
        index = self.work / "validation_index.npz"
        if not index.exists():                                  # never rewrite stored evidence
            np.savez(index, unit_ids=self.data.unit_ids["validation"],
                     end_cycles=self.data.end_cycles["validation"], y=self.data.y["validation"])
        return path

    def records(self):
        out = {}
        for c in self.plan["conditions"]:
            record = read_record(self.work / "conditions" / f"{c['experiment_id']}.json")
            if record is not None:
                out[c["experiment_id"]] = record
        return out

    def validated_records(self):
        """Records for aggregation: a 'completed' record whose evidence fails a check is relabelled
        'invalid' in memory (the stored file is not changed) and loses its metrics."""
        store = RunStore(resolve_path(self.bench["tracking"]["tracking_dir"]))
        experiment = self.cfg["execution"]["experiment_name"]
        records, problems = self.records(), {}
        for c in self.plan["conditions"]:
            record = records.get(c["experiment_id"])
            if record is None or record.get("status") != OK:
                continue
            valid, reason = validate_record(c, record, self.data.y["validation"], store, experiment, sha256_file)
            if not valid:
                problems[c["experiment_id"]] = reason
                records[c["experiment_id"]] = {k: v for k, v in record.items() if k != "metrics"} | {
                    "status": "invalid", "error": reason}
        return records, problems

    def current_provenance(self):
        git = self.environment()["git"]
        return {"code_fingerprint": code_fingerprint(), "execution_fingerprint": execution_fingerprint(self.cfg),
                "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]}

    def check_plan(self):
        stored = read_record(self.work / "experiment_plan.json")
        if stored is not None and [c["identity_sha256"] for c in stored["conditions"]] != \
                [c["identity_sha256"] for c in self.plan["conditions"]]:
            raise ValueError(f"the experiment plan differs from the predeclared one in {self.work}")


def execute(session, limit):
    cfg, plan = session.cfg, session.plan
    environment = session.environment()
    executor = Executor(cfg, session.text, session.config_path, plan, session.sources, session.data, environment)
    free_gb = shutil.disk_usage(session.work).free / 1e9
    say(f"Free disk: {free_gb:.1f} GB; source commit {environment['git']['git_commit'][:12]}, "
        f"dirty={environment['git']['git_dirty']}")
    if free_gb < 5:
        raise RuntimeError("less than 5 GB free; refusing to start")
    executed, reused, started = 0, 0, time.perf_counter()
    for stage in (1, 2, 3):
        conditions = [c for c in plan["conditions"] if c["stage"] == stage]
        say(f"\nStage {stage}: {len(conditions)} conditions")
        for i, condition in enumerate(conditions, 1):
            reusable, _ = executor.check_completed(condition)
            if not reusable and limit is not None and executed >= limit:
                say(f"Stopping after {executed} new conditions (--limit).")
                return executed, reused
            record = executor.run_condition(condition)
            if record.get("reused"):
                reused += 1
                continue
            executed += 1
            value = f"RMSE {record['metrics']['rmse']:.3f}" if "metrics" in record else record.get("error", "")
            say(f"  [{stage}:{i:>3}/{len(conditions)}] {condition['experiment_id']:<48} {record['status']:<9} "
                f"{value}  ({record['wall_time_s']:.0f}s, elapsed {(time.perf_counter() - started) / 60:.1f} min)")
        if stage == 1:
            records = session.records()
            gate = reproducibility_gate(cfg, session.sources,
                                        [records.get(c["experiment_id"], {**c, "status": "not_run"})
                                         for c in conditions])
            atomic_write_json(session.work / "reproducibility_gate.json", gate)
            for check in gate["checks"]:
                say(f"  gate {check['model']:<18} passed={check['passed']} rmse={check.get('rmse')} "
                    f"canonical={check.get('canonical_rmse')}")
            if not gate["passed"]:
                raise GateFailed()
            estimate_runtime(session, records)
    return executed, reused


class GateFailed(Exception):
    pass


def estimate_runtime(session, records):
    """Predicted remaining compute from the measured full-sensor runs."""
    wall = {r["model"]: r["wall_time_s"] for r in records.values()
            if r.get("strategy") == "full" and r.get("status") == OK}
    remaining = [c for c in session.plan["conditions"]
                 if records.get(c["experiment_id"], {}).get("status") != OK]
    seconds = sum(wall.get(c["model"], 0.0) for c in remaining)
    path = session.work / "runtime_estimate.json"
    if not path.exists():
        atomic_write_json(path, {"full_sensor_wall_time_s": wall, "remaining_conditions": len(remaining),
                                 "estimated_remaining_s": seconds,
                                 "note": "upper-bound style estimate: reduced inputs are not slower"})
    say(f"Estimated remaining compute: {seconds / 3600:.2f} h for {len(remaining)} conditions")


def summarize(session, write=True):
    """Aggregate the validated records. With write=False nothing is stored."""
    cfg, plan = session.cfg, session.plan
    records, problems = session.validated_records()
    for key, reason in problems.items():
        say(f"  INVALID {key}: {reason}")
    rows = master_rows(plan, records)
    work = session.work
    columns = list(rows[0])
    if write:
        write_csv(work / "master_experiments.csv", rows, columns)
        atomic_write_json(work / "master_experiments.json", {"schema_version": 1, "columns": columns,
                                                             "results": rows})

    y, units = session.data.y["validation"], session.data.unit_ids["validation"]
    engine_rmse, engine_rows = {}, []
    for r in rows:
        if r["status"] == OK:
            per = per_engine_metrics(y, load_predictions(r), units)
            engine_rmse[r["experiment_id"]] = {e["unit_id"]: e["rmse"] for e in per}
            engine_rows += [{"experiment_id": r["experiment_id"], **e} for e in per]
    if engine_rows and write:
        write_csv(work / "per_engine_metrics.csv", engine_rows, list(engine_rows[0]))

    models, budgets = cfg["matrix"]["models"], cfg["matrix"]["budgets"]
    degradation = degradation_rows(rows)
    versus = ranked_vs_random_rows(rows)
    robustness = robustness_rows(rows, float(cfg["analysis"]["close_to_full_tolerance_pct"]), budgets)
    stats = planned_comparisons(rows, engine_rmse, cfg["analysis"]["statistics"], models, budgets)
    pareto = pareto_front(rows, ["val_rmse", "inference_latency_ms_per_batch"])
    pareto3 = pareto_front(rows, ["val_rmse", "inference_latency_ms_per_batch", "budget"])
    random = random_control_rows(rows)
    gate = read_record(work / "reproducibility_gate.json") or {"checks": [], "passed": False}
    for name, data in (("degradation", degradation), ("random_controls", random), ("ranked_vs_random", versus),
                       ("robustness", robustness), ("statistics", stats)):
        if write:
            atomic_write_json(work / f"{name}.json", data)
    if write:
        atomic_write_json(work / "pareto.json", {"rmse_latency": pareto, "rmse_latency_sensors": pareto3})
    return {"records": records, "rows": rows, "engine_rmse": engine_rmse, "degradation": degradation,
            "versus": versus, "robustness": robustness, "statistics": stats, "pareto_ids": pareto,
            "pareto3_ids": pareto3, "random": random, "gate": gate, "problems": problems}


def _definition_text(model, sources):
    if model in ("random_forest", "gradient_boosting"):
        return ", ".join(f"{k}={v}" for k, v in sources["selected_models"]["models"][model]["params"].items())
    skip = ("n_sensors", "n_parameters", "head", "readout", "receptive_field")
    return ", ".join(f"{k}={v}" for k, v in sources[model]["architecture"].items() if k not in skip)


def output_paths(session, root=None):
    """Configured destinations, or the same layout under `root`."""
    outputs = session.cfg["outputs"]
    if root is None:
        return {"report": resolve_path(outputs["report"]), "figures": resolve_path(outputs["figures_dir"]),
                "artifacts": resolve_path(outputs["frozen_dir"])}
    root = Path(root)
    return {"report": root / Path(outputs["report"]).name, "figures": root / "figures",
            "artifacts": root / "artifacts"}


def _refuse_other_execution(artifacts, rows):
    """Published outputs of a different execution are historical evidence: never replace them."""
    manifest = read_record(artifacts / "manifest.json")
    if manifest is None:
        return
    new = {r["experiment_id"]: r["mlflow_run_id"] for r in rows if r["status"] == OK}
    for key, old in manifest.get("conditions", {}).items():
        if old.get("mlflow_run_id") and key in new and new[key] != old["mlflow_run_id"]:
            raise RuntimeError(f"{artifacts} holds the outputs of another execution; move it before writing "
                               "new outputs")


def _execution_provenance(records):
    """What the records say about how they were produced (not the state at report time)."""
    done = [r for r in records.values() if r.get("status") == OK]
    environments = {}
    for r in done:
        env = read_record(Path(r["run_dir"]) / "environment.json")
        if env:
            key = json.dumps({"python": env["python"]["version"], "platform": env["os"]["platform"],
                              "hardware": env["hardware"]["description"], "libraries": env["libraries"],
                              "thread_env": env["thread_env"], "git": env["git"]}, sort_keys=True)
            environments[key] = environments.get(key, 0) + 1
    recorded = sorted({r["code_fingerprint"] for r in done})
    return {"code_fingerprints": recorded, "current_code_fingerprint": code_fingerprint(),
            "source_matches_current": recorded == [code_fingerprint()],
            "per_record_source_hashes_recorded": bool(done) and all("source_hashes" in r for r in done),
            "git_commits": sorted({r["git_commit"] for r in done}),
            "git_dirty_values": sorted({bool(r["git_dirty"]) for r in done}),
            "first_started_at": min((r["started_at"] for r in done), default=None),
            "last_ended_at": max((r["ended_at"] for r in done), default=None),
            "environments": [{**json.loads(k), "n_records": n} for k, n in environments.items()]}


def report(session, summary, paths):
    cfg, plan = session.cfg, session.plan
    rows, records = summary["rows"], summary["records"]
    environment = session.environment()
    git = environment["git"]
    dirty_flags = {r["git_dirty"] for r in rows if r["status"] == OK}
    commits = sorted({r["git_commit"] for r in rows if r["git_commit"]})
    run_git = {"git_commit": commits[0] if len(commits) == 1 else ", ".join(commits) or git["git_commit"],
               "git_dirty": bool(True in dirty_flags) if dirty_flags else git["git_dirty"]}
    budgets, models = cfg["matrix"]["budgets"], cfg["matrix"]["models"]
    con = conclusions(rows, summary["degradation"], summary["versus"], summary["robustness"], summary["statistics"],
                      summary["pareto_ids"], budgets, float(cfg["analysis"]["close_to_full_tolerance_pct"]))
    _refuse_other_execution(paths["artifacts"], rows)
    fig_dir = paths["figures"]
    figure_rmse_vs_budget(fig_dir / FIGURES[0], rows, models, budgets, cfg["matrix"]["full_budget"])
    figure_pareto(fig_dir / FIGURES[1], rows, summary["pareto_ids"])
    figure_ranked_vs_random(fig_dir / FIGURES[2], rows, models, budgets)
    figure_heatmap(fig_dir / FIGURES[3], rows, models, budgets)
    figure_per_engine(fig_dir / FIGURES[4], rows, summary["engine_rmse"], models, budgets)

    seen, subsets = set(), []
    for c in plan["conditions"]:
        key = (c["budget"], c["strategy"], c["subset_seed"])
        if c["model"] == models[0] and key not in seen:
            seen.add(key)
            subsets.append({"budget": c["budget"], "strategy": f"{c['strategy']} {c['subset_seed'] or ''}".strip(),
                            "sensors": ", ".join(s.replace("sensor_", "s") for s in c["sensors"])})
    subsets.sort(key=lambda s: (-s["budget"], s["strategy"]))
    by_key = {(r["model"], r["budget"], r["strategy"]): r for r in rows if r["status"] == OK}
    pairs = [{"model": m, "budget": k, "correlation_rmse": by_key[(m, k, "correlation")]["val_rmse"],
              "permutation_rmse": by_key[(m, k, "permutation")]["val_rmse"],
              "permutation_minus_correlation": by_key[(m, k, "permutation")]["val_rmse"]
              - by_key[(m, k, "correlation")]["val_rmse"]}
             for m in models for k in budgets if (m, k, "correlation") in by_key and (m, k, "permutation") in by_key]
    efficiency = []
    for m in models:
        for k in [cfg["matrix"]["full_budget"], *budgets]:
            group = [r for r in rows if r["status"] == OK and r["model"] == m and r["budget"] == k]
            if group:
                med = lambda key: float(np.median([float(r[key] or 0) for r in group]))
                efficiency.append({"model": m, "budget": k, "conditions": len(group),
                                   "median_latency_ms": med("inference_latency_ms_per_batch"),
                                   "median_train_time_s": med("train_time_s"),
                                   "median_model_size_bytes": med("model_size_bytes"),
                                   "median_n_parameters": med("n_parameters")})
    front = sorted((r for r in rows if r["experiment_id"] in summary["pareto_ids"]),
                   key=lambda r: r["inference_latency_ms_per_batch"])
    pareto_rows = [{**r, "rank": i, "subset": f"{r['strategy']} {r['subset_seed']}".strip()}
                   for i, r in enumerate(front, 1)]
    total_wall = sum(r.get("wall_time_s", 0.0) for r in records.values())
    artifacts = resolve_path("artifacts")
    upstream = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in
                [*sorted(artifacts.glob("session[1-6]/manifest.json")), *sorted(resolve_path("configs").glob("session[1-6]_*.yaml"))]}
    incident = read_record(INCIDENT_PATH) or read_record(session.work / "gate_diagnosis.json")
    manifest = {
        "schema_version": 2, "session": 7, "config_ref": session.config_ref,
        "config_sha256": sha256_file(session.config_path), "upstream_files": upstream,
        "source_hashes": plan["source_hashes"], "data": plan["data"], "training_protocol": plan["training_protocol"],
        "training_seeds": cfg["matrix"]["training_seeds"], "subset_seeds": cfg["matrix"]["random_subset_seeds"],
        "report_environment": environment, "git_commit": run_git["git_commit"], "git_dirty": run_git["git_dirty"],
        "canonical": False if run_git["git_dirty"] else None,
        "evidence_class": ("verification runs executed from an uncommitted working tree; not canonical"
                           if run_git["git_dirty"] else "runs executed from a clean commit"),
        "official_test_evaluation": "not performed",
        "execution_provenance": _execution_provenance(records), "report_source_hashes": source_hashes(),
        "record_files_sha256": {r["experiment_id"]: sha256_file(session.work / "conditions" /
                                                                f"{r['experiment_id']}.json")
                                for r in rows if r["status"] == OK},
        "mlflow_experiment": cfg["execution"]["experiment_name"],
        "conditions": {r["experiment_id"]: {"status": r["status"], "mlflow_run_id": r["mlflow_run_id"],
                                            "predictions_sha256": r["predictions_sha256"]} for r in rows},
        "n_planned": len(rows), "n_completed": con["completed"],
        "superseded_runs": (incident or {}).get("superseded_runs", {}),
        "incident_record_sha256": sha256_file(INCIDENT_PATH) if INCIDENT_PATH.exists() else None,
        "failed_conditions": [r["experiment_id"] for r in rows if r["status"] in ("failed", "unstable", "interrupted", "invalid")],
        "missing_conditions": [r["experiment_id"] for r in rows if r["status"] == "not_run"]}
    observed = read_record(CONDITIONS_PATH)            # observations about one execution, never reused for another
    if observed and observed.get("applies_to_code_fingerprint") not in \
            manifest["execution_provenance"]["code_fingerprints"]:
        observed = None
    ctx = {**summary, "cfg": cfg, "plan": plan, "conclusions": con, "git": run_git, "environment": environment,
           "config_ref": session.config_ref, "definitions": {m: _definition_text(m, session.sources) for m in models},
           "subsets": subsets, "ranking_pairs": pairs, "efficiency": efficiency, "pareto_rows": pareto_rows,
           "total_wall_s": total_wall, "manifest": manifest, "gate_diagnosis": incident,
           "execution_conditions": observed, "artifacts_ref": _ref(paths["artifacts"]),
           "figures_link": os.path.relpath(fig_dir, paths["report"].parent)}
    write_report(paths["report"], ctx)
    write_package(paths["artifacts"], ctx)
    return con


def _ref(path):
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def verify(session):
    """Read-only inspection of the stored evidence."""
    session.check_plan()
    records, problems = session.validated_records()
    for key, reason in problems.items():
        say(f"  INVALID {key}: {reason}")
    ids = [c["experiment_id"] for c in session.plan["conditions"]]
    status = Counter(records[i]["status"] if i in records else "not_run" for i in ids)
    say(f"Record status after integrity checks: {dict(status)}")
    current = session.current_provenance()
    reasons = Counter(reason for r in records.values() for reason in provenance_mismatch(r, current))
    say(f"Differences from the present source state (such records are never reused): {dict(reasons) or 'none'}")
    say(f"Recorded (commit, dirty): "
        f"{dict(Counter((str(r.get('git_commit'))[:12], r.get('git_dirty')) for r in records.values()))}")
    return 0 if status[OK] == len(ids) else 3


def publish(session, output_dir):
    """Report, figures and summaries of the stored records into a new folder."""
    if not output_dir:
        raise ValueError("--stage publish needs --output-dir")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty; publish never overwrites existing files")
    paths, originals = output_paths(session, root), output_paths(session)
    if paths["artifacts"] == originals["artifacts"] or paths["report"] == originals["report"] \
            or root == session.work or session.work in root.parents:
        raise ValueError("the publish folder must be separate from the original outputs and the working directory")
    session.check_plan()
    summary = summarize(session, write=False)
    done = sum(r["status"] == OK for r in summary["rows"])
    if done < session.plan["n_conditions"]:
        say(f"Only {done} of {session.plan['n_conditions']} records are complete and valid; nothing published.")
        return 3
    report(session, summary, paths)
    say(f"Published {done} records to {root}")
    return 0


def _run(config_path, stage, limit, output_dir=None):
    session = Session(config_path)
    plan = session.plan
    say(f"Frozen sources and training/validation data verified. Planned conditions: {plan['n_conditions']} "
        f"({plan['per_model']})")
    if stage == "validate":
        return 0
    if stage == "verify":
        return verify(session)
    if stage == "publish":
        return publish(session, output_dir)
    executor = Executor(session.cfg, session.text, session.config_path, plan, session.sources, session.data,
                        session.environment())
    executor.refuse_foreign_records()                           # before anything is written
    with execution_lock(session.work):
        executor.refuse_foreign_records()
        return _run_locked(session, stage, limit)


def _run_locked(session, stage, limit):
    plan = session.plan
    say(f"Plan: {session.write_plan()}")
    if stage == "plan":
        return 0
    gate_failed = False
    if stage in ("run", "resume", "all"):
        try:
            executed, reused = execute(session, limit)
            say(f"\nExecuted {executed} conditions, reused {reused} valid results.")
        except GateFailed:
            say("REPRODUCIBILITY GATE FAILED: the reduced-sensor matrix was not started.")
            gate_failed = True
    summary = summarize(session)
    done = sum(r["status"] == OK for r in summary["rows"])
    say(f"Completed and valid conditions: {done} of {plan['n_conditions']}")
    if gate_failed:
        return 2
    complete = done == plan["n_conditions"]
    if stage in ("report", "all"):
        paths = output_paths(session, None if complete else session.work / "incomplete_outputs")
        report(session, summary, paths)
        say(f"Report: {paths['report']}")
    if not complete:
        say(f"INCOMPLETE. Resume with: python -m src.run_session7 --config {session.config_ref} --stage all")
        return 3
    say("Matrix complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
