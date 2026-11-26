"""Resumable, sequential execution of the Session 7 conditions.

Every condition has one status record (written atomically) next to its outputs. A condition is
reused only when its evidence is intact (session7_integrity.validate_record) AND it was produced by
the present source state (execution code, settings, commit, clean/dirty). Records produced by any
other source state are never overwritten: execution is refused until the working directory is moved.
A record of the present source state that fails its checks is archived and the condition run again.
Records left 'running' by an interruption have their MLflow runs closed as KILLED.
Only one process may write at a time (session7_integrity.execution_lock). Training, evaluation, timing and logging are
the unchanged Session 2 engine (`run_experiment`), pointed at Session 7 output files only.
"""
import copy
import json
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .benchmark_config import BenchmarkConfig, load_benchmark_config, resolve_path
from .experiment import run_experiment
from .metrics import compute_metrics
from .session7_integrity import RunStore, execution_lock, provenance_mismatch, validate_record
from .session7_models import build_condition_model
from .session7_plan import sha256_file, sha256_json, subset_data
from .tracking import MlflowTracker, mlflow

FINGERPRINT_MODULES = ("benchmark_config", "benchmark_data", "checks", "classical_models", "convolutional_models",
                       "data_loading", "evaluation", "experiment", "metrics", "recurrent_models", "results", "rul",
                       "scaling", "sequence_data", "session7_execution", "session7_models", "session7_plan",
                       "splitting", "tabular_features", "timing", "tracking", "transformer_models", "windows")


# every source file that produces training results, checks them or writes the research outputs
SOURCE_MODULES = (*FINGERPRINT_MODULES, "environment", "run_session7", "session3_outputs", "session7_analysis",
                  "session7_integrity", "session7_outputs")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_fingerprint():
    folder = Path(__file__).resolve().parent
    return sha256_json({m: sha256_file(folder / f"{m}.py") for m in FINGERPRINT_MODULES})


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {f"src/{m}.py": sha256_file(folder / f"{m}.py") for m in sorted(SOURCE_MODULES)}


def execution_fingerprint(cfg):
    return sha256_json({k: cfg[k] for k in ("benchmark", "execution", "session1", "session2")})


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def read_record(path):
    """The stored record, or None when it is absent or unreadable (half-written / corrupted)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class ConditionTracker(MlflowTracker):
    """The Session 2 tracker plus Session 7 tags/params; reports the run ID as soon as it exists."""

    def __init__(self, tracking_dir, experiment_name, tags, params, on_start):
        super().__init__(tracking_dir, experiment_name)
        self.tags, self.params, self.on_start = tags, params, on_start

    @contextmanager
    def start_run(self, run_name, tags=None):
        with super().start_run(self.tags["s7.experiment_id"], tags={**(tags or {}), **self.tags}) as run_id:
            self.on_start(run_id)
            mlflow.log_params(self.params)
            yield run_id


class Executor:
    def __init__(self, cfg, text, config_path, plan, sources, data, environment):
        self.cfg, self.plan, self.sources, self.data, self.environment = cfg, plan, sources, data, environment
        base = load_benchmark_config(resolve_path(cfg["session2"]["config"]))
        self.work = resolve_path(cfg["outputs"]["working_dir"])
        values = copy.deepcopy(base.values)
        values["session1"] = copy.deepcopy(cfg["session1"])
        values["benchmark"] = copy.deepcopy(cfg["benchmark"])
        # isolated destinations: the historical benchmark table is never written by Session 7
        values["outputs"] = {"results_csv": str(self.work / "engine_results.csv"),
                             "results_json": str(self.work / "engine_results.json"),
                             "runs_dir": str(self.work / "runs")}
        self.values, self.text, self.config_path = values, text, config_path
        self.tracking_dir = resolve_path(base["tracking"]["tracking_dir"])
        self.experiment_name = cfg["execution"]["experiment_name"]
        if self.experiment_name == base["tracking"]["experiment_name"]:
            raise ValueError("Session 7 must use its own MLflow experiment")
        self.code, self.settings = code_fingerprint(), execution_fingerprint(cfg)

    def record_path(self, condition):
        return self.work / "conditions" / f"{condition['experiment_id']}.json"

    def _client(self):
        MlflowTracker(self.tracking_dir, self.experiment_name)
        return mlflow.MlflowClient()

    def provenance(self):
        git = self.environment["git"]
        return {"code_fingerprint": self.code, "execution_fingerprint": self.settings,
                "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]}

    def check_completed(self, condition, record=None):
        """(reusable, reason): same source state and intact evidence."""
        record = record if record is not None else read_record(self.record_path(condition))
        if record is None:
            return False, "no readable record"
        if record.get("status") != "completed":
            return False, f"status {record.get('status')}"
        if record.get("identity_sha256") != condition["identity_sha256"]:
            return False, "condition identity changed"
        mismatch = provenance_mismatch(record, self.provenance())
        if mismatch:
            return False, mismatch[0]
        return validate_record(condition, record, self.data.y["validation"], RunStore(self.tracking_dir),
                               self.experiment_name, sha256_file)

    def refuse_foreign_records(self):
        """Raise before anything is written if a stored record comes from another source state."""
        current, foreign, reasons = self.provenance(), 0, set()
        for condition in self.plan["conditions"]:
            record = read_record(self.record_path(condition))
            mismatch = provenance_mismatch(record, current) if record is not None else []
            foreign += bool(mismatch)
            reasons.update(mismatch)
        if foreign:
            raise RuntimeError(
                f"{foreign} records in {self.work} were produced by another source state "
                f"({', '.join(sorted(reasons))}). They are never reused or overwritten. Inspect them with "
                "--stage verify, publish them with --stage publish, or move the working directory elsewhere "
                "before a new execution.")

    def _archive(self, path):
        """Keep a copy of a record that is about to be replaced."""
        folder = path.parent / "archive"
        folder.mkdir(parents=True, exist_ok=True)
        n = 1
        while (target := folder / f"{path.stem}.{n}.json").exists():
            n += 1
        target.write_bytes(path.read_bytes())

    def _close_interrupted(self, record):
        run_id = record.get("mlflow_run_id")
        if run_id:
            try:
                client = self._client()
                client.set_tag(run_id, "s7.status", "interrupted")
                if client.get_run(run_id).info.status == "RUNNING":
                    client.set_terminated(run_id, status="KILLED")
            except Exception:                                   # the run may never have been stored
                pass

    def run_condition(self, condition):
        """Run one condition unless a valid result already exists. Returns the record."""
        path = self.record_path(condition)
        previous = read_record(path)
        if previous is not None:
            reusable, _ = self.check_completed(condition, previous)
            if reusable:
                return {**previous, "reused": True}
            if provenance_mismatch(previous, self.provenance()):
                raise RuntimeError(f"{path} comes from another source state and will not be overwritten")
            if previous.get("status") == "running":
                self._close_interrupted(previous)
        if path.exists():
            self._archive(path)
        history = list((previous or {}).get("previous_attempts", []))
        if previous is not None:
            history.append({k: previous.get(k) for k in ("status", "mlflow_run_id", "started_at", "ended_at", "error")}
                           | ({"status": "interrupted"} if previous.get("status") == "running" else {}))
        git = self.environment["git"]
        record = {**condition, "status": "running", "attempt": len(history) + 1, "previous_attempts": history,
                  "started_at": now(), "ended_at": None, "git_commit": git["git_commit"],
                  "git_dirty": git["git_dirty"], "code_fingerprint": self.code,
                  "execution_fingerprint": self.settings, "source_hashes": source_hashes(),
                  "mlflow_run_id": None, "error": None}
        atomic_write_json(path, record)
        wall = time.perf_counter()

        def started(run_id):
            record["mlflow_run_id"] = run_id
            atomic_write_json(path, record)

        try:
            record.update(self._execute(condition, started))
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          traceback=traceback.format_exc(limit=8))
        except BaseException:                                   # Ctrl-C / termination
            record.update(status="interrupted", ended_at=now())
            atomic_write_json(path, record)
            raise
        record["ended_at"], record["wall_time_s"] = now(), time.perf_counter() - wall
        atomic_write_json(path, record)
        if record["mlflow_run_id"]:
            self._client().set_tag(record["mlflow_run_id"], "s7.status", record["status"])
        return record

    def _execute(self, condition, started):
        data = subset_data(self.data, condition["sensor_indices"])
        if data.sensors != condition["sensors"]:
            raise ValueError("sensor order does not match the plan")
        values = copy.deepcopy(self.values)
        tag = condition["strategy"] + (str(condition["subset_seed"]) if condition["subset_seed"] else "")
        values["benchmark"]["sensor_set_name"] = f"s7_{tag}_k{condition['budget']}"
        bench = BenchmarkConfig(values=values, path=self.config_path, text=self.text)
        model = build_condition_model(condition["model"], self.sources, data, execution=self.cfg["execution"],
                                      device=self.cfg["execution"]["device"])
        tags = {"session": "7", "s7.experiment_id": condition["experiment_id"], "s7.model": condition["model"],
                "s7.budget": str(condition["budget"]), "s7.strategy": condition["strategy"],
                "s7.verification": str(bool(self.environment["git"]["git_dirty"]))}
        params = {"s7.experiment_id": condition["experiment_id"], "s7.family": condition["family"],
                  "s7.budget": condition["budget"], "s7.strategy": condition["strategy"],
                  "s7.subset_seed": condition["subset_seed"] or "", "s7.training_seed": condition["training_seed"],
                  "s7.sensor_indices": ",".join(map(str, condition["sensor_indices"])),
                  "s7.memberships": ",".join(condition["memberships"]),
                  "s7.definition_sha256": condition["definition_sha256"],
                  "s7.identity_sha256": condition["identity_sha256"]}
        tracker = ConditionTracker(self.tracking_dir, self.experiment_name, tags, params, started)
        result = run_experiment(model, data, bench, tracker, self.environment, config_filename="session7_config.yaml",
                                extra_columns={"experiment_id": condition["experiment_id"],
                                               "budget": condition["budget"], "strategy": condition["strategy"]})
        run_dir = resolve_path(values["outputs"]["runs_dir"]) / result.run_id
        pred = result.predictions["validation"]
        if set(result.predictions) != {"validation"}:
            raise RuntimeError("only validation predictions may be produced in Session 7")
        summary = getattr(model, "fit_summary_", None)
        files = ["predictions.npz", *[str(Path(p).relative_to(run_dir)) for p in result.artifact_files]]
        out = {"status": "completed" if not summary or summary.get("stable", True) else "unstable",
               "run_dir": str(run_dir), "metrics": compute_metrics(data.y["validation"], pred),
               "predictions_sha256": result.prediction_digests["validation"],
               "n_parameters": result.n_parameters, "train_time_s": result.train_time_s,
               "inference_latency_ms_per_batch": result.timing["latency_ms_per_batch"],
               "inference_latency_ms_per_sample": result.timing["latency_ms_per_sample"],
               "throughput_samples_per_s": result.timing["throughput_samples_per_s"],
               "model_size_bytes": result.model_size_bytes,
               "files": {rel: sha256_file(run_dir / rel) for rel in files}}
        if summary:
            out.update(best_epoch=summary["best_epoch"], epochs_run=summary["epochs_run"],
                       stop_reason=summary["stop_reason"], weights_sha256=model.weights_sha256())
        return out


def reference_values(cfg):
    """Canonical full-sensor validation rows of Sessions 3-6 (read only)."""
    rows = json.loads(resolve_path(cfg["frozen_sources"]["historical_results"]).read_text())["results"]
    out = {}
    for model in cfg["matrix"]["models"]:
        mine = [r for r in rows if r["model"] == model and r["eval_set"] == "validation"
                and str(r.get("git_dirty")).lower() == "false"]
        if mine:
            out[model] = mine[-1]
    return out


def reproducibility_gate(cfg, sources, records):
    """Compare each full-sensor run with the canonical predictions (and weights for deep models)."""
    references, out = reference_values(cfg), []
    for record in records:
        model = record["model"]
        ref = references.get(model)
        if record.get("status") != "completed" or ref is None:
            out.append({"model": model, "passed": False,
                        "reason": "no canonical reference" if ref is None else f"status {record.get('status')}"})
            continue
        frozen = sources.get(model, {}).get("checkpoint", {}).get("weights_sha256") if model in sources else None
        entry = {"model": model, "experiment_id": record["experiment_id"], "sensors_ok": record["budget"] == 14,
                 "rmse": record["metrics"]["rmse"], "canonical_rmse": ref["rmse"],
                 "rmse_difference": record["metrics"]["rmse"] - ref["rmse"],
                 "predictions_identical": record["predictions_sha256"] == ref["predictions_sha256"],
                 "weights_identical": None if frozen is None else record.get("weights_sha256") == frozen,
                 "n_parameters": record["n_parameters"], "canonical_n_parameters": ref.get("n_parameters")}
        entry["passed"] = bool(entry["sensors_ok"] and entry["predictions_identical"]
                               and entry["weights_identical"] is not False
                               and (ref.get("n_parameters") in (None, "") or
                                    int(ref["n_parameters"]) == record["n_parameters"]))
        out.append(entry)
    return {"checks": out, "passed": bool(out) and all(c["passed"] for c in out)}
