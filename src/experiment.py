"""Run one benchmark experiment: train, evaluate, save, track and record the results.

This is the entry point every model family uses, so no model needs its own metric, timing,
logging or export code:

    result = run_experiment(model, data, cfg, tracker, environment)
"""
import json
import random
from dataclasses import asdict

import numpy as np

from .benchmark_config import resolve_path
from .evaluation import evaluate_model
from .results import append_results, build_rows
from .timing import time_training


def _flat_params(result, data, cfg, model, environment):
    bench, timing = cfg["benchmark"], cfg["timing"]
    git = environment["git"]
    params = {
        "model": result.model_name,
        "seed": bench["seed"],
        "sensor_set": bench["sensor_set_name"],
        "sensor_count": len(data.sensors),
        "sensors": ",".join(data.sensors),
        "target_type": data.target_type,
        "rul_cap": data.rul_cap,
        "sequence_length": data.sequence_length,
        "evaluation_sets": ",".join(bench["evaluation_sets"]),
        "timing.warmup_runs": timing["warmup_runs"],
        "timing.timed_runs": timing["timed_runs"],
        "timing.batch_size": timing["batch_size"],
        "timing.timing_set": timing["timing_set"],
        "timing.latency_unit": timing["latency_unit"],
        "timing.aggregate": "median",
        "config_ref": cfg.ref,
        "config_sha256": cfg.sha256,
        "session1_config_ref": cfg["session1"]["config"],
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "hardware": environment["hardware"]["description"],
        "python": environment["python"]["version"],
        "platform": environment["os"]["platform"],
    }
    params.update({f"model.{k}": v for k, v in model.get_config().items()})
    return params


def run_experiment(model, data, cfg, tracker, environment):
    """Train `model` on the training split, evaluate it, and record the run everywhere.

    Returns the EvaluationResult; its run_id is the MLflow run ID and appears in the exported
    results rows.
    """
    bench, timing = cfg["benchmark"], cfg["timing"]
    seed = int(bench["seed"])
    git = environment["git"]
    tags = {"model": model.name, "git_commit": str(git["git_commit"]),
            "git_dirty": str(git["git_dirty"]), "target_type": data.target_type}

    with tracker.start_run(model.name, tags=tags) as run_id:
        run_dir = resolve_path(cfg["outputs"]["runs_dir"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        random.seed(seed)
        np.random.seed(seed)
        train_time_s = time_training(model, data.X["train"], data.y["train"])

        eval_sets = {name: (data.X[name], data.y[name]) for name in bench["evaluation_sets"]}
        result = evaluate_model(
            model, eval_sets, data.X[timing["timing_set"]],
            target_type=data.target_type, train_time_s=train_time_s, timing_config=timing,
            artifact_dir=run_dir / "model", run_id=run_id,
            metadata={"seed": seed, "sensor_set": bench["sensor_set_name"], "sensors": data.sensors})

        np.savez(run_dir / "predictions.npz", **result.predictions)
        (run_dir / "session2_config.yaml").write_text(cfg.text)
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        (run_dir / "sensors.json").write_text(json.dumps(data.sensors, indent=2) + "\n")
        (run_dir / "model" / "model_config.json").write_text(json.dumps(model.get_config(), indent=2) + "\n")
        summary = {k: v for k, v in asdict(result).items() if k != "predictions"}
        (run_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")

        tracker.log_evaluation(result, _flat_params(result, data, cfg, model, environment), run_dir)

        rows = build_rows(result, sensor_set=bench["sensor_set_name"], sensor_count=len(data.sensors),
                          seed=seed, git=git, config_ref=cfg.ref, config_sha256=cfg.sha256,
                          session1_config_ref=cfg["session1"]["config"])
        append_results(rows, resolve_path(cfg["outputs"]["results_csv"]),
                       resolve_path(cfg["outputs"]["results_json"]))
    return result
