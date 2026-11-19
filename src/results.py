"""Benchmark results table: schema, row construction and append-only CSV/JSON export.

The JSON file is the source of truth: `append_results` loads it, appends the new rows and
rewrites both files, so later sessions add results programmatically without touching old rows.

One row is written per (run, evaluation set). Training time, inference timing and model size are
properties of the run, so they repeat across the rows of the same run.
"""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Columns required by the roadmap (first block) plus traceability fields.
RESULT_COLUMNS = [
    "model", "sensor_set", "sensor_count", "seed",
    "rmse", "mae", "r2", "nasa_score",
    "train_time_s", "inference_latency_ms_per_batch", "inference_latency_ms_per_sample",
    "inference_batch_size", "throughput_samples_per_s", "model_size_bytes",
    "run_id", "eval_set", "n_samples", "target_type", "predictions_sha256",
    "git_commit", "git_dirty", "config_ref", "config_sha256", "session1_config_ref",
    "timestamp_utc",
]


def build_rows(result, *, sensor_set, sensor_count, seed, git, config_ref, config_sha256,
               session1_config_ref):
    """One table row per evaluation set of an EvaluationResult."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for eval_set, m in result.metrics.items():
        rows.append({
            "model": result.model_name,
            "sensor_set": sensor_set,
            "sensor_count": int(sensor_count),
            "seed": int(seed),
            "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "nasa_score": m["nasa_score"],
            "train_time_s": result.train_time_s,
            "inference_latency_ms_per_batch": result.timing["latency_ms_per_batch"],
            "inference_latency_ms_per_sample": result.timing["latency_ms_per_sample"],
            "inference_batch_size": result.timing["batch_size"],
            "throughput_samples_per_s": result.timing["throughput_samples_per_s"],
            "model_size_bytes": result.model_size_bytes,
            "run_id": result.run_id,
            "eval_set": eval_set,
            "n_samples": result.n_samples[eval_set],
            "target_type": result.target_type,
            "predictions_sha256": result.prediction_digests[eval_set],
            "git_commit": git["git_commit"],
            "git_dirty": git["git_dirty"],
            "config_ref": config_ref,
            "config_sha256": config_sha256,
            "session1_config_ref": session1_config_ref,
            "timestamp_utc": timestamp,
        })
        if result.n_parameters is not None:     # optional extra column, absent for older rows
            rows[-1]["n_parameters"] = int(result.n_parameters)
    return rows


def _check_rows(rows):
    for row in rows:
        missing = [c for c in RESULT_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"result row is missing required columns: {missing}")


def load_results(json_path):
    """Rows already stored in the JSON results file (empty list if the file does not exist)."""
    json_path = Path(json_path)
    if not json_path.exists():
        return []
    return json.loads(json_path.read_text())["results"]


def _atomic_write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append_results(rows, csv_path, json_path):
    """Append rows to the results table and rewrite the CSV and JSON exports."""
    _check_rows(rows)
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = load_results(json_path) + list(rows)

    extras = []
    for row in all_rows:
        extras += [k for k in row if k not in RESULT_COLUMNS and k not in extras]
    columns = RESULT_COLUMNS + extras

    document = {"schema_version": SCHEMA_VERSION, "columns": columns, "results": all_rows}
    _atomic_write(json_path, json.dumps(document, indent=2) + "\n")

    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp, csv_path)
    return all_rows
