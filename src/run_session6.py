"""Session 6 entry point: compact Transformer for RUL regression.

    python -m src.run_session6 --config configs/session6_transformer.yaml

Stages: frozen Session 1 windows -> stability gate -> Stage A (8 candidates, last pooling) ->
leading architecture (validation only) -> Stage B (other pooling) -> final Transformer ->
the final model through the Session 2 engine (MLflow, results table; the test set is evaluated
here, after the choice is frozen) -> frozen definition, checkpoint, figures, report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .benchmark_config import PROJECT_ROOT, resolve_path
from .convolutional_study import load_historical_representatives, select_candidate, threshold_near_miss
from .environment import capture_environment
from .experiment import run_experiment
from .recurrent_models import resolve_device
from .sequence_data import load_sequence_data
from .session3_config import to_benchmark_config
from .session3_outputs import session1_identity, write_csv, write_json
from .session6_config import check_protocol_matches_reference, load_session6_config
from .session6_outputs import (comparison_note, file_sha256, freeze_checkpoint, interpretation_notes, latency_notes,
                               plot_architecture_study, plot_family_rmse_vs_latency, plot_prediction_vs_target,
                               plot_training_curves, write_report)
from .tracking import MlflowTracker
from .transformer_models import ARCHITECTURE_KEYS, TransformerRegressor, build_from_definition
from .transformer_study import arch_label, make_model, run_candidate, stability_gate, stage_a_grid

HISTORICAL = ("random_forest", "gradient_boosting", "lstm", "gru", "tcn")
FAMILY_OF = {"random_forest": "classical", "gradient_boosting": "classical", "lstm": "recurrent",
             "gru": "recurrent", "tcn": "convolutional", "transformer": "attention"}
STUDY_COLUMNS = ["stage", "candidate", "run_name", "d_model", "num_heads", "num_layers", "ff_dim", "dropout",
                 "pooling", "sequence_length", "n_parameters", "status", "stop_reason", "best_epoch", "epochs_run",
                 "train_loss_at_best", "best_val_loss", "val_rmse", "val_mae", "val_r2", "val_nasa_score",
                 "weights_sha256", "role"]
RUN_COLUMNS = [*STUDY_COLUMNS, "train_time_s", "mean_epoch_time_s", "max_grad_norm", "mean_clip_fraction",
               "latency_ms_per_batch", "latency_ms_per_sample", "throughput_samples_per_s", "model_size_bytes",
               "mlflow_run_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Session 6: Transformer")
    parser.add_argument("--config", default="configs/session6_transformer.yaml")
    args = parser.parse_args(argv)
    try:
        return _run(args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _relative(path):
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _flat(rec, role=""):
    row = {**rec, "val_rmse": repr(rec["validation"]["rmse"]), "val_mae": repr(rec["validation"]["mae"]),
           "val_r2": repr(rec["validation"]["r2"]), "val_nasa_score": repr(rec["validation"]["nasa_score"]),
           "role": role}
    return row


def _pick(rows, columns):
    return [{k: row.get(k, "") for k in columns} for row in rows]


def _report_row(r):
    return {"candidate": r["run_name"], "n_parameters": r["n_parameters"], "val_rmse": r["validation"]["rmse"],
            "val_mae": r["validation"]["mae"], "val_r2": r["validation"]["r2"],
            "val_nasa": r["validation"]["nasa_score"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
            "train_loss": r["train_loss_at_best"], "val_loss": r["best_val_loss"], "train_time_s": r["train_time_s"],
            "latency_ms": r.get("latency_ms_per_batch", float("nan")),
            "throughput": r.get("throughput_samples_per_s", float("nan")), "status": r["status"],
            "run_id": r["mlflow_run_id"][:8]}


def _near_miss_text(records, chosen, rule):
    miss = threshold_near_miss(records, chosen, rule)
    if miss is None:
        return "no smaller candidate lies outside the 1% threshold."
    return (f"threshold {miss['threshold']:.4f}; closest smaller candidate outside it: {miss['run_name']} "
            f"({miss['n_parameters']:,} parameters, RMSE {miss['value']:.4f}, missed by {miss['margin']:.4f})"
            + (", so the selection is sensitive to the exact threshold."
               if miss["margin"] / miss["threshold"] < 0.005 else "."))


def early_stopping_summary(records, selected_name, min_delta):
    """Which retained epochs differ from the numerical validation-loss minimum (min_delta rule)."""
    differing, selected = [], None
    for r in records:
        if not r["history"] or not r["best_epoch"]:
            continue
        kept = r["history"][r["best_epoch"] - 1]
        lowest = min(r["history"], key=lambda e: e["val_loss"])
        entry = {"run_name": r["run_name"], "retained_epoch": kept["epoch"], "retained_val_loss": kept["val_loss"],
                 "lowest_epoch": lowest["epoch"], "lowest_val_loss": lowest["val_loss"],
                 "gap": kept["val_loss"] - lowest["val_loss"]}
        if entry["gap"] > 0:
            differing.append(entry)
        if r["run_name"] == selected_name:
            selected = entry
    return {"min_delta": float(min_delta), "selected": selected, "differing": differing}


def session7_compatibility(definition, counts, sequence_length):
    """Instantiate the frozen definition for each sensor budget (no training) and check the output."""
    rows = []
    for k in counts:
        model = build_from_definition(definition, n_sensors=k, device="cpu")
        net = model._build().eval()
        with torch.no_grad():
            out = net(torch.zeros(3, sequence_length, k))
        projection = sum(p.numel() for p in net.input_projection.parameters())
        rows.append({"n_sensors": k, "n_parameters": model.n_parameters(), "projection_parameters": projection,
                     "output_shape": str(tuple(out.shape))})
    return rows


def _run(config_path):
    cfg, text, path = load_session6_config(config_path)
    check_protocol_matches_reference(cfg)
    bench_cfg = to_benchmark_config(cfg, text, path)
    timing_cfg, s1, seed = bench_cfg["timing"], cfg["session1"], int(cfg["benchmark"]["seed"])
    device = resolve_device(cfg["execution"]["device"])
    scale, L = float(cfg["training"]["target_scale"]), int(cfg["input"]["sequence_length"])
    work_dir, frozen_dir = resolve_path(cfg["outputs"]["working_dir"]), resolve_path(cfg["outputs"]["frozen_dir"])
    ckpt_dir = resolve_path(cfg["outputs"]["checkpoint_dir"])

    data = load_sequence_data(s1["processed_dir"], s1["config"], cfg["benchmark"]["target_type"], L)
    if data.X["train"].shape[1:] != (L, int(cfg["input"]["n_sensors"])):
        raise ValueError(f"loaded windows have shape {data.X['train'].shape}, config declares "
                         f"[n, {L}, {cfg['input']['n_sensors']}]")
    print(f"Device: {device}, torch {torch.__version__}; windows train {data.X['train'].shape}, validation "
          f"{data.X['validation'].shape}")

    environment = capture_environment(PROJECT_ROOT, [*bench_cfg["environment"]["library_names"], "torch"],
                                      {"session6_config": bench_cfg.ref, "session2_config": cfg["session2"]["config"]})
    environment["torch"] = {"version": torch.__version__, "device": device, "threads": torch.get_num_threads(),
                            "mps_available": bool(torch.backends.mps.is_available()),
                            "cuda_available": bool(torch.cuda.is_available())}

    # 1. stability gate (both poolings); the study does not start unless every check passes
    grid = stage_a_grid(cfg)
    gate = [stability_gate(cfg, {**grid[0], "pooling": p}, data) for p in cfg["study"]["stage_b"]["pooling"]]
    for g in gate:
        print(f"Gate {g['candidate']}: tiny-batch loss {g['first_loss']:.4f} -> {g['last_loss']:.5f}, passed={g['passed']}")
    write_json(work_dir / "gate.json", gate)
    if not all(g["passed"] for g in gate):
        failed = sorted({k for g in gate for k, v in g["checks"].items() if not v})
        raise RuntimeError(f"stability gate failed {failed}; architecture study not started")

    tracker = MlflowTracker(bench_cfg["tracking"]["tracking_dir"], bench_cfg["tracking"]["experiment_name"])
    cand_dir = work_dir / "candidates"
    show = lambda r: print(f"  {r['run_name']:<22} val RMSE {r['validation']['rmse']:8.3f}  best epoch "
                           f"{r['best_epoch']:>2}/{r['epochs_run']:<2} params {r['n_parameters']:>6}  [{r['status']}]")

    # 2. Stage A
    print(f"\nStage A: {len(grid)} candidates")
    stage_a = []
    for i, arch in enumerate(grid):
        stage_a.append(run_candidate(cfg, timing_cfg, arch, data, tracker, "ablation_transformer", environment, i,
                                     cand_dir))
        show(stage_a[-1])
    sel = cfg["selection"]
    leader, rule_a = select_candidate(stage_a, sel["metric"], float(sel["tie_tolerance_rel"]))
    print(f"Stage A leader: {leader['run_name']} (decided by {rule_a['decided_by']})")

    # 3. Stage B: pooling; the leader's own run is reused for its pooling
    pooling_records = []
    for order, pooling in enumerate(cfg["study"]["stage_b"]["pooling"]):
        if pooling == leader["pooling"]:
            pooling_records.append({**leader, "candidate": order})
            continue
        arch = {k: leader[k] for k in ("d_model", "num_heads", "num_layers")} | {"pooling": pooling}
        rec = run_candidate(cfg, timing_cfg, arch, data, tracker, "pooling_transformer", environment,
                            len(grid) + order, cand_dir)
        show(rec)
        pooling_records.append({**rec, "candidate": order})
    final, rule_pool = select_candidate(pooling_records, sel["metric"], float(sel["tie_tolerance_rel"]))
    print(f"Selected Transformer: {final['run_name']} (decided by {rule_pool['decided_by']})")

    # 4. the final model through the Session 2 engine (test evaluated after the choice)
    arch = {k: final[k] for k in ("d_model", "num_heads", "num_layers", "pooling")}
    model = make_model(cfg, arch, data, name="transformer")
    result = run_experiment(
        model, data, bench_cfg, tracker, environment, config_filename="session6_config.yaml",
        extra_columns=lambda _r, m=model: {
            "model_family": "transformer", "d_model": m.d_model, "num_heads": m.num_heads,
            "num_layers": m.num_layers, "ff_dim": m.ff_multiplier * m.d_model, "dropout": m.dropout,
            "pooling": m.pooling, "sequence_length": m.sequence_length,
            "best_epoch": m.fit_summary_["best_epoch"], "epochs_run": m.fit_summary_["epochs_run"], "device": device})
    run_model_dir = resolve_path(bench_cfg["outputs"]["runs_dir"]) / result.run_id / "model"
    reloaded = TransformerRegressor.load(run_model_dir, name="transformer")
    reload_ok = bool(np.array_equal(reloaded.predict(data.X["validation"]), result.predictions["validation"]))
    retrain_matches = model.weights_sha256() == final["weights_sha256"]
    print(f"Final: validation RMSE {result.metrics['validation']['rmse']:.3f}, test RMSE "
          f"{result.metrics['test']['rmse']:.3f}; reload ok={reload_ok}, identical to candidate run={retrain_matches}")

    # 5. frozen definition, study summaries, checkpoint, manifest
    ck_path, ck_sha = freeze_checkpoint(run_model_dir, "transformer", ckpt_dir)
    checkpoint = {"path": _relative(ck_path), "file_sha256": ck_sha, "weights_sha256": model.weights_sha256()}
    architecture = model.architecture()
    definition = {
        "schema_version": 1, "family": "transformer", "architecture": architecture,
        "n_parameters": model.n_parameters(), "training": {**cfg["training"], "seed": seed, "device": device},
        "selection": {"data": "validation only", "stage_a": rule_a, "pooling": rule_pool,
                      "tiebreak": "fewest parameters, then lower inference latency, then declared order"},
        "training_outcome": {k: final[k] for k in ("best_epoch", "epochs_run", "stop_reason", "best_val_loss")},
        "validation_metrics": result.metrics["validation"], "checkpoint": checkpoint,
        "retrain_matches_candidate_weights": retrain_matches,
        "session7": {"rebuild_with": "src.transformer_models.build_from_definition(definition, n_sensors=k)",
                     "varies_with_sensor_count": ["input projection"],
                     "frozen": [*ARCHITECTURE_KEYS, "positional_encoding", "training"]}}
    write_json(frozen_dir / "selected_transformer.json", definition)
    roles = {leader["run_name"]: "stage_a_leader", final["run_name"]: "selected"}
    all_records = stage_a + [{**r, "candidate": len(grid) + r["candidate"]} for r in pooling_records
                             if r["stage"] == "pooling_transformer"]
    write_csv(frozen_dir / "architecture_study.csv",
              _pick([_flat(r, roles.get(r["run_name"], "")) for r in all_records], STUDY_COLUMNS), STUDY_COLUMNS)
    write_json(frozen_dir / "pooling_study.json", {
        "architecture": {k: leader[k] for k in ("d_model", "num_heads", "num_layers")}, "rule": rule_pool,
        "selected_pooling": final["pooling"],
        "results": [{"pooling": r["pooling"], "run_name": r["run_name"], "n_parameters": r["n_parameters"],
                     "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], **r["validation"],
                     "weights_sha256": r.get("weights_sha256", "")} for r in pooling_records]})

    git = environment["git"]
    s1_id = session1_identity(s1, sensors=data.sensors, sequence_length=L, rul_cap=data.rul_cap,
                              target_type=data.target_type, root=PROJECT_ROOT)
    pin = lambda rel: file_sha256(PROJECT_ROOT / rel) if (PROJECT_ROOT / rel).exists() else None
    names = ["selected_transformer.json", "architecture_study.csv", "pooling_study.json"]
    write_json(frozen_dir / "manifest.json", {
        "schema_version": 1, "session": 6, "config_ref": bench_cfg.ref, "config_sha256": bench_cfg.sha256,
        "protocol_reference": cfg["protocol_reference"]["config"],
        "protocol_reference_sha256": pin(cfg["protocol_reference"]["config"]), "seed": seed,
        "target_type": data.target_type, "device": device, "torch_version": torch.__version__,
        "sequence_length": L, "n_sensors": len(data.sensors), "session1": s1_id,
        "session2_config_sha256": pin(cfg["session2"]["config"]),
        "session3_manifest_sha256": pin("artifacts/session3/manifest.json"),
        "session4_manifest_sha256": pin("artifacts/session4/manifest.json"),
        "session5_manifest_sha256": pin("artifacts/session5/manifest.json"),
        "selected_architecture": architecture, "n_parameters": model.n_parameters(),
        "files": {n: file_sha256(frozen_dir / n) for n in names}, "checkpoint": checkpoint,
        "final_run_id": result.run_id, "selected_candidate_run_id": final["mlflow_run_id"],
        "candidate_run_ids": {r["run_name"]: r["mlflow_run_id"] for r in all_records},
        "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]})
    write_csv(work_dir / "architecture_runs.csv",
              _pick([_flat(r, roles.get(r["run_name"], "")) for r in all_records], RUN_COLUMNS), RUN_COLUMNS)
    write_json(work_dir / "selection.json", {"stage_a": rule_a, "pooling": rule_pool, "selected": final["run_name"]})
    write_json(work_dir / "history_selected.json", model.history_)
    session7_rows = session7_compatibility(definition, cfg["stability_gate"]["session7_sensor_counts"], L)

    # 6. comparison with the frozen earlier families (read only), figures, report
    hist = load_historical_representatives(resolve_path(bench_cfg["outputs"]["results_json"]), HISTORICAL)
    historical_test = {name: by_set["test"]["rmse"] for name, by_set in hist.items() if "test" in by_set}
    comparison = []
    for name, by_set in hist.items():
        v = by_set["validation"]
        comparison.append({"family": FAMILY_OF[name], "model": name, "n_parameters": int(v.get("n_parameters") or 0),
                           "val_rmse": v["rmse"], "val_mae": v["mae"], "val_r2": v["r2"], "val_nasa": v["nasa_score"],
                           "latency_ms": v["inference_latency_ms_per_batch"], "train_time_s": v["train_time_s"]})
    mv = result.metrics["validation"]
    comparison.append({"family": "attention", "model": "transformer", "n_parameters": model.n_parameters(),
                       "val_rmse": mv["rmse"], "val_mae": mv["mae"], "val_r2": mv["r2"], "val_nasa": mv["nasa_score"],
                       "latency_ms": result.timing["latency_ms_per_batch"], "train_time_s": result.train_time_s})
    figures = ["fig13_session6_training_curves.png", "fig14_session6_validation_prediction_vs_target.png",
               "fig15_session6_architecture_study.png", "fig16_session6_validation_rmse_vs_latency.png"]
    fig_dir = resolve_path(cfg["outputs"]["figures_dir"])
    plot_training_curves(fig_dir / figures[0], {"transformer": model.history_}, scale)
    plot_prediction_vs_target(fig_dir / figures[1], [("Transformer", data.y["validation"],
                                                     result.predictions["validation"], mv["rmse"])], data.rul_cap)
    plot_architecture_study(fig_dir / figures[2], all_records, final["run_name"])
    plot_family_rmse_vs_latency(fig_dir / figures[3], [
        {"model": c["model"], "family": c["family"], "latency": c["latency_ms"], "rmse": c["val_rmse"],
         "n_parameters": c["n_parameters"]} for c in comparison])
    reproducibility = (
        f"Checkpoint `{checkpoint['path']}` (file SHA-256 `{ck_sha[:16]}...`, weights digest "
        f"`{checkpoint['weights_sha256'][:16]}...`). It reloads to identical predictions: {reload_ok}. Retraining the "
        f"selected architecture through the Session 2 engine gave weights identical to its candidate run: "
        f"{retrain_matches}. Manifest: `artifacts/session6/manifest.json`.")
    write_report(
        resolve_path(cfg["outputs"]["report"]), cfg=cfg, config_ref=bench_cfg.ref, seed=seed, device=device,
        gate=gate, stage_a_rows=[_report_row(r) for r in stage_a],
        pooling_rows=[_report_row(r) for r in pooling_records], rule_a=rule_a, rule_pool=rule_pool,
        near_misses={"Stage A": _near_miss_text(stage_a, leader, rule_a)}, selected=definition,
        final_metrics=[{"eval_set": es, "n_samples": result.n_samples[es], **result.metrics[es]}
                       for es in ("validation", "test")],
        efficiency={"n_parameters": model.n_parameters(), "model_size_bytes": result.model_size_bytes,
                    "best_epoch": model.fit_summary_["best_epoch"], "epochs_run": model.fit_summary_["epochs_run"],
                    "train_time_s": result.train_time_s,
                    "latency_ms_per_batch": result.timing["latency_ms_per_batch"],
                    "latency_ms_per_sample": result.timing["latency_ms_per_sample"],
                    "throughput_per_s": result.timing["throughput_samples_per_s"]},
        comparison=comparison, comparison_text=comparison_note(comparison, float(sel["tie_tolerance_rel"])),
        timing_notes=latency_notes(comparison), reproducibility=reproducibility, session7_rows=session7_rows,
        figures=figures, unstable=[r for r in all_records if r["status"] != "ok"], git=git,
        early_stopping=early_stopping_summary(all_records, final["run_name"], cfg["training"]["min_delta"]),
        interpretation=interpretation_notes(comparison, historical_test, result.metrics["test"]["rmse"], stage_a))

    print("\nFinal Transformer (Session 2 engine):")
    for es in ("validation", "test"):
        m = result.metrics[es]
        print(f"  {es:<10} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  R2 {m['r2']:.4f}  NASA {m['nasa_score']:10.1f}")
    print(f"Run ID: transformer={result.run_id}\nFrozen artifacts: {frozen_dir}\nCheckpoint: {ck_path}\n"
          f"Report: {resolve_path(cfg['outputs']['report'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
