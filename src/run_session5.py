"""Session 5 entry point: temporal CNN and TCN architecture family.

    python -m src.run_session5 --config configs/session5_convolutional.yaml

Stages: frozen Session 1 windows -> stability gate -> 8 CNN + 8 TCN candidates (validation only) ->
representative CNN and TCN (validation-only rule) -> final convolutional architecture (same rule)
-> the final architecture through the Session 2 engine (MLflow, results table; the test set is
evaluated here, after the choice is frozen) -> frozen definitions, checkpoints, figures, report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .benchmark_config import PROJECT_ROOT, resolve_path
from .convolutional_models import ConvolutionalRegressor, cnn_receptive_field, tcn_receptive_field
from .convolutional_study import (candidate_grid, load_historical_representatives, make_model, run_candidate,
                                  select_candidate, threshold_near_miss, tiny_batch_gate)
from .environment import capture_environment
from .experiment import run_experiment
from .recurrent_models import resolve_device
from .sequence_data import load_sequence_data
from .session3_config import to_benchmark_config
from .session3_outputs import session1_identity, write_csv, write_json
from .session5_config import check_protocol_matches_reference, load_session5_config
from .session5_outputs import (file_sha256, freeze_checkpoint, latency_ratio_notes, plot_prediction_vs_target,
                               plot_rmse_vs_latency, plot_training_curves, write_report)
from .tracking import MlflowTracker

STUDY_COLUMNS = ["family", "candidate", "channels", "kernel_size", "depth", "dilations", "receptive_field",
                 "n_parameters", "status", "stop_reason", "best_epoch", "epochs_run", "val_rmse", "val_mae", "val_r2",
                 "val_nasa_score", "weights_sha256", "role"]
RUN_COLUMNS = [*STUDY_COLUMNS[:-1], "train_time_s", "mean_epoch_time_s", "max_grad_norm", "mean_clip_fraction",
               "latency_ms_per_batch", "latency_ms_per_sample", "throughput_samples_per_s", "model_size_bytes",
               "mlflow_run_id", "role"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Session 5: CNN and TCN")
    parser.add_argument("--config", default="configs/session5_convolutional.yaml")
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


def _arch_text(kind, rec):
    tail = f"depth {rec['depth']}" if kind == "cnn" else f"dilations {rec['dilations']}"
    return f"channels {rec['channels']}, kernel {rec['kernel_size']}, {tail}"


def _pick(rows, columns):
    return [{k: row.get(k, "") for k in columns} for row in rows]


def _depth_or_dil(rec):
    return rec["depth"] if rec["family"] == "cnn" else "[" + ", ".join(str(d) for d in rec["dilations"]) + "]"


def _flat(rec, role=""):
    v = rec["validation"]
    return {"family": rec["family"], "candidate": rec["candidate"], "channels": rec["channels"],
            "kernel_size": rec["kernel_size"], "depth": rec.get("depth", ""),
            "dilations": rec.get("dilations_label", ""), "receptive_field": rec["receptive_field"],
            "n_parameters": rec["n_parameters"], "status": rec["status"], "stop_reason": rec["stop_reason"],
            "best_epoch": rec["best_epoch"], "epochs_run": rec["epochs_run"], "val_rmse": repr(v["rmse"]),
            "val_mae": repr(v["mae"]), "val_r2": repr(v["r2"]), "val_nasa_score": repr(v["nasa_score"]),
            "weights_sha256": rec.get("weights_sha256", ""), "role": role,
            **{k: rec.get(k, "") for k in ("train_time_s", "mean_epoch_time_s", "max_grad_norm",
                                           "mean_clip_fraction", "latency_ms_per_batch", "latency_ms_per_sample",
                                           "throughput_samples_per_s", "model_size_bytes", "mlflow_run_id")}}


def _run(config_path):
    cfg, text, path = load_session5_config(config_path)
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
          f"{data.X['validation'].shape}, test {data.X['test'].shape}")

    environment = capture_environment(PROJECT_ROOT, [*bench_cfg["environment"]["library_names"], "torch"],
                                      {"session5_config": bench_cfg.ref, "session2_config": cfg["session2"]["config"]})
    environment["torch"] = {"version": torch.__version__, "device": device, "threads": torch.get_num_threads(),
                            "mps_available": bool(torch.backends.mps.is_available()),
                            "cuda_available": bool(torch.cuda.is_available())}
    tracker = MlflowTracker(bench_cfg["tracking"]["tracking_dir"], bench_cfg["tracking"]["experiment_name"])

    # 1. stability gate: each family must overfit a tiny batch before the study starts
    grids = {k: candidate_grid(cfg, k) for k in ("cnn", "tcn")}
    g = cfg.get("stability_gate", {})
    gate = [tiny_batch_gate(cfg, k, grids[k][0], data, n=int(g.get("samples", 64)), epochs=int(g.get("epochs", 300)),
                            lr=float(g.get("learning_rate", 0.01)), min_drop=float(g.get("min_loss_drop", 0.5)))
            for k in ("cnn", "tcn")]
    for r in gate:
        print(f"Gate {r['family']}: tiny-batch loss {r['first_loss']:.4f} -> {r['last_loss']:.5f}, passed={r['passed']}")
    if not all(r["passed"] for r in gate):
        raise RuntimeError("stability gate failed for " + ", ".join(r["family"] for r in gate if not r["passed"]))

    # 2. candidates (validation only)
    records, cand_dir = {}, work_dir / "candidates"
    for kind in ("cnn", "tcn"):
        print(f"\n{kind.upper()}: {len(grids[kind])} candidates (L={L})")
        records[kind] = []
        for i, arch in enumerate(grids[kind]):
            rec = run_candidate(cfg, timing_cfg, kind, arch, data, tracker, f"ablation_{kind}", environment, i, cand_dir)
            records[kind].append(rec)
            print(f"  {rec['run_name']:<24} val RMSE {rec['validation']['rmse']:8.3f}  best epoch {rec['best_epoch']:>2}/"
                  f"{rec['epochs_run']:<2} params {rec['n_parameters']:>7}  RF {rec['receptive_field']:>3}  [{rec['status']}]")

    # 3. representatives per family, then the final convolutional architecture (same rule)
    sel = cfg["selection"]
    reps, family_rule = {}, {}
    for kind in ("cnn", "tcn"):
        reps[kind], family_rule[kind] = select_candidate(records[kind], sel["metric"], float(sel["tie_tolerance_rel"]))
        print(f"{kind.upper()} representative: {reps[kind]['run_name']} (decided by {family_rule[kind]['decided_by']})")
    ordered = [{**reps["cnn"], "candidate": 0}, {**reps["tcn"], "candidate": 1}]
    final_rec, final_rule = select_candidate(ordered, sel["metric"], float(sel["tie_tolerance_rel"]))
    final_kind = final_rec["family"]
    print(f"Final convolutional architecture: {final_kind.upper()} {final_rec['run_name']} "
          f"(decided by {final_rule['decided_by']})")

    # 4. the final architecture through the Session 2 engine (test evaluated after the choice)
    arch = {k: final_rec[k] for k in ("channels", "kernel_size", "depth", "dilations") if k in final_rec}
    model = make_model(cfg, final_kind, arch, data, name=final_kind)
    result = run_experiment(
        model, data, bench_cfg, tracker, environment, config_filename="session5_config.yaml",
        extra_columns=lambda _r, m=model: {
            "model_family": m.kind, "channels": m.channels, "kernel_size": m.kernel_size,
            "depth": m.depth if m.depth is not None else "",
            "dilations": "-".join(map(str, m.dilations)) if m.dilations else "",
            "receptive_field": m.receptive_field, "sequence_length": m.sequence_length,
            "best_epoch": m.fit_summary_["best_epoch"], "epochs_run": m.fit_summary_["epochs_run"],
            "device": device})
    run_model_dir = resolve_path(bench_cfg["outputs"]["runs_dir"]) / result.run_id / "model"
    reloaded = ConvolutionalRegressor.load(run_model_dir, name=final_kind)
    reload_ok = bool(np.array_equal(reloaded.predict(data.X["validation"]), result.predictions["validation"]))
    retrain_matches = model.weights_sha256() == final_rec["weights_sha256"]
    print(f"Final {final_kind.upper()}: validation RMSE {result.metrics['validation']['rmse']:.3f}, test RMSE "
          f"{result.metrics['test']['rmse']:.3f}; reload ok={reload_ok}, identical to candidate run={retrain_matches}")

    # 5. frozen definitions and checkpoints
    checkpoints = {}
    for kind in ("cnn", "tcn"):
        source = run_model_dir if kind == final_kind else cand_dir / reps[kind]["run_name"]
        ck_path, ck_sha = freeze_checkpoint(source, kind, ckpt_dir)
        checkpoints[kind] = {"path": _relative(ck_path), "file_sha256": ck_sha,
                             "weights_sha256": reps[kind]["weights_sha256"]}
    git = environment["git"]
    rf_rows = []
    for kind, tails in (("cnn", cfg["architecture"]["cnn"]["depth"]), ("tcn", cfg["architecture"]["tcn"]["dilations"])):
        for k in cfg["architecture"][kind]["kernel_size"]:
            for t in tails:
                rf = cnn_receptive_field(k, t) if kind == "cnn" else tcn_receptive_field(k, t)
                rf_rows.append({"family": kind, "kernel_size": k, "depth_or_dilations": t, "receptive_field": rf,
                                "vs_window": "exceeds window (extra positions are zero padding)" if rf > L
                                else "within window"})
    write_json(frozen_dir / "receptive_field.json", {"sequence_length": L, "formulas": {
        "tcn": "1 + 2 * (kernel_size - 1) * sum(dilations)", "cnn": "1 + depth * (kernel_size - 1)"},
        "configurations": rf_rows})
    roles = {reps["cnn"]["run_name"]: "representative_cnn", reps["tcn"]["run_name"]: "representative_tcn"}
    roles[final_rec["run_name"]] = roles[final_rec["run_name"]] + "+final"
    all_records = [r for k in ("cnn", "tcn") for r in records[k]]
    write_csv(frozen_dir / "architecture_study.csv", _pick([_flat(r, roles.get(r["run_name"], ""))
                                                             for r in all_records], STUDY_COLUMNS), STUDY_COLUMNS)
    for kind in ("cnn", "tcn"):
        r = reps[kind]
        write_json(frozen_dir / f"selected_{kind}.json", {
            "schema_version": 1, "family": kind,
            "architecture": {**{k: r[k] for k in ("channels", "kernel_size") + (("depth",) if kind == "cnn" else ("dilations",))},
                             "dropout": cfg["architecture"]["dropout"], "sequence_length": L,
                             "n_sensors": len(data.sensors), "receptive_field": r["receptive_field"],
                             "readout": "last time step", "n_parameters": r["n_parameters"]},
            "training": {**cfg["training"], "seed": seed, "device": device},
            "selection": {"data": "validation only", **family_rule[kind]},
            "training_outcome": {"best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                                 "stop_reason": r["stop_reason"], "best_val_loss": r["best_val_loss"]},
            "validation_metrics": r["validation"], "checkpoint": checkpoints[kind]})
    write_json(frozen_dir / "selected_convolutional.json", {
        "schema_version": 1, "final_family": final_kind, "definition_file": f"selected_{final_kind}.json",
        "architecture": {"channels": final_rec["channels"], "kernel_size": final_rec["kernel_size"],
                         "receptive_field": final_rec["receptive_field"], "n_parameters": final_rec["n_parameters"],
                         "sequence_length": L, "n_sensors": len(data.sensors),
                         **({"depth": final_rec["depth"]} if final_kind == "cnn" else {"dilations": final_rec["dilations"]})},
        "selection": {"data": "validation only", **final_rule},
        "representatives_validation": {k: {"run_name": reps[k]["run_name"], "n_parameters": reps[k]["n_parameters"],
                                           **reps[k]["validation"]} for k in ("cnn", "tcn")},
        "checkpoint": checkpoints[final_kind], "retrain_matches_candidate_weights": retrain_matches,
        "intended_use": "representative convolutional architecture for Session 7"})
    s1_id = session1_identity(s1, sensors=data.sensors, sequence_length=L, rul_cap=data.rul_cap,
                              target_type=data.target_type, root=PROJECT_ROOT)
    pin = lambda rel: file_sha256(PROJECT_ROOT / rel) if (PROJECT_ROOT / rel).exists() else None
    names = ["receptive_field.json", "architecture_study.csv", "selected_cnn.json", "selected_tcn.json",
             "selected_convolutional.json"]
    write_json(frozen_dir / "manifest.json", {
        "schema_version": 1, "session": 5, "config_ref": bench_cfg.ref, "config_sha256": bench_cfg.sha256,
        "protocol_reference": cfg["protocol_reference"]["config"],
        "protocol_reference_sha256": pin(cfg["protocol_reference"]["config"]), "seed": seed,
        "target_type": data.target_type, "device": device, "torch_version": torch.__version__,
        "sequence_length": L, "n_sensors": len(data.sensors), "session1": s1_id,
        "session3_manifest_sha256": pin("artifacts/session3/manifest.json"),
        "session4_manifest_sha256": pin("artifacts/session4/manifest.json"),
        "final_family": final_kind, "final_architecture": final_rec["run_name"],
        "final_n_parameters": final_rec["n_parameters"], "final_receptive_field": final_rec["receptive_field"],
        "files": {n: file_sha256(frozen_dir / n) for n in names}, "checkpoints": checkpoints,
        "final_run_id": result.run_id,
        "candidate_run_ids": {k: reps[k]["mlflow_run_id"] for k in ("cnn", "tcn")},
        "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]})
    write_csv(work_dir / "architecture_runs.csv", _pick([_flat(r, roles.get(r["run_name"], "")) for r in all_records],
                                                         RUN_COLUMNS), RUN_COLUMNS)
    write_json(work_dir / "selection.json", {"cnn": family_rule["cnn"], "tcn": family_rule["tcn"], "final": final_rule,
                                              "final_family": final_kind})
    for kind in ("cnn", "tcn"):
        write_json(work_dir / f"history_{kind}.json", reps[kind]["history"])
    write_json(work_dir / "gate.json", gate)

    # 6. comparison with the frozen earlier families (read-only), figures and report
    hist = load_historical_representatives(resolve_path(bench_cfg["outputs"]["results_json"]))
    family_of = {"random_forest": "classical", "gradient_boosting": "classical", "lstm": "recurrent", "gru": "recurrent"}
    comparison, points = [], []
    for model_name, by_set in hist.items():
        v, t = by_set["validation"], by_set.get("test")
        comparison.append({"family": family_of[model_name], "model": model_name,
                           "n_parameters": v.get("n_parameters", ""), "val_rmse": v["rmse"], "val_mae": v["mae"],
                           "val_r2": v["r2"], "val_nasa": v["nasa_score"],
                           "latency_ms": v["inference_latency_ms_per_batch"], "train_time_s": v["train_time_s"],
                           "test_rmse": f"{t['rmse']:.3f}" if t else "n/a"})
        points.append({"label": model_name, "family": family_of[model_name], "latency": v["inference_latency_ms_per_batch"],
                       "rmse": v["rmse"], "n_parameters": int(v.get("n_parameters") or 0)})
    for kind in ("cnn", "tcn"):
        r = reps[kind]
        comparison.append({"family": "convolutional", "model": f"{kind} ({r['run_name']})", "n_parameters": r["n_parameters"],
                           "val_rmse": r["validation"]["rmse"], "val_mae": r["validation"]["mae"],
                           "val_r2": r["validation"]["r2"], "val_nasa": r["validation"]["nasa_score"],
                           "latency_ms": r["latency_ms_per_batch"], "train_time_s": r["train_time_s"],
                           "test_rmse": f"{result.metrics['test']['rmse']:.3f}" if kind == final_kind else "n/a"})
        points.append({"label": kind, "family": "convolutional", "latency": r["latency_ms_per_batch"],
                       "rmse": r["validation"]["rmse"], "n_parameters": r["n_parameters"]})
    fig_dir = resolve_path(cfg["outputs"]["figures_dir"])
    figures = ["fig10_session5_training_curves.png", "fig11_session5_validation_prediction_vs_target.png",
               "fig12_session5_validation_rmse_vs_latency.png"]
    plot_training_curves(fig_dir / figures[0], {k: reps[k]["history"] for k in ("cnn", "tcn")}, scale)
    plot_prediction_vs_target(fig_dir / figures[1], [(final_kind.upper(), data.y["validation"],
                                                     result.predictions["validation"],
                                                     result.metrics["validation"]["rmse"])], data.rul_cap)
    plot_rmse_vs_latency(fig_dir / figures[2], points)

    def table(kind):
        return [{"channels": r["channels"], "kernel_size": r["kernel_size"], "depth_or_dilations": _depth_or_dil(r),
                 "receptive_field": r["receptive_field"], "n_parameters": r["n_parameters"],
                 "val_rmse": r["validation"]["rmse"], "val_mae": r["validation"]["mae"], "val_r2": r["validation"]["r2"],
                 "val_nasa": r["validation"]["nasa_score"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                 "train_time_s": r["train_time_s"], "latency_ms": r.get("latency_ms_per_batch", float("nan")),
                 "status": r["status"], "chosen": "yes" if r["run_name"] == reps[kind]["run_name"] else ""}
                for r in records[kind]]
    rep_rows = [{"model": k, "architecture": _arch_text(k, reps[k]), "receptive_field": reps[k]["receptive_field"],
                 "n_parameters": reps[k]["n_parameters"], "val_rmse": reps[k]["validation"]["rmse"],
                 "val_mae": reps[k]["validation"]["mae"], "val_r2": reps[k]["validation"]["r2"],
                 "val_nasa": reps[k]["validation"]["nasa_score"], "best_epoch": reps[k]["best_epoch"],
                 "epochs_run": reps[k]["epochs_run"], "train_time_s": reps[k]["train_time_s"],
                 "latency_ms": reps[k]["latency_ms_per_batch"], "throughput": reps[k]["throughput_samples_per_s"]}
                for k in ("cnn", "tcn")]
    repro = (f"On CPU, retraining the final architecture through the Session 2 engine gave weights identical to the "
             f"candidate run: {retrain_matches}; the checkpoint reloads to identical predictions: {reload_ok}. MPS and CUDA "
             "kernels are not guaranteed to be bit-reproducible.")
    write_report(resolve_path(cfg["outputs"]["report"]), cfg=cfg, device=device, config_ref=bench_cfg.ref, seed=seed,
                 gate=gate, tables={k: table(k) for k in ("cnn", "tcn")}, family_rule=family_rule, reps=rep_rows,
                 final={"family": final_kind, "architecture": _arch_text(final_kind, final_rec), "rule": final_rule,
                        "metrics": [{"eval_set": es, "n_samples": result.n_samples[es], **result.metrics[es]}
                                    for es in ("validation", "test")]},
                 comparison=comparison, unstable=[r for r in all_records if r["status"] != "ok"],
                 reproducibility=repro, figures=figures, receptive_fields=rf_rows,
                 near_miss={k: threshold_near_miss(records[k], reps[k], family_rule[k]) for k in ("cnn", "tcn")},
                 latency_notes=latency_ratio_notes(comparison, final_kind))

    print("\nFinal convolutional model (Session 2 engine):")
    for es in ("validation", "test"):
        m = result.metrics[es]
        print(f"  {final_kind:<4} {es:<10} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  R2 {m['r2']:.4f}  NASA {m['nasa_score']:10.1f}")
    print(f"Run ID: {final_kind}={result.run_id}\nFrozen artifacts: {frozen_dir}\nCheckpoints: {ckpt_dir}\n"
          f"Report: {resolve_path(cfg['outputs']['report'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
