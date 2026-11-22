"""Session 4 entry point: recurrent architecture family (LSTM, GRU).

    python -m src.run_session4 --config configs/session4_recurrent.yaml

Stages: frozen Session 1 windows -> reference stability gate -> Stage A ablation (3 hidden sizes x
2 depths x 2 dropouts per family, L=30) -> validation-only selection -> Stage B (the winner at the
other window lengths, compared on common validation endpoints) -> final selection -> the two
selected models through the Session 2 engine (MLflow, results table; the test set is evaluated
here for the first time) -> frozen definitions, checkpoints, curves, plots, report.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .benchmark_config import PROJECT_ROOT, resolve_path
from .environment import capture_environment
from .experiment import run_experiment
from .recurrent_models import RecurrentRegressor, resolve_device
from .recurrent_study import (GRID_KEYS, make_model, reference_run, run_candidate, select_sequence_length,
                              select_stage_a, stage_a, stage_a_grid)
from .sequence_data import load_sequence_data, window_counts
from .session3_config import to_benchmark_config
from .session3_outputs import session1_identity, write_csv, write_json
from .session4_config import load_session4_config
from .session4_outputs import (file_sha256, freeze_checkpoint, markdown_table, plot_prediction_vs_target,
                               plot_training_curves)
from .tracking import MlflowTracker

ABLATION_COLUMNS = ["stage", "family", "hidden_size", "num_layers", "dropout", "sequence_length", "candidate",
                    "status", "stop_reason", "best_epoch", "epochs_run", "n_parameters", "val_rmse", "val_mae",
                    "val_r2", "val_nasa_score", "best_val_loss", "train_time_s", "mean_epoch_time_s",
                    "max_grad_norm", "mean_clip_fraction", "latency_ms_per_batch", "throughput_samples_per_s",
                    "model_size_bytes", "weights_sha256", "mlflow_run_id"]


def _row(rec):
    out = {k: rec.get(k, "") for k in ABLATION_COLUMNS}
    out.update({f"val_{k}": v for k, v in rec["validation"].items()})
    out["candidate"] = rec.get("candidate", "")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Session 4: LSTM and GRU")
    parser.add_argument("--config", default="configs/session4_recurrent.yaml")
    args = parser.parse_args(argv)
    try:
        return _run(args.config)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _run(config_path):
    cfg, text, path = load_session4_config(config_path)
    bench_cfg = to_benchmark_config(cfg, text, path)
    timing_cfg, s1, seed = bench_cfg["timing"], cfg["session1"], int(cfg["benchmark"]["seed"])
    device = resolve_device(cfg["execution"]["device"])
    scale = float(cfg["training"]["target_scale"])
    work_dir, frozen_dir = resolve_path(cfg["outputs"]["working_dir"]), resolve_path(cfg["outputs"]["frozen_dir"])

    environment = capture_environment(
        PROJECT_ROOT, [*bench_cfg["environment"]["library_names"], "torch"],
        {"session4_config": bench_cfg.ref, "session2_config": cfg["session2"]["config"]})
    environment["torch"] = {"version": torch.__version__, "device": device,
                            "mps_available": bool(torch.backends.mps.is_available()),
                            "cuda_available": bool(torch.cuda.is_available()), "threads": torch.get_num_threads(),
                            "deterministic_algorithms_requested": device != "mps"}
    tracker = MlflowTracker(bench_cfg["tracking"]["tracking_dir"], bench_cfg["tracking"]["experiment_name"])
    print(f"Device: {device} (preference '{cfg['execution']['device']}'), torch {torch.__version__}")

    # 1. data: canonical windows plus the declared alternative lengths (same split and scaler)
    lengths = sorted({int(cfg["ablation"]["stage_a"]["sequence_length"]),
                      *[int(v) for v in cfg["ablation"]["stage_b"]["sequence_lengths"]]})
    datasets = {L: load_sequence_data(s1["processed_dir"], s1["config"], cfg["benchmark"]["target_type"], L)
                for L in lengths}
    canonical_L = int(cfg["ablation"]["stage_a"]["sequence_length"])
    counts = {L: window_counts(d) for L, d in datasets.items()}
    for L in lengths:
        print(f"Windows L={L}: " + ", ".join(f"{k} {v}" for k, v in counts[L].items()))
    data = datasets[canonical_L]

    # 2. reference stability gate (the ablation does not start unless it passes)
    ref = reference_run(cfg, data)
    print(f"Reference {ref['family']}: loss {ref['first_train_loss']:.4f} -> {ref['last_train_loss']:.4f}, best val "
          f"{ref['best_val_loss']:.4f} at epoch {ref['best_epoch']}, predictions in [{ref['prediction_min']:.1f}, "
          f"{ref['prediction_max']:.1f}], max grad norm {ref['max_grad_norm']:.2f}, passed={ref['passed']}")
    write_json(work_dir / "reference_run.json", ref)
    if not ref["passed"]:
        failed = [k for k, v in ref["checks"].items() if not v]
        raise RuntimeError(f"reference run failed stability checks {failed}; ablation not started")

    # 3. Stage A: full grid at the canonical length
    print(f"\nStage A (L={canonical_L}): {len(stage_a_grid(cfg))} architectures per family")
    records = stage_a(cfg, timing_cfg, data, tracker, environment)
    chosen_a = select_stage_a(cfg, records)

    # 4. Stage B: the Stage A winner of each family at the other lengths, then the final choice
    other = [L for L in lengths if L != canonical_L]
    final, seq_rows, stage_b_records = {}, {}, {}
    for family, (winner, rule_a) in chosen_a.items():
        arch = {k: winner[k] for k in GRID_KEYS}
        cands = [winner]
        for L in other:
            rec = run_candidate(cfg, timing_cfg, family, arch, datasets[L], tracker, f"sequence_length_L{L}",
                                environment)
            rec["candidate"] = len(stage_a_grid(cfg)) + len(cands) - 1
            cands.append(rec)
            stage_b_records.setdefault(family, []).append(rec)
            print(f"  {rec['run_name']:<28} val RMSE {rec['validation']['rmse']:8.3f}  [{rec['status']}]")
        cands = sorted([c for c in cands if c["status"] == "ok"], key=lambda c: c["sequence_length"])
        picked, rule_b, rows = select_sequence_length(cfg, family, cands, datasets, max(lengths))
        final[family] = {"record": picked, "rule_stage_a": rule_a, "rule_stage_b": rule_b,
                         "stage_a_run_id": winner["mlflow_run_id"]}
        seq_rows[family] = rows
        print(f"{family}: Stage A winner {_arch_label(winner)}; final choice L={picked['sequence_length']}")

    # 5. selected models through the Session 2 engine (test is evaluated here, after selection)
    run_dir_root = resolve_path(bench_cfg["outputs"]["runs_dir"])
    selected = {}
    for family, info in final.items():
        rec = info["record"]
        d = datasets[rec["sequence_length"]]
        arch = {k: rec[k] for k in GRID_KEYS}
        model = make_model(cfg, family, arch, d, name=family)
        result = run_experiment(
            model, d, bench_cfg, tracker, environment, config_filename="session4_config.yaml",
            extra_columns=lambda _result, m=model, f=family, d=d, a=arch: {
                "model_family": f, "hidden_size": a["hidden_size"], "num_layers": a["num_layers"],
                "dropout": a["dropout"], "sequence_length": d.sequence_length,
                "best_epoch": m.fit_summary_["best_epoch"], "epochs_run": m.fit_summary_["epochs_run"],
                "device": device})
        model_dir = run_dir_root / result.run_id / "model"
        ckpt_path, ckpt_sha = freeze_checkpoint(model_dir, family, resolve_path(cfg["outputs"]["checkpoint_dir"]))
        reloaded = RecurrentRegressor.load(model_dir, name=family)
        reload_ok = bool(np.array_equal(reloaded.predict(d.X["validation"]), result.predictions["validation"]))
        selected[family] = {"model": model, "result": result, "data": d, "checkpoint": ckpt_path,
                            "checkpoint_sha256": ckpt_sha, "reload_ok": reload_ok,
                            "retrain_matches_ablation": model.weights_sha256() == rec.get("weights_sha256")}
        print(f"{family}: selected {_arch_label(arch)} L={d.sequence_length}; validation RMSE "
              f"{result.metrics['validation']['rmse']:.3f}, test RMSE {result.metrics['test']['rmse']:.3f}; "
              f"reload ok={reload_ok}, identical to ablation run={selected[family]['retrain_matches_ablation']}")

    # 6. artifacts
    git = environment["git"]
    s1_identity = session1_identity(s1, sensors=data.sensors, sequence_length=canonical_L, rul_cap=data.rul_cap,
                                    target_type=data.target_type, root=PROJECT_ROOT)
    s3_manifest = PROJECT_ROOT / "artifacts/session3/manifest.json"
    frozen = {}
    for family, sel in selected.items():
        m, res, rec = sel["model"], sel["result"], final[family]["record"]
        frozen[family] = {
            "schema_version": 1, "family": family, "name": family,
            "architecture": {"cell": family, "hidden_size": m.hidden_size, "num_layers": m.num_layers,
                             "dropout": m.dropout, "sequence_length": m.sequence_length,
                             "n_sensors": m.n_sensors, "head": "linear(hidden, 1) on the last time step",
                             "n_parameters": m.n_parameters()},
            "training": {**cfg["training"], "seed": seed, "device": device},
            "selection": {"data": "validation only", "metric": cfg["selection"]["metric"],
                          "tie_tolerance_rel": cfg["selection"]["tie_tolerance_rel"],
                          "tiebreak": "fewest parameters, then shorter sequence length, then grid order",
                          "stage_a_rule": final[family]["rule_stage_a"],
                          "stage_b_rule": final[family]["rule_stage_b"],
                          "stage_b_common_endpoint_rows": seq_rows[family]},
            "training_outcome": {"best_epoch": m.fit_summary_["best_epoch"], "epochs_run": m.fit_summary_["epochs_run"],
                                 "stop_reason": m.fit_summary_["stop_reason"],
                                 "best_val_loss": m.fit_summary_["best_val_loss"]},
            "validation_metrics": res.metrics["validation"],
            "checkpoint": {"path": _relative(sel["checkpoint"]),
                           "file_sha256": sel["checkpoint_sha256"], "weights_sha256": m.weights_sha256()},
            "retrain_matches_ablation_weights": sel["retrain_matches_ablation"],
        }
        write_json(frozen_dir / f"selected_{family}.json", frozen[family])
    manifest = {
        "schema_version": 1, "session": 4, "config_ref": bench_cfg.ref, "config_sha256": bench_cfg.sha256,
        "seed": seed, "target_type": data.target_type, "device": device, "torch_version": torch.__version__,
        "sequence_lengths_tested": lengths, "session1": s1_identity,
        "session3_manifest_sha256": file_sha256(s3_manifest) if s3_manifest.exists() else None,
        "files": {f"selected_{f}.json": file_sha256(frozen_dir / f"selected_{f}.json") for f in selected},
        "checkpoints": {f: frozen[f]["checkpoint"] for f in selected},
        "source_run_ids": {f: selected[f]["result"].run_id for f in selected},
        "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]}
    write_json(frozen_dir / "manifest.json", manifest)

    all_records = [r for fam in records.values() for r in fam] + [r for fam in stage_b_records.values() for r in fam]
    write_csv(work_dir / "ablation_runs.csv", [_row(r) for r in all_records], ABLATION_COLUMNS)
    write_json(work_dir / "selection.json", {f: {"stage_a": final[f]["rule_stage_a"], "stage_b": final[f]["rule_stage_b"],
                                                  "stage_b_common_endpoint_rows": seq_rows[f]} for f in final})
    write_json(work_dir / "window_counts.json", {str(L): c for L, c in counts.items()})
    for family, sel in selected.items():
        write_json(work_dir / f"history_{family}.json", sel["model"].history_)

    fig_dir = resolve_path(cfg["outputs"]["figures_dir"])
    plot_training_curves(fig_dir / "fig8_session4_training_curves.png",
                         {f: s["model"].history_ for f, s in selected.items()}, scale)
    plot_prediction_vs_target(fig_dir / "fig9_session4_validation_prediction_vs_target.png",
                              [(f.upper(), s["data"].y["validation"], s["result"].predictions["validation"],
                                s["result"].metrics["validation"]["rmse"]) for f, s in selected.items()],
                              data.rul_cap)
    _write_report(resolve_path(cfg["outputs"]["report"]), cfg, device, counts, ref, records, stage_b_records,
                  final, seq_rows, selected, bench_cfg.ref, seed)

    print("\nFinal models (Session 2 engine):")
    for f, s in selected.items():
        r = s["result"]
        for es in ("validation", "test"):
            m = r.metrics[es]
            print(f"  {f:<5} {es:<10} RMSE {m['rmse']:7.3f}  MAE {m['mae']:7.3f}  R2 {m['r2']:.4f}  NASA {m['nasa_score']:10.1f}")
    print("Run IDs: " + ", ".join(f"{f}={s['result'].run_id}" for f, s in selected.items()))
    print(f"Frozen artifacts: {frozen_dir}\nCheckpoints: {resolve_path(cfg['outputs']['checkpoint_dir'])}\n"
          f"Report: {resolve_path(cfg['outputs']['report'])}")
    return 0


def _relative(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _arch_label(a):
    return f"hidden={a['hidden_size']} layers={a['num_layers']} dropout={a['dropout']}"


def _write_report(path, cfg, device, counts, ref, records, stage_b, final, seq_rows, selected, config_ref, seed):
    tr = cfg["training"]
    lines = ["# Session 4: recurrent architectures (LSTM, GRU)", "",
             f"Config `{config_ref}`, seed {seed}, target `{cfg['benchmark']['target_type']}`, device `{device}`. "
             "Architectures were selected on validation data only; the test set was evaluated once, for the two "
             "selected models.", "",
             "## 1. Architecture", "",
             "Input `[batch, time, 14 sensors]` (Session 1 windows, no tabular features) -> LSTM or GRU "
             "(`batch_first=True`, stacked, inter-layer dropout) -> hidden state of the last time step of the "
             "top layer -> dropout -> `Linear(hidden, 1)` -> RUL. No attention.", "",
             "## 2. Training protocol", "",
             f"Adam (lr {tr['learning_rate']}), MSE loss on RUL / {tr['target_scale']}, batch {tr['batch_size']}, "
             f"max {tr['max_epochs']} epochs, gradient-norm clipping at {tr['grad_clip']}, ReduceLROnPlateau "
             f"(factor {tr['scheduler']['factor']}, patience {tr['scheduler']['patience']}) and early stopping "
             f"(patience {tr['early_stopping_patience']}, min delta {tr['min_delta']}) both on the validation loss; "
             "the best validation checkpoint is restored. One seed per configuration. CPU runs are repeatable; MPS/CUDA "
             "kernels are not guaranteed to be bit-reproducible.", "",
             "## 3. Reference run (stability gate)", "",
             f"{ref['family'].upper()} {_arch_label(ref['architecture'])}: training loss {ref['first_train_loss']:.4f} -> "
             f"{ref['last_train_loss']:.4f}, best validation loss {ref['best_val_loss']:.4f} (epoch {ref['best_epoch']}), "
             f"validation predictions in [{ref['prediction_min']:.1f}, {ref['prediction_max']:.1f}], max gradient norm "
             f"{ref['max_grad_norm']:.2f}, mean fraction of clipped steps {ref['mean_clip_fraction']:.2f}. Tiny-batch "
             f"overfit: loss {ref['tiny_batch']['first_loss']:.4f} -> {ref['tiny_batch']['last_loss']:.5f}. "
             f"All checks passed: {ref['passed']}.", "",
             "## 4. Windows per sequence length", "",
             markdown_table([{"L": L, **c} for L, c in counts.items()], ["L", "train", "validation", "test"]), "",
             "Alternative lengths reuse the Session 1 split and scaler (windows only are regenerated). L=50 is not "
             "possible because some FD001 test engines have only 31 cycles and Session 1 does not pad.", ""]
    cols = ["hidden_size", "num_layers", "dropout", "val_rmse", "val_mae", "val_r2", "val_nasa", "n_parameters",
            "best_epoch", "epochs_run", "train_time_s", "latency_ms", "status", "chosen"]
    for family, recs in records.items():
        chosen = final[family]["rule_stage_a"]
        rows = []
        for r in recs:
            rows.append({"hidden_size": r["hidden_size"], "num_layers": r["num_layers"], "dropout": r["dropout"],
                         "val_rmse": r["validation"]["rmse"], "val_mae": r["validation"]["mae"],
                         "val_r2": r["validation"]["r2"], "val_nasa": r["validation"]["nasa_score"],
                         "n_parameters": r["n_parameters"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                         "train_time_s": r["train_time_s"], "latency_ms": r.get("latency_ms_per_batch", float("nan")),
                         "status": r["status"],
                         "chosen": "yes" if r["mlflow_run_id"] == final[family]["stage_a_run_id"] else ""})
        lines += [f"## 5. {family.upper()} ablation (L={cfg['ablation']['stage_a']['sequence_length']}, validation)", "",
                  markdown_table(rows, cols, {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f",
                                              "train_time_s": ".1f", "latency_ms": ".2f"}), "",
                  f"Stage A rule: lowest validation {chosen['metric']}; candidates within {chosen['tie_tolerance_rel']:.0%} "
                  f"of the best ({chosen['best_validation_value']:.3f}, {chosen['n_equivalent']} equivalent) -> fewest "
                  "parameters.", ""]
    lines += ["## 6. Sequence length (common validation endpoints)", ""]
    for family, rows in seq_rows.items():
        lines += [f"{family.upper()}:", "", markdown_table(rows, ["sequence_length", "n_common_windows", "rmse", "mae", "r2", "nasa_score"],
                                                         {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa_score": ".0f"}), ""]
    lines += ["## 7. Selected models", ""]
    comp = []
    for f, s in selected.items():
        m, r = s["model"], s["result"]
        for es in ("validation", "test"):
            comp.append({"model": f, "architecture": f"{_arch_label(m.get_config())}, L={m.sequence_length}",
                         "eval_set": es, "rmse": r.metrics[es]["rmse"], "mae": r.metrics[es]["mae"],
                         "r2": r.metrics[es]["r2"], "nasa": r.metrics[es]["nasa_score"]})
    lines += [markdown_table(comp, ["model", "architecture", "eval_set", "rmse", "mae", "r2", "nasa"],
                             {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa": ".1f"}), "",
              "NASA score is a sum, so validation (thousands of windows) and test (100) are not comparable; targets are "
              "capped at 125 on every split.", "",
              "Efficiency (timings vary between runs):", "",
              markdown_table([{"model": f, "parameters": s["model"].n_parameters(), "best_epoch": s["model"].fit_summary_["best_epoch"],
                               "epochs_run": s["model"].fit_summary_["epochs_run"], "train_time_s": s["result"].train_time_s,
                               "latency_ms_batch": s["result"].timing["latency_ms_per_batch"],
                               "throughput": s["result"].timing["throughput_samples_per_s"],
                               "size_bytes": s["result"].model_size_bytes} for f, s in selected.items()],
                             ["model", "parameters", "best_epoch", "epochs_run", "train_time_s", "latency_ms_batch",
                              "throughput", "size_bytes"], {"train_time_s": ".1f", "latency_ms_batch": ".2f", "throughput": ".0f"}), "",
              "## 8. Figures", "", "![training curves](figures/fig8_session4_training_curves.png)", "",
              "![validation prediction vs target](figures/fig9_session4_validation_prediction_vs_target.png)", "",
              "## 9. Unstable or failed runs", ""]
    bad = [r for fam in records.values() for r in fam] + [r for fam in stage_b.values() for r in fam]
    bad = [r for r in bad if r["status"] != "ok"]
    lines += [("None: every run trained without non-finite loss or gradient values." if not bad else
               markdown_table([{"run": r["run_name"], "stop_reason": r["stop_reason"]} for r in bad], ["run", "stop_reason"])), ""]
    lines += ["## 10. Reproducibility", "",
              "Retraining the selected configuration through the Session 2 engine gave weights identical to the "
              "ablation run: " + ", ".join(f"{f}={s['retrain_matches_ablation']}" for f, s in selected.items()) +
              ". Checkpoints reload to identical predictions: " +
              ", ".join(f"{f}={s['reload_ok']}" for f, s in selected.items()) + ".", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
