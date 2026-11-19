"""Session 3 entry point: classical ML benchmark and leakage-safe sensor rankings.

    python -m src.run_session3 --config configs/session3_classical_ml.yaml

Steps: load frozen Session 1 windows -> window-to-tabular features -> validation-only
hyperparameter search for random forest and gradient boosting -> evaluate the SELECTED models
through the Session 2 engine (MLflow, results table) -> sensor ranking A (training-only
correlation) -> ranking B (validation permutation importance) -> comparison -> frozen top-k and
random-k subsets -> report.

The official test set is only passed to the Session 2 engine for the final evaluation of the
already-selected models. Search, ranking and subset functions never receive it.
"""
import argparse
import sys
from pathlib import Path

from .benchmark_config import PROJECT_ROOT, resolve_path
from .benchmark_data import load_benchmark_data
from .classical_models import FAMILIES, build_model
from .environment import capture_environment
from .experiment import run_experiment
from .session3_config import load_session3_config, to_benchmark_config
from .session3_outputs import (markdown_table, selected_models_document, session1_identity, write_csv,
                               write_frozen_artifacts, write_json, write_manifest)
from .sensor_ranking import (compare_rankings, correlation_ranking, permutation_ranking,
                             random_k_subsets, top_k_subsets)
from .tabular_features import WindowFeatureTransformer
from .tracking import MlflowTracker
from .tuning import expand_grid, search, select


def tune_family(family, model_cfg, cfg, transformer, F_train, y_train, F_val, y_val):
    """Validation-only search. Receives training and validation arrays only (no test data)."""
    seed, ex = int(cfg["benchmark"]["seed"]), cfg["execution"]
    fixed = dict(model_cfg.get("fixed") or {})

    def build(params):
        return build_model(family, {**fixed, **params}, sensors=transformer.sensors,
                           features=transformer.features, sequence_length=transformer.sequence_length,
                           seed=seed, n_jobs=ex["n_jobs_train"], predict_n_jobs=ex["n_jobs_predict"])

    records = search(build, model_cfg["search_space"], F_train, y_train, F_val, y_val)
    chosen, rule = select(records, cfg["selection"]["metric"], float(cfg["selection"]["tie_tolerance_rel"]))
    return records, chosen, {**rule, "fixed_params": fixed}


def sensor_intelligence(cfg, transformer, F_train, y_train, F_val, y_val, source_model):
    """Rankings A and B, their comparison and the frozen subsets. Training/validation only."""
    rk, budgets = cfg["ranking"], cfg["budgets"]["sensor_counts"]
    ranking_a = correlation_ranking(F_train, y_train, transformer, rk["correlation"]["feature"])
    perm = rk["permutation"]
    ranking_b, perm_detail = permutation_ranking(source_model, F_val, y_val, transformer,
                                                 int(perm["n_repeats"]), int(perm["seed"]),
                                                 perm["aggregation"])
    comparison = compare_rankings(ranking_a, ranking_b, budgets, int(rk["top_disagreements"]))
    topk = {"correlation": top_k_subsets(ranking_a, budgets, transformer.sensors),
            "permutation": top_k_subsets(ranking_b, budgets, transformer.sensors)}
    randomk = random_k_subsets(transformer.sensors, budgets, cfg["budgets"]["random_control_seeds"])
    return ranking_a, ranking_b, perm_detail, comparison, topk, randomk


def _param_label(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def write_report(path, figures, **s):
    t = s["transformer"]
    lines = ["# Session 3: classical ML and sensor intelligence", "",
             f"Config `{s['config_ref']}`, seed {s['seed']}, target `{s['target_type']}`. "
             "Hyperparameters were selected on validation only; the test set was used once, for the "
             "selected models. Timing values are in the results table / MLflow, not here.", "",
             "## Features", "",
             f"{len(t.sensors)} sensors x {len(t.features)} temporal features "
             f"({', '.join(t.features)}) = {t.n_features} columns, sensor-major order "
             "(`<sensor>__<feature>`). Slope = OLS slope over cycle index 0..29, std = population std.", ""]
    for family, info in s["tuning"].items():
        lines += [f"## {family}: validation search ({len(info['records'])} candidates)", "",
                  markdown_table(
                      [{"candidate": r["candidate"], "params": _param_label(r["params"]),
                        "rmse": r["validation"]["rmse"], "mae": r["validation"]["mae"],
                        "r2": r["validation"]["r2"], "nasa": r["validation"]["nasa_score"],
                        "nodes": r["n_parameters"],
                        "selected": "yes" if r["candidate"] == info["chosen"]["candidate"] else ""}
                       for r in info["records"]],
                      ["candidate", "params", "rmse", "mae", "r2", "nasa", "nodes", "selected"],
                      {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa": ".0f"}), "",
                  f"Rule: lowest validation {info['rule']['metric']}; candidates within "
                  f"{info['rule']['tie_tolerance_rel']:.0%} of the best ({info['rule']['best_validation_value']:.3f}) "
                  f"are equivalent ({info['rule']['n_equivalent']} here); the smallest model is chosen.", ""]
    lines += ["## Selected full-sensor models", "",
              markdown_table(s["comparison_rows"],
                             ["model", "eval_set", "n_samples", "rmse", "mae", "r2", "nasa_score"],
                             {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa_score": ".1f"}), "",
              "NASA score is a sum, so validation (3272 windows) and test (100) are not comparable. "
              "Targets are capped at 125 on every split (see the Session 2 notes on comparability).", "",
              "## Ranking A: training-only correlation", "",
              f"Pearson r between the sensor's `{s['corr_feature']}` value and capped RUL over the training "
              "windows; ranked by |r|.", "",
              markdown_table(s["ranking_a"], ["rank", "sensor", "correlation", "abs_correlation"],
                             {"correlation": ".4f", "abs_correlation": ".4f"}), "",
              "## Ranking B: validation permutation importance", "",
              f"Model: `{s['perm_model']}`; {s['perm_repeats']} shuffles per feature; importance = "
              f"increase in validation RMSE; sensor score = {s['aggregation']} of its six feature "
              f"importances. Baseline validation RMSE {s['perm_baseline']:.3f}.", "",
              markdown_table(s["ranking_b_rows"], ["rank", "sensor", "importance", *t.features],
                             {"importance": ".3f", **{f: ".3f" for f in t.features}}), "",
              "## Ranking agreement", "",
              f"Spearman rho = {s['comparison']['spearman_rho']:.3f} (p = {s['comparison']['spearman_p_value']:.3g}). "
              "Neither ranking is declared correct.", "",
              "Largest rank disagreements:", "",
              markdown_table(s["comparison"]["largest_disagreements"],
                             ["sensor", "rank_correlation_method", "rank_permutation_method", "rank_difference"]), "",
              "Top-k overlap:", "",
              markdown_table([{"k": k.split("_")[1], "shared": len(v["shared"]),
                               "only correlation": ", ".join(v["only_correlation"]) or "-",
                               "only permutation": ", ".join(v["only_permutation"]) or "-",
                               "jaccard": v["jaccard"]} for k, v in s["comparison"]["top_k_overlap"].items()],
                             ["k", "shared", "only correlation", "only permutation", "jaccard"],
                             {"jaccard": ".2f"}), "",
              "## Frozen subsets", "",
              markdown_table([{"k": v["k"], "method": m, "sensors (by rank)": ", ".join(v["sensors_by_rank"])}
                              for m, sub in s["topk"].items() for v in sub.values()],
                             ["method", "k", "sensors (by rank)"]), "",
              f"Random-k controls: seeds {s['random_seeds']} per k, in `sensor_subsets_randomk.json`.", ""]
    if figures:
        lines += ["## Figures", ""] + [f"![{name}](figures/{name})" for name in figures] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def make_figures(figures_dir, tuning, ranking_a, ranking_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (family, info) in enumerate(tuning.items()):
        x = [r["validation"]["rmse"] for r in info["records"]]
        ax.scatter(x, [i] * len(x), alpha=0.6, label=f"{family} candidates")
        ax.scatter([info["chosen"]["validation"]["rmse"]], [i], marker="*", s=220, edgecolor="k",
                   label=f"{family} selected", zorder=3)
    ax.set_yticks(range(len(tuning)), list(tuning))
    ax.set_xlabel("validation RMSE (cycles)")
    ax.set_title("Hyperparameter candidates on validation")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_session3_validation_search.png", dpi=150)
    plt.close(fig)
    names.append("fig6_session3_validation_search.png")

    sensors = [r["sensor"] for r in ranking_a]
    rank_b = {r["sensor"]: r["rank"] for r in ranking_b}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.scatter([r["rank"] for r in ranking_a], [rank_b[s] for s in sensors])
    for r in ranking_a:
        ax.annotate(r["sensor"].replace("sensor_", "s"), (r["rank"], rank_b[r["sensor"]]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    top = len(sensors)
    ax.plot([1, top], [1, top], color="grey", lw=1, ls="--")
    ax.set_xlabel("rank A: training correlation")
    ax.set_ylabel("rank B: validation permutation")
    ax.set_title("Sensor rank agreement")
    ax = axes[1]
    ordered = sorted(ranking_b, key=lambda r: r["rank"])
    ax.barh([r["sensor"] for r in ordered][::-1], [r["importance"] for r in ordered][::-1])
    ax.set_xlabel("summed RMSE increase when shuffled (cycles)")
    ax.set_title("Permutation importance by sensor")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig7_session3_sensor_rankings.png", dpi=150)
    plt.close(fig)
    names.append("fig7_session3_sensor_rankings.png")
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Session 3: classical ML and sensor rankings")
    parser.add_argument("--config", default="configs/session3_classical_ml.yaml")
    args = parser.parse_args(argv)
    try:
        return _run(args.config)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _run(config_path):
    cfg, text, path = load_session3_config(config_path)
    bench_cfg = to_benchmark_config(cfg, text, path)
    s1 = cfg["session1"]
    data = load_benchmark_data(s1["processed_dir"], s1["config"], cfg["benchmark"]["target_type"])
    seed = int(cfg["benchmark"]["seed"])
    transformer = WindowFeatureTransformer(data.sensors, cfg["features"]["temporal"], data.sequence_length)

    F = {"train": transformer.transform(data.X["train"]),
         "validation": transformer.transform(data.X["validation"])}
    print(f"Features: train {F['train'].shape}, validation {F['validation'].shape}, "
          f"test {(len(data.X['test']), transformer.n_features)}")

    # 1. validation-only search (no test arrays are passed)
    tuning = {}
    for family in FAMILIES:
        records, chosen, rule = tune_family(family, cfg["models"][family], cfg, transformer,
                                            F["train"], data.y["train"], F["validation"], data.y["validation"])
        tuning[family] = {"records": records, "chosen": chosen, "rule": rule}
        print(f"{family}: {len(records)} candidates, selected #{chosen['candidate']} "
              f"{_param_label(chosen['params'])} (validation RMSE {chosen['validation']['rmse']:.3f})")

    # 2. final evaluation of the selected models through the Session 2 engine
    environment = capture_environment(
        PROJECT_ROOT, bench_cfg["environment"]["library_names"],
        {"session3_config": bench_cfg.ref, "session2_config": cfg["session2"]["config"]})
    tracker = MlflowTracker(bench_cfg["tracking"]["tracking_dir"], bench_cfg["tracking"]["experiment_name"])
    ex = cfg["execution"]
    results, fitted = {}, {}
    for family in FAMILIES:
        info = tuning[family]
        params = {**info["rule"]["fixed_params"], **info["chosen"]["params"]}
        model = build_model(family, params, sensors=data.sensors, features=transformer.features,
                            sequence_length=data.sequence_length, seed=seed,
                            name=cfg["models"][family]["name"], n_jobs=ex["n_jobs_train"],
                            predict_n_jobs=ex["n_jobs_predict"])
        result = run_experiment(
            model, data, bench_cfg, tracker, environment, config_filename="session3_config.yaml",
            extra_columns={"model_family": family,
                           "feature_set": f"temporal_{transformer.n_features}",
                           "selected_params": _param_label(params)})
        results[family], fitted[family] = result, model

    # 3. sensor rankings: training / validation data only
    source = cfg["ranking"]["permutation"]["source_model"]
    ranking_a, ranking_b, perm_detail, comparison, topk, randomk = sensor_intelligence(
        cfg, transformer, F["train"], data.y["train"], F["validation"], data.y["validation"], fitted[source])
    print(f"Ranking comparison: Spearman rho = {comparison['spearman_rho']:.3f}")

    # 4. write artifacts
    frozen_dir = resolve_path(cfg["outputs"]["frozen_dir"])
    work_dir = resolve_path(cfg["outputs"]["working_dir"])
    git = environment["git"]
    metadata = {"session": 3, "config_ref": bench_cfg.ref, "config_sha256": bench_cfg.sha256,
                "seed": seed, "target_type": data.target_type, "sensors": data.sensors,
                "temporal_features": list(transformer.features),
                "permutation_source_model": source,
                "random_control_seeds": cfg["budgets"]["random_control_seeds"],
                "sensor_counts": cfg["budgets"]["sensor_counts"]}
    comparison_doc = {**metadata, **comparison}
    topk_doc = {**metadata, "subsets": topk}
    randomk_doc = {**metadata, "subsets": randomk}
    write_frozen_artifacts(
        frozen_dir, ranking_a=ranking_a, ranking_b=ranking_b, comparison=comparison_doc,
        topk=topk_doc, randomk=randomk_doc, features=list(transformer.features))
    write_json(frozen_dir / "permutation_feature_importance.json", {**perm_detail, "n_repeats":
               int(cfg["ranking"]["permutation"]["n_repeats"]), "seed": int(cfg["ranking"]["permutation"]["seed"])})
    selected = {f: {"params": {**tuning[f]["rule"]["fixed_params"], **tuning[f]["chosen"]["params"]},
                    "candidate": tuning[f]["chosen"]["candidate"], "rule": tuning[f]["rule"],
                    "validation": tuning[f]["chosen"]["validation"],
                    "n_parameters": tuning[f]["chosen"]["n_parameters"]} for f in FAMILIES}
    write_json(frozen_dir / "selected_models.json",
               selected_models_document(selected, features=transformer.features, seed=seed, metadata=metadata))
    identity = session1_identity(s1, sensors=data.sensors, sequence_length=data.sequence_length,
                                 rul_cap=data.rul_cap, target_type=data.target_type, root=PROJECT_ROOT)
    write_manifest(frozen_dir, {**metadata, "session1": identity},
                   {"permutation_source_run_id": results[source].run_id,
                    "git_commit": git["git_commit"], "git_dirty": git["git_dirty"]})

    comparison_rows = []
    for family in FAMILIES:
        r = results[family]
        for eval_set in ("validation", "test"):
            comparison_rows.append({"model": r.model_name, "eval_set": eval_set, "run_id": r.run_id,
                                    "n_samples": r.n_samples[eval_set], **r.metrics[eval_set],
                                    "train_time_s": r.train_time_s,
                                    "latency_ms_per_batch": r.timing["latency_ms_per_batch"],
                                    "throughput_samples_per_s": r.timing["throughput_samples_per_s"],
                                    "model_size_bytes": r.model_size_bytes, "n_parameters": r.n_parameters})
    write_csv(work_dir / "model_comparison.csv", comparison_rows, list(comparison_rows[0]))
    for family, info in tuning.items():
        write_csv(work_dir / f"search_{family}.csv",
                  [{"candidate": r["candidate"], **r["params"], **{f"val_{k}": v for k, v in r["validation"].items()},
                    "n_parameters": r["n_parameters"], "fit_time_s": r["fit_time_s"],
                    "selected": r["candidate"] == info["chosen"]["candidate"]} for r in info["records"]],
                  ["candidate", *cfg["models"][family]["search_space"], "val_rmse", "val_mae", "val_r2",
                   "val_nasa_score", "n_parameters", "fit_time_s", "selected"])
    write_json(work_dir / "selection.json", {
        family: {"selected_params": {**info["rule"]["fixed_params"], **info["chosen"]["params"]},
                 "selected_candidate": info["chosen"]["candidate"], "rule": info["rule"],
                 "run_id": results[family].run_id} for family, info in tuning.items()})

    figures = make_figures(resolve_path(cfg["outputs"]["figures_dir"]), tuning, ranking_a, ranking_b)
    write_report(resolve_path(cfg["outputs"]["report"]), figures,
                 transformer=transformer, config_ref=bench_cfg.ref, seed=seed, target_type=data.target_type,
                 tuning=tuning, comparison_rows=comparison_rows, ranking_a=ranking_a,
                 ranking_b_rows=[{**r, **r["components"]} for r in ranking_b],
                 corr_feature=cfg["ranking"]["correlation"]["feature"], perm_model=source,
                 perm_repeats=int(cfg["ranking"]["permutation"]["n_repeats"]),
                 aggregation=cfg["ranking"]["permutation"]["aggregation"],
                 perm_baseline=perm_detail["baseline_validation_rmse"], comparison=comparison,
                 topk=topk, random_seeds=cfg["budgets"]["random_control_seeds"])

    # 5. summary
    print("\nSelected models (Session 2 engine):")
    for row in comparison_rows:
        print(f"  {row['model']:<18} {row['eval_set']:<10} RMSE {row['rmse']:7.3f}  MAE {row['mae']:7.3f}  "
              f"R2 {row['r2']:.4f}  NASA {row['nasa_score']:10.1f}")
    print("\nTop-k subsets (correlation | permutation):")
    for k in cfg["budgets"]["sensor_counts"]:
        print(f"  k={k:<2} {','.join(s.replace('sensor_', '') for s in topk['correlation'][f'top_{k}']['sensors_by_rank'])}"
              f" | {','.join(s.replace('sensor_', '') for s in topk['permutation'][f'top_{k}']['sensors_by_rank'])}")
    print(f"\nRun IDs: " + ", ".join(f"{f}={r.run_id}" for f, r in results.items()))
    print(f"Frozen artifacts: {frozen_dir}")
    print(f"Working outputs : {work_dir}")
    print(f"Results table   : {resolve_path(bench_cfg['outputs']['results_csv'])}")
    print(f"Report          : {resolve_path(cfg['outputs']['report'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
