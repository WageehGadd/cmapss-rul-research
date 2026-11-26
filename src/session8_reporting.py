"""Session 8 figures and research report, generated from the saved tables and predictions.

Every number in the report is read from artifacts/session8 or the frozen Session 7 artifacts. Each
qualitative statement is guarded by `claim`, which stops the report if the data no longer support it.
Figures 1-21 are never touched; Session 8 adds figures 22-32.
"""
import json
import statistics as st

import numpy as np

from .benchmark_config import PROJECT_ROOT, resolve_path
from .data_loading import apply_schema, read_raw_table
from .session3_outputs import markdown_table
from .session7_outputs import MODEL_COLORS
from .session8_data import TEST_SET, fd003_frames, sensor_list
from .session8_evaluate import candidate_members, read_rows, targets, write_rows

FIGURES = {
    "transfer": "fig22_session8_fd001_vs_fd003.png",
    "stages": "fig23_session8_rul_stage_full_vs_reduced.png",
    "bias": "fig24_session8_rul_stage_signed_bias.png",
    "trajectories": "fig25_session8_worst_engine_predictions.png",
    "errors": "fig26_session8_worst_engine_errors.png",
    "sensors": "fig27_session8_worst_engine_sensors.png",
    "asymmetry": "fig28_session8_error_asymmetry.png",
    "engines": "fig29_session8_cross_model_engine_errors.png",
    "tradeoffs": "fig30_session8_tradeoffs.png",
    "caps": "fig31_session8_rul_cap_sensitivity.png",
    "shift": "fig32_session8_input_shift.png",
}
DATASET_LABEL = {"fd001_validation": "FD001 validation (selection-affected)",
                 "fd003_train": "FD003 training trajectories (secondary)",
                 "fd003_test": "FD003 official test (primary)"}
STAGE_LABEL = {"near_failure": "RUL<=50", "intermediate": "51-100", "early_life": ">100"}


def claim(condition, text):
    if not condition:
        raise AssertionError(f"the data no longer support this statement: {text}")
    return text


def load_tables(cfg):
    folder = resolve_path(cfg["outputs"]["artifacts_dir"])
    names = ["architecture_order", "posthoc_checks", "master_results", "candidate_summary", "generalization_gap", "per_engine_candidates", "stage_metrics",
             "stage_full_vs_reduced", "asymmetry", "worst_engines", "family_agreement", "statistics",
             "sensitivity_ranking", "sensitivity_target", "sensitivity_cap", "stage_counts_fd003_test", "input_shift"]
    return {n: read_rows(folder / f"{n}.csv") for n in names}


def sensor_exposure(master, features):
    """EXPLORATORY (post hoc): FD003 test RMSE of reduced conditions with and without each sensor."""
    rows = []
    for sensor in features:
        for model in sorted({r["model"] for r in master}):
            reduced = [r for r in master if r["model"] == model and r["budget"] < 14]
            has = [r["fd003_test_rmse"] for r in reduced if sensor in r["sensors"].split("|")]
            without = [r["fd003_test_rmse"] for r in reduced if sensor not in r["sensors"].split("|")]
            rows.append({"sensor": sensor, "model": model, "n_with": len(has), "n_without": len(without),
                         "median_rmse_with": float(np.median(has)), "median_rmse_without": float(np.median(without)),
                         "label": "exploratory"})
    return rows


# ---- figures -----------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def fig_transfer(plt, t, path):
    summary = {(r["candidate"], r["dataset"]): r for r in t["candidate_summary"]}
    keys = [r["candidate"] for r in t["generalization_gap"]]
    old = {r["candidate"]: r["fd001_test_rmse_previously_inspected"] for r in t["generalization_gap"]}
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(keys))
    for i, (dataset, color) in enumerate(zip(DATASET_LABEL, ("tab:gray", "tab:orange", "tab:red"))):
        values = [summary[k, dataset]["rmse"] for k in keys]
        ax.bar(x + (i - 1) * 0.27, values, 0.26, color=color, label=DATASET_LABEL[dataset])
        if dataset == TEST_SET:
            rows = [summary[k, dataset] for k in keys]
            ax.errorbar(x + 0.27, values, yerr=[[max(v - r["ci95_low"], 0) for v, r in zip(values, rows)],
                                                [max(r["ci95_high"] - v, 0) for v, r in zip(values, rows)]],
                        fmt="none", ecolor="black", lw=1, capsize=2)
    full = [(i, old[k]) for i, k in enumerate(keys) if old[k] != ""]
    ax.scatter([i for i, _ in full], [v for _, v in full], marker="D", color="black", zorder=4,
               label="FD001 official test, earlier sessions (already inspected)")
    ax.set_xticks(x, keys, rotation=30, ha="right")
    ax.set_ylim(0, 40)
    for i, k in enumerate(keys):
        if summary[k, TEST_SET]["rmse"] > 40:
            ax.text(i, 39, f"off scale:\n{summary[k, 'fd003_train']['rmse']:.0f} / {summary[k, TEST_SET]['rmse']:.0f}",
                    ha="center", va="top", fontsize=8)
    ax.set_ylabel("RMSE (cycles), capped target")
    ax.set_title("Zero-shot transfer from FD001 to FD003 (different protocols; bars are not paired)")
    ax.legend(fontsize=8)
    return _save(plt, fig, path)


def fig_stages(plt, t, stages, path):
    keys = ["gru_full", "gru_k10_perm", "gru_k02_perm", "gru_k02_corr"]
    rows = {(r["candidate"], r["dataset"], r["stage"]): r for r in t["stage_metrics"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    x = np.arange(len(stages))
    for ax, dataset in zip(axes, DATASET_LABEL):
        for i, key in enumerate(keys):
            ax.bar(x + (i - 1.5) * 0.2, [rows[key, dataset, s]["engine_mean_mae"] for s in stages], 0.2, label=key)
        ax.set_xticks(x, [f"{STAGE_LABEL[s]}\nn={rows[keys[0], dataset, s]['n_engines']}" for s in stages])
        ax.set_title(DATASET_LABEL[dataset], fontsize=9)
    axes[0].set_ylabel("MAE per engine, averaged over engines (cycles)")
    axes[0].legend(fontsize=8)
    fig.suptitle("GRU: full against reduced sensor sets by true RUL stage (n = engines)")
    return _save(plt, fig, path)


def fig_bias(plt, t, stages, models, path):
    rows = {(r["candidate"], r["dataset"], r["stage"]): r for r in t["stage_metrics"]}
    short = {"random_forest": "rf_full", "gradient_boosting": "gb_full"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(stages))
    for ax, dataset in zip(axes, DATASET_LABEL):
        for i, model in enumerate(models):
            key = short.get(model, f"{model}_full")
            ax.bar(x + (i - 2.5) * 0.13, [rows[key, dataset, s]["engine_mean_signed"] for s in stages], 0.13,
                   color=MODEL_COLORS[model], label=model)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [STAGE_LABEL[s] for s in stages])
        ax.set_title(DATASET_LABEL[dataset], fontsize=9)
    axes[0].set_ylabel("mean signed error, predicted - true (cycles)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Signed bias by stage, full-sensor models (positive = optimistic; panels have own scales)")
    return _save(plt, fig, path)


def selected_engines(cfg, t, dataset):
    return [r["unit_id"] for r in t["worst_engines"]
            if r["dataset"] == dataset and r["criterion"] == "absolute_error"
            and r["view"] == "all_full_models"][:cfg["error_analysis"]["plot_n"]]


def fig_trajectories(plt, cfg, t, inputs, truth, predictions, members, path, errors=False):
    datasets = ("fd001_validation", "fd003_train")
    n = cfg["error_analysis"]["plot_n"]
    models = cfg["error_analysis"]["full_models"]
    key_of = {c["model"]: c["key"] for c in cfg["candidates"] if c["strategy"] == "full"}
    preferred = cfg["preferred_reduced"]
    fig, axes = plt.subplots(len(datasets), n, figsize=(4.6 * n, 7.2), squeeze=False)
    for row, dataset in zip(axes, datasets):
        windows, (target, raw) = inputs[dataset], truth[dataset]
        for ax, unit in zip(row, selected_engines(cfg, t, dataset)):
            mine = windows.unit_ids == unit
            cycles = windows.end_cycles[mine]
            for model in models:
                p = predictions[dataset][members[key_of[model]][0]][mine]
                ax.plot(cycles, p - target[mine] if errors else p, color=MODEL_COLORS[model], lw=1, label=model)
            p = predictions[dataset][members[preferred][0]][mine]
            ax.plot(cycles, p - target[mine] if errors else p, color="black", lw=1, ls="--", label=preferred)
            if errors:
                ax.axhline(0, color="black", lw=0.8)
            else:
                ax.plot(cycles, target[mine], color="black", lw=2, label="true RUL (capped)")
            if (raw[mine] <= 50).any():
                ax.axvspan(cycles[raw[mine] <= 50].min(), cycles.max(), color="tab:red", alpha=0.07)
            ax.set_title(f"{dataset} engine {unit}", fontsize=9)
            ax.set_xlabel("cycle")
        row[0].set_ylabel("predicted - true (cycles)" if errors else "RUL (cycles)")
        if not errors:
            for ax in row:
                ax.set_ylim(-20, 200)
        else:
            for ax in row:
                ax.set_ylim(-60, 150)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(("Error across cycles" if errors else "True and predicted RUL") +
                 ": three highest-error engines per trajectory set (shaded: RUL<=50; values beyond the axis are clipped)")
    fig.tight_layout()
    return _save(plt, fig, path)


def fig_sensors(plt, cfg, t, path):
    features, _, _ = sensor_list(cfg)
    shown = cfg["error_analysis"]["sensor_plot"]
    s1_raw = PROJECT_ROOT / "data/raw/train_FD001.txt"
    frames = {"fd001_validation": apply_schema(read_raw_table(s1_raw)), "fd003_train": fd003_frames(cfg)["train"]}
    engines = [(d, u) for d in frames for u in selected_engines(cfg, t, d)]
    fig, axes = plt.subplots(len(shown), len(engines), figsize=(2.9 * len(engines), 2.1 * len(shown)),
                             sharey="row", squeeze=False)
    for j, (dataset, unit) in enumerate(engines):
        frame = frames[dataset]
        g = frame[frame["unit_id"] == unit]
        for i, sensor in enumerate(shown):
            axes[i][j].plot(g["cycle"], g[sensor], lw=0.8)
            if j == 0:
                axes[i][j].set_ylabel(f"{sensor} (raw)")
        axes[0][j].set_title(f"{dataset.split('_')[0]} engine {unit}", fontsize=9)
        axes[-1][j].set_xlabel("cycle")
    fig.suptitle("Raw sensor trajectories of the highest-error engines")
    fig.tight_layout()
    return _save(plt, fig, path)


def fig_asymmetry(plt, t, models, path):
    short = {"random_forest": "rf_full", "gradient_boosting": "gb_full"}
    rows = {(r["candidate"], r["dataset"]): r for r in t["asymmetry"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(len(models))
    for ax, (field, title) in zip(axes, (("over_fraction", "fraction of optimistic (late) predictions"),
                                         ("mean_signed", "mean signed error (cycles)"),
                                         ("nasa_late_share", "share of NASA score from late predictions"))):
        for i, (dataset, color) in enumerate(zip(DATASET_LABEL, ("tab:gray", "tab:orange", "tab:red"))):
            ax.bar(x + (i - 1) * 0.27, [rows[short.get(m, f"{m}_full"), dataset][field] for m in models], 0.26,
                   color=color, label=DATASET_LABEL[dataset])
        ax.set_xticks(x, models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].axhline(0.5, color="black", lw=0.8, ls=":")
    axes[1].legend(fontsize=7)
    fig.suptitle("Late versus early asymmetry, full-sensor models, pooled over predictions (error = predicted - true)")
    fig.tight_layout()
    return _save(plt, fig, path)


def fig_engines(plt, cfg, t, path):
    models = cfg["error_analysis"]["full_models"]
    key_of = {c["model"]: c["key"] for c in cfg["candidates"] if c["strategy"] == "full"}
    rows = [r for r in t["per_engine_candidates"] if r["dataset"] == TEST_SET]
    values = {(r["candidate"], r["unit_id"]): r["mean_signed"] for r in rows}
    units = sorted({r["unit_id"] for r in rows},
                   key=lambda u: -np.mean([abs(values[key_of[m], u]) for m in models]))
    grid = np.array([[values[key_of[m], u] for u in units] for m in models])
    fig, ax = plt.subplots(figsize=(16, 3.6))
    image = ax.imshow(grid, cmap="RdBu_r", vmin=-80, vmax=80, aspect="auto")
    ax.set_yticks(range(len(models)), models)
    ax.set_xticks(range(len(units)), units, rotation=90, fontsize=5)
    ax.set_xlabel("FD003 test engine (ordered by mean absolute error over the six models; the failed TCN weighs heavily "
                  "in this order)")
    ax.set_title("Signed error per engine and architecture, FD003 official test (colour clipped at +/-80 cycles)")
    fig.colorbar(image, ax=ax, label="predicted - true")
    return _save(plt, fig, path)


def fig_tradeoffs(plt, t, models, path):
    master = t["master_results"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for model in models:
        mine = [r for r in master if r["model"] == model]
        axes[0].scatter([r["fd001_validation_rmse"] for r in mine], [r["fd003_test_rmse"] for r in mine], s=14,
                        color=MODEL_COLORS[model], label=model)
        budgets = sorted({r["budget"] for r in mine}, reverse=True)
        axes[1].plot(budgets, [np.median([r["fd003_test_rmse"] for r in mine if r["budget"] == k]) for k in budgets],
                     marker="o", color=MODEL_COLORS[model])
        full = next(r for r in mine if r["strategy"] == "full")
        for x, y, marker in ((full["fd001_validation_rmse"], full["inference_latency_ms_per_batch"], "o"),
                             (full["fd003_test_rmse"], full["inference_latency_ms_per_batch"], "x")):
            axes[2].scatter(x, y, color=MODEL_COLORS[model], marker=marker, s=50)
    axes[0].set(xlabel="FD001 validation RMSE", ylabel="FD003 test RMSE (log)", yscale="log",
                title="All 174 conditions: in-domain vs transferred error")
    axes[0].set_yticks([20, 40, 80, 160], ["20", "40", "80", "160"])
    axes[0].minorticks_off()
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    axes[0].set_ylim(top=420)
    axes[1].set(xlabel="number of input sensors", ylabel="median FD003 test RMSE (log)", yscale="log",
                title="Median over all subsets at each budget (strategies mixed)")
    axes[1].set_yticks([20, 40, 80, 160], ["20", "40", "80", "160"])
    axes[1].minorticks_off()
    axes[1].invert_xaxis()
    axes[2].set(xlabel="RMSE (o FD001 validation, x FD003 test)", ylabel="latency, ms per 256 windows (log)",
                yscale="log", xscale="log", title="Full-sensor accuracy and cost (latency is coarse)")
    axes[2].set_xticks([12, 20, 40, 80, 120], ["12", "20", "40", "80", "120"])
    axes[2].set_xticks([], minor=True)
    axes[2].set_yticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
    axes[2].set_yticks([], minor=True)
    fig.tight_layout()
    return _save(plt, fig, path)


def fig_caps(plt, t, path):
    rows = t["sensitivity_cap"]
    keys = list(dict.fromkeys(r["candidate"] for r in rows))
    caps = sorted({r["training_cap"] for r in rows})
    value = {(r["candidate"], r["training_cap"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(keys))
    for ax, (field, label) in zip(axes, (("common_subset_rmse", "RMSE (cycles)"),
                                         ("common_subset_mean_signed", "mean signed error (cycles)"))):
        for i, cap in enumerate(caps):
            ax.bar(x + (i - 1) * 0.27, [value[k, cap][field] for k in keys], 0.26, label=f"training cap {cap}")
        ax.set_xticks(x, keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.axhline(0, color="black", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Training-cap sensitivity on FD001 validation windows with true RUL<=100 "
                 f"(n={rows[0]['n_common_windows']}, same targets for every cap).\nLower caps reduce error on this subset "
                 "by construction; one seed", fontsize=10)
    fig.tight_layout()
    return _save(plt, fig, path)


def fig_shift(plt, t, path):
    rows = [r for r in t["input_shift"] if r["used_by_models"]]
    names = [r["sensor"].replace("sensor_", "s") for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    sets = (("fd003_", "FD003 training rows", "tab:blue", -0.2),
            ("fd003_test_windows_", "FD003 test windows (inputs only)", "tab:orange", 0.2))
    for prefix, label, color, dx in sets:
        mean = [r[f"{prefix}mean_in_fd001_sd"] for r in rows]
        axes[0].bar(x + dx, mean, 0.4, color=color, label=label)
        axes[0].errorbar(x + dx, mean, yerr=[r[f"{prefix}sd_in_fd001_sd"] for r in rows], fmt="none",
                         ecolor="black", capsize=2)
        fraction = "fraction_beyond_3_sd" if prefix == "fd003_" else f"{prefix}fraction_beyond_3_sd"
        axes[1].bar(x + dx, [100 * r[fraction] for r in rows], 0.4, color=color)
    axes[0].legend(fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("FD003 mean +/- sd\n(FD001 training sd units)")
    axes[1].set_ylabel("% of rows beyond\n3 FD001 sd")
    axes[1].set_xticks(x, names)
    fig.suptitle("Input shift of the 14 model sensors, FD003 against the FD001-fitted scaler")
    return _save(plt, fig, path)


def make_figures(cfg, canonical, inputs, predictions, tables):
    plt = _plt()
    folder = resolve_path(cfg["outputs"]["figures_dir"])
    path = {k: folder / v for k, v in FIGURES.items()}
    stages = [s["name"] for s in cfg["evaluation"]["stages"]]
    models = cfg["error_analysis"]["full_models"]
    members = candidate_members(cfg, canonical.conditions)
    # trajectory sets only: no test labels are needed for any cycle-wise plot
    truth = targets(cfg, inputs)
    return [fig_transfer(plt, tables, path["transfer"]), fig_stages(plt, tables, stages, path["stages"]),
            fig_bias(plt, tables, stages, models, path["bias"]),
            fig_trajectories(plt, cfg, tables, inputs, truth, predictions, members, path["trajectories"]),
            fig_trajectories(plt, cfg, tables, inputs, truth, predictions, members, path["errors"], errors=True),
            fig_sensors(plt, cfg, tables, path["sensors"]), fig_asymmetry(plt, tables, models, path["asymmetry"]),
            fig_engines(plt, cfg, tables, path["engines"]), fig_tradeoffs(plt, tables, models, path["tradeoffs"]),
            fig_caps(plt, tables, path["caps"]), fig_shift(plt, tables, path["shift"])]


def build_report(cfg, canonical, inputs, predictions):
    from .session8_report_text import write_report

    tables = load_tables(cfg)
    features, _, _ = sensor_list(cfg)
    artifacts = resolve_path(cfg["outputs"]["artifacts_dir"])
    exposure = sensor_exposure(tables["master_results"], features)
    write_rows(artifacts / "exploratory_sensor_exposure.csv", exposure)
    tables["exposure"] = exposure
    figures = make_figures(cfg, canonical, inputs, predictions, tables)
    work = resolve_path(cfg["outputs"]["working_dir"])
    write_report(cfg, tables, json.loads((work / "dataset_facts.json").read_text()))
    return figures
