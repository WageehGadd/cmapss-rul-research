"""Figures, report and frozen output package of Session 7. Everything is generated from the
condition records; no result value or conclusion is typed by hand."""
from pathlib import Path

import numpy as np

from .session3_outputs import markdown_table, write_csv, write_json
from .session7_analysis import OK, full_references
from .session7_plan import sha256_file

MODEL_COLORS = {"random_forest": "tab:green", "gradient_boosting": "tab:olive", "lstm": "tab:orange",
                "gru": "tab:red", "tcn": "tab:blue", "transformer": "tab:purple"}
STRATEGY_MARKERS = {"full": "*", "correlation": "o", "permutation": "s", "random": "x"}
FIGURES = ["fig17_session7_rmse_vs_sensor_count.png", "fig18_session7_rmse_vs_latency_pareto.png",
           "fig19_session7_ranked_vs_random.png", "fig20_session7_relative_degradation_heatmap.png",
           "fig21_session7_per_engine_paired_differences.png"]
CONTAMINATED = ("**Selection-contaminated:** the permutation ranking was computed from the validation labels of the same "
                "engines that score it here. Its comparisons are descriptive and do not show that permutation-based "
                "selection generalizes.")


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ok(rows, **match):
    return [r for r in rows if r["status"] == OK and all(r[k] == v for k, v in match.items())]


def figure_rmse_vs_budget(path, rows, models, budgets, full_budget):
    plt = _pyplot()
    full = full_references(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    ticks = [full_budget, *budgets]
    for ax, model in zip(axes.flat, models):
        for strategy, color in (("correlation", "tab:blue"), ("permutation", "tab:red")):
            pts = [(k, r["val_rmse"]) for k in budgets for r in _ok(rows, model=model, budget=k, strategy=strategy)]
            if model in full:
                pts = [(full_budget, full[model]["val_rmse"]), *pts]
            if pts:
                ax.plot(*zip(*pts), marker=STRATEGY_MARKERS[strategy], color=color, label=strategy)
        means = []
        for k in budgets:
            values = [r["val_rmse"] for r in _ok(rows, model=model, budget=k, strategy="random")]
            if values:
                ax.scatter([k] * len(values), values, s=14, color="grey", alpha=0.7,
                           label="random draws" if k == budgets[0] else None)
                means.append((k, float(np.mean(values))))
        if means:
            ax.plot(*zip(*means), color="grey", ls="--", label="random mean")
        if model in full:
            ax.axhline(full[model]["val_rmse"], color="black", lw=0.8, ls=":", label="full (14 sensors)")
        ax.set_title(model, fontsize=10)
        ax.set_xticks(ticks)
    for ax in axes[-1]:
        ax.set_xlabel("number of sensors")
    for ax in axes[:, 0]:
        ax.set_ylabel("validation RMSE (cycles)")
    axes.flat[0].invert_xaxis()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Validation RMSE versus sensor count")
    fig.tight_layout()
    _save(plt, fig, path)


def figure_pareto(path, rows, front):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ok = [r for r in rows if r["status"] == OK]
    for model, color in MODEL_COLORS.items():
        for strategy, marker in STRATEGY_MARKERS.items():
            pts = [r for r in ok if r["model"] == model and r["strategy"] == strategy]
            if pts:
                ax.scatter([r["inference_latency_ms_per_batch"] for r in pts], [r["val_rmse"] for r in pts],
                           color=color, marker=marker, s=70 if strategy == "full" else 22, alpha=0.75)
    chosen = sorted((r for r in ok if r["experiment_id"] in front), key=lambda r: r["inference_latency_ms_per_batch"])
    if chosen:
        x = [r["inference_latency_ms_per_batch"] for r in chosen]
        y = [r["val_rmse"] for r in chosen]
        ax.plot(x, y, color="black", lw=0.8, drawstyle="steps-post")
        ax.scatter(x, y, s=130, facecolors="none", edgecolors="black", linewidths=1.2)
        for i, (xi, yi) in enumerate(zip(x, y), 1):
            ax.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(9, -3) if i % 2 else (-16, -3),
                        fontsize=7)                             # alternate sides: neighbours do not overlap
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("inference latency per batch of 256 windows (ms, log scale)")
    ax.set_ylabel("validation RMSE (cycles)")
    ax.set_title("All completed conditions; circled = numerically Pareto-efficient (RMSE, latency)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="", label=m) for m, c in MODEL_COLORS.items()]
    handles += [Line2D([], [], color="black", marker=mk, ls="", label=s) for s, mk in STRATEGY_MARKERS.items()]
    ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _save(plt, fig, path)


def figure_ranked_vs_random(path, rows, models, budgets):
    plt = _pyplot()
    fig, axes = plt.subplots(len(budgets), 1, figsize=(10, 2.6 * len(budgets)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), budgets):
        for i, model in enumerate(models):
            draws = [r["val_rmse"] for r in _ok(rows, model=model, budget=k, strategy="random")]
            if draws:
                ax.scatter(i + np.linspace(-0.18, 0.18, len(draws)), draws, color="grey", s=22,
                           label="random draws" if i == 0 else None)
                ax.hlines(np.mean(draws), i - 0.25, i + 0.25, color="grey", label="random mean" if i == 0 else None)
            for strategy, color, dx in (("correlation", "tab:blue", -0.3), ("permutation", "tab:red", 0.3)):
                for r in _ok(rows, model=model, budget=k, strategy=strategy):
                    ax.scatter([i + dx], [r["val_rmse"]], color=color, marker=STRATEGY_MARKERS[strategy], s=45,
                               label=strategy if i == 0 else None)
        ax.set_title(f"k = {k} sensors", fontsize=10)
        ax.set_ylabel("validation RMSE")
    np.atleast_1d(axes)[-1].set_xticks(range(len(models)), models)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Ranked subsets versus five random controls")
    fig.tight_layout()
    _save(plt, fig, path)


def figure_heatmap(path, rows, models, budgets):
    plt = _pyplot()
    full = full_references(rows)
    columns = [(k, s) for k in budgets for s in ("correlation", "permutation", "random")]
    grid = np.full((len(models), len(columns)), np.nan)
    for i, model in enumerate(models):
        for j, (k, strategy) in enumerate(columns):
            values = [r["val_rmse"] for r in _ok(rows, model=model, budget=k, strategy=strategy)]
            if values and model in full:
                grid[i, j] = 100 * (np.mean(values) - full[model]["val_rmse"]) / full[model]["val_rmse"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    limit = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    image = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(columns)), [f"k={k}\n{'random mean' if s == 'random' else s}" for k, s in columns], fontsize=8,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)), models)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(grid[i, j]) > 0.6 * limit else "black")
    fig.colorbar(image, ax=ax, label="relative RMSE increase vs own full-sensor run (%)")
    ax.set_title("Relative RMSE degradation (negative = better than 14 sensors)")
    _save(plt, fig, path)


def figure_per_engine(path, rows, engine_rmse, models, budgets):
    plt = _pyplot()
    full = full_references(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, model in zip(axes.flat, models):
        positions, data, colors = [], [], []
        for i, k in enumerate(budgets):
            for strategy, color, dx in (("correlation", "tab:blue", -0.2), ("permutation", "tab:red", 0.2)):
                for r in _ok(rows, model=model, budget=k, strategy=strategy):
                    if model not in full:
                        continue
                    reference = engine_rmse[full[model]["experiment_id"]]
                    mine = engine_rmse[r["experiment_id"]]
                    positions.append(i + dx)
                    data.append([mine[u] - reference[u] for u in sorted(mine)])
                    colors.append(color)
        if data:
            boxes = ax.boxplot(data, positions=positions, widths=0.32, patch_artist=True, manage_ticks=False)
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(budgets)), [f"k={k}" for k in budgets])
        ax.set_title(model, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("per-engine RMSE change vs full")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="tab:blue", alpha=0.5, label="correlation"),
                        Patch(color="tab:red", alpha=0.5, label="permutation")], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Paired per-engine RMSE differences (validation engines)")
    fig.tight_layout()
    _save(plt, fig, path)


def conclusions(rows, degradation, versus, robustness, stats, pareto_ids, budgets, tolerance):
    """Answers to the research questions, derived from the tables. No claim beyond the data."""
    done = sum(r["status"] == OK for r in rows)
    out = {"complete": done == len(rows), "completed": done, "planned": len(rows), "questions": {}}
    q = out["questions"]
    lines = []
    labels = {"correlation": "correlation subsets (training data only)",
              "permutation": "permutation subsets (selection-contaminated)", "random": "random subsets"}
    for k in budgets:
        for strategy, label in labels.items():
            values = [d["relative_rmse_increase_pct"] for d in degradation
                      if d["budget"] == k and d["strategy"] == strategy]
            if values:
                lines.append(f"k={k}, {label}: relative RMSE change {min(values):+.1f}% to {max(values):+.1f}% "
                             f"(median {np.median(values):+.1f}%, {len(values)} runs).")
    q["accuracy_versus_sensor_count"] = lines
    lines = []
    for strategy in ("correlation", "permutation"):
        mine = [v for v in versus if v["strategy"] == strategy]
        if mine:
            below = sum(v["difference_from_random_mean"] < 0 for v in mine)
            best = sum(v["random_draws_better"] == 0 for v in mine)
            sig = [s for s in stats if s["family"] == "ranked_vs_random_mean" and s["strategy"] == strategy
                   and s.get("available")]
            lower = sum(s["significant_after_correction"] and s["median_difference"] < 0 for s in sig)
            higher = sum(s["significant_after_correction"] and s["median_difference"] > 0 for s in sig)
            lines.append(f"{strategy}: lower RMSE than the random-control mean in {below} of {len(mine)} model-budget "
                         f"pairs and lower than all five draws in {best}; per-engine tests after Holm correction: "
                         f"{lower} significantly lower, {higher} significantly higher, of {len(sig)}.")
    q["ranked_versus_random"] = lines
    ranked = [r for r in robustness if r["strategy"] in ("correlation", "permutation")]
    lines = []
    if ranked:
        by_model = {}
        for r in ranked:
            by_model.setdefault(r["model"], []).append(r["mean_increase_pct"])
        order = sorted(by_model, key=lambda m: np.mean(by_model[m]))
        lines.append("Mean relative increase over the ranked subsets and budgets, smallest first: "
                     + ", ".join(f"{m} {np.mean(by_model[m]):+.1f}%" for m in order) + ".")
        smallest = min(budgets)
        at = [r for r in rows if r["status"] == OK and r["budget"] == smallest
              and r["strategy"] in ("correlation", "permutation")]
        if at:
            best = min(at, key=lambda r: r["val_rmse"])
            lines.append(f"Lowest absolute RMSE at k={smallest}: {best['model']} ({best['strategy']}) "
                         f"{best['val_rmse']:.3f}. Relative robustness and absolute accuracy are different criteria.")
    q["architecture_robustness"] = lines
    chosen = [r for r in rows if r["experiment_id"] in pareto_ids]
    q["performance_latency_sensor_cost"] = [
        f"{len(chosen)} conditions are Pareto-efficient in (RMSE, latency): " + "; ".join(
            f"{r['model']} k={r['budget']} {r['strategy']}{r['subset_seed'] and ' ' + str(r['subset_seed'])} "
            f"(RMSE {r['val_rmse']:.3f}, {r['inference_latency_ms_per_batch']:.2f} ms)"
            for r in sorted(chosen, key=lambda r: r["inference_latency_ms_per_batch"])) + "."] if chosen else []
    q["smaller_sensor_sets"] = [
        f"{r['model']} ({_strategy_label(r['strategy'])}): smallest budget within {tolerance:g}% of its full-sensor "
        f"RMSE: {r['smallest_budget_within_tolerance'] or 'none'}." for r in robustness]
    out["evidence"] = _evidence(rows, versus, robustness, stats, chosen, budgets, tolerance)
    return out


def _strategy_label(strategy):
    return {"random": "mean of the random draws", "permutation": "permutation, selection-contaminated"}.get(strategy,
                                                                                                          strategy)


def _evidence(rows, versus, robustness, stats, front, budgets, tolerance):
    """Sort the findings by how much they can support. All counts come from the tables."""
    random_ok = [f"{r['model']} at k={r['smallest_budget_within_tolerance']}" for r in robustness
                 if r["strategy"] == "random" and r["smallest_budget_within_tolerance"] != ""]
    corr_ok = [f"{r['model']} at k={r['smallest_budget_within_tolerance']}" for r in robustness
               if r["strategy"] == "correlation" and r["smallest_budget_within_tolerance"] != ""]
    perm_ok = [f"{r['model']} at k={r['smallest_budget_within_tolerance']}" for r in robustness
               if r["strategy"] == "permutation" and r["smallest_budget_within_tolerance"] != ""]
    sequences = [r for r in robustness if r["strategy"] in ("correlation", "permutation", "random")]
    monotone = 0
    for r in sequences:
        steps = [0.0, *[r[f"increase_pct_k{k}"] for k in budgets if r[f"increase_pct_k{k}"] != ""]]
        monotone += all(b >= a for a, b in zip(steps, steps[1:]))
    observed = [
        f"Error was higher at smaller budgets overall; it rose at every step in {monotone} of {len(sequences)} "
        "model-strategy sequences (full set first).",
        f"Within the {tolerance:g}% tolerance fixed in the config, the mean of the random draws: "
        f"{', '.join(random_ok) or 'no model'}. Informed selection is therefore not shown to be necessary there.",
        f"Within the tolerance, correlation subsets: {', '.join(corr_ok) or 'no model'}."]

    contaminated = []
    pairs = {}
    for v in versus:
        pairs.setdefault((v["model"], v["budget"]), {})[v["strategy"]] = v
    both = [p for p in pairs.values() if len(p) == 2]
    perm = [v for v in versus if v["strategy"] == "permutation"]
    if perm:
        contaminated.append(f"Permutation subsets had lower RMSE than the random mean in "
                            f"{sum(v['difference_from_random_mean'] < 0 for v in perm)} of {len(perm)} model-budget "
                            "pairs. The pairs share engines and subsets, so they are not independent confirmations.")
    if both:
        contaminated.append(f"Permutation subsets had lower RMSE than correlation subsets in "
                            f"{sum(p['permutation']['ranked_rmse'] < p['correlation']['ranked_rmse'] for p in both)} of "
                            f"{len(both)} pairs. The two methods also differ in kind (univariate last-value correlation "
                            "against importance in one fitted multivariate model), not only in the data they used.")
    contaminated.append(f"Within the tolerance, permutation subsets: {', '.join(perm_ok) or 'no model'}.")

    statistical = []
    done = [s for s in stats if s.get("available")]
    for family, text in (("reduced_ranked_vs_full", "Reduced against full set"),
                         ("ranked_vs_random_mean", "Ranked against the random mean")):
        for strategy in ("correlation", "permutation"):
            mine = [s for s in done if s["family"] == family and s["strategy"] == strategy]
            if mine:
                statistical.append(f"{text}, {strategy}: {sum(s['significant_after_correction'] for s in mine)} of "
                                   f"{len(mine)} significant after Holm correction"
                                   + (" (the test does not account for selection on these engines)."
                                      if strategy == "permutation" else "."))
    open_ = [s for s in done if s["family"] == "reduced_ranked_vs_full" and not s["significant_after_correction"]]
    if open_:
        statistical.append("A non-significant difference is not equivalence: among the non-significant comparisons "
                           f"with the full set, the upper 95% limit of the median per-engine RMSE increase reaches "
                           f"{max(s['ci95_high'] for s in open_):+.1f} cycles.")
    statistical.append("These statements describe 20 validation engines and one training seed.")

    exploratory = [
        "The permutation-against-correlation counts, the practical reading of the Pareto front and the latency "
        "repeatability figures were examined after the results were seen. No hypothesis test supports them.",
        "The hypotheses in section 1 and the choice of Pareto objectives have no timestamped record before training; "
        "the tolerance, test families, correction and bootstrap settings are in the config logged with every run."]

    practical = {}
    for r in front:
        if r["model"] not in practical or r["val_rmse"] < practical[r["model"]]["val_rmse"]:
            practical[r["model"]] = r
    pareto = [f"Numerical: {len(front)} conditions are non-dominated in (RMSE, latency)."]
    if practical:
        pareto.append("Practical (post-hoc reading: lowest-RMSE efficient condition per model, because latency "
                      "differences within a model are not interpretable): " + "; ".join(
                          f"{r['model']} k={r['budget']} {r['strategy']} ({r['val_rmse']:.3f} cycles, "
                          f"{r['inference_latency_ms_per_batch']:.1f} ms)" for r in sorted(
                              practical.values(), key=lambda r: r["inference_latency_ms_per_batch"])) + ".")
    return {"observed_validation_results": observed, "selection_contaminated": contaminated,
            "statistical_20_engines": statistical, "exploratory_post_hoc": exploratory,
            "pareto_numerical_vs_practical": pareto}


def _short_answers(evidence, versus, stats):
    """One line per research question, with every count taken from the tables."""
    beat = []
    for strategy, note in (("correlation", ""), ("permutation", " (selection-contaminated)")):
        mine = [v for v in versus if v["strategy"] == strategy]
        if mine:
            beat.append(f"{strategy}{note} subsets beat the random mean in "
                        f"{sum(v['difference_from_random_mean'] < 0 for v in mine)} of {len(mine)} pairs")
    tests = [s for s in stats if s["family"] == "ranked_vs_random_mean" and s.get("available")]
    significant = sum(s["significant_after_correction"] for s in tests)
    return ["Short answers (details and limits in section 16):", "",
            f"1. {evidence['observed_validation_results'][0]}",
            f"2. {'; '.join(beat) or 'No comparison available'}; {significant} of {len(tests)} ranked-against-random "
            "tests are significant. A general advantage of informed selection is not established.",
            "3. Descriptive only (section 12).",
            "4. Latency differences within one model are not interpretable (section 14).",
            f"5. {evidence['observed_validation_results'][1]} {evidence['observed_validation_results'][2]} "
            f"{evidence['selection_contaminated'][-1][:-1]} (selection-contaminated).", ""]


def _incident_evidence(diagnosis):
    """Measured support for the diagnosis, when the versioned incident record has it."""
    measured = diagnosis.get("measured_evidence")
    if not measured:
        return []
    train = measured["features"]["train"]
    own = all(measured["cross_prediction"][key][flag] for key, flag in (
        ("failed_model_on_noncontiguous_features", "equals_failed_predictions"),
        ("final_model_on_contiguous_features", "equals_final_predictions")))
    return [f"Measured afterwards without training: {train['differing_feature_values']:,} training feature values "
            f"differ by at most {train['max_absolute_difference']:.1e}; {train['differing_after_float32_cast']} still "
            f"differ after a float32 cast. Each saved model reproduces its own predictions exactly on its own feature "
            f"layout: {own}. The two models' trees first differ at tree "
            f"{measured['first_structurally_different_tree']} of {measured['n_trees']}.", "",
            *[f"- Correction: {text}" for text in diagnosis.get("corrections", [])],
            *[f"- Limit: {text}" for text in diagnosis.get("limits", [])], ""]


def _hardware_notes(conditions):
    if not conditions:
        return ["Power source and concurrent load were not recorded for these runs."]
    seen, repeat = conditions["operator_observations"], conditions["latency_repeatability"]
    return [f"Power source and load during the runs: {seen['power']}; {seen['load']}. These are operator "
            "observations, not run metadata.",
            "The same full-sensor definition timed twice differed by up to "
            f"{repeat['max_relative_difference_pct']:.0f}% in latency (one repeat per model)."]


def _gate_history(diagnosis):
    """An earlier gate failure and its diagnosis, when one was recorded (never hidden)."""
    if not diagnosis:
        return []
    failed = [c for c in diagnosis["gate"]["checks"] if not c["passed"]]
    text = ", ".join(f"{c['model']} (RMSE {c['rmse']:.4f} vs canonical {c['canonical_rmse']:.4f})" for c in failed)
    return [f"**Earlier gate failure (verification stage).** A first execution failed the gate for {text}; the reduced matrix was not "
            f"started. Diagnosis: {diagnosis['diagnosis']} Fix: {diagnosis['fix']} The {len(diagnosis['superseded_runs'])} "
            "superseded MLflow runs are kept and tagged `s7.status=superseded`.", "",
            "This also shows that gradient boosting results here can move by a few hundredths of RMSE under changes in "
            "the last bits of the features; small gradient-boosting differences should not be interpreted.", ""]


def _latency_caveat(front):
    """How little latency separates the efficient conditions of one model (timing noise, not a sensor effect)."""
    spans = {}
    for r in front:
        spans.setdefault(r["model"], []).append(r["inference_latency_ms_per_batch"])
    out = [f"{m}: {len(v)} efficient conditions within {max(v) - min(v):.2f} ms of each other"
           for m, v in spans.items() if len(v) > 1]
    if not out:
        return []
    return ["Read this front with care. " + "; ".join(out) + ". Each latency is the median of repeated calls "
            "in a single timing block per condition, so differences this small are within machine noise: they do "
            "not show that a smaller or random subset is faster.", ""]


def write_report(path, ctx):
    cfg, plan, rows = ctx["cfg"], ctx["plan"], ctx["rows"]
    evidence, figures_link = ctx["conclusions"]["evidence"], ctx["figures_link"]
    produced = ctx["manifest"]["execution_provenance"]
    con, git, env = ctx["conclusions"], ctx["git"], ctx["environment"]
    budgets = cfg["matrix"]["budgets"]
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    complete = con["complete"]
    banner = ("" if complete else f"**INCOMPLETE EXPERIMENT: {con['completed']} of {con['planned']} conditions "
              "completed. No conclusion below should be read as the finished study.**")
    verification = ("**Verification run from an uncommitted working tree (`git_dirty=True`); not canonical.**"
                    if git["git_dirty"] else "")
    num = {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa_score": ".0f", "train_time_s": ".1f",
           "inference_latency_ms_per_batch": ".2f", "relative_pct": "+.1f"}

    def table(data, columns, formats=None):
        clean = [{c: ("-" if d.get(c, "") == "" else d.get(c, "")) for c in columns} for d in data]
        formats = {k: v for k, v in (formats or {}).items() if k in columns}
        safe = {k: v for k, v in formats.items() if all(not isinstance(c[k], str) for c in clean)}
        for k, v in formats.items():
            if k not in safe:
                for c in clean:
                    if not isinstance(c[k], str):
                        c[k] = format(c[k], v)
        return markdown_table(clean, columns, safe) if clean else "_No data._"

    deg = {d["experiment_id"]: d["relative_rmse_increase_pct"] for d in ctx["degradation"]}
    all_rows = [{**r, "relative_pct": deg.get(r["experiment_id"], ""), "run": r["mlflow_run_id"][:8],
                 "subset": f"{r['strategy']} {r['subset_seed']}".strip()} for r in rows]
    lines = ["# Session 7: sensor budget x model architecture", "",
             f"Config `{ctx['config_ref']}`, source commit `{str(git['git_commit'])[:12]}`. {verification} {banner}", "",
             "**Read first.**", "",
             "- All results are validation results. **No official-test evaluation was performed.**",
             f"- {CONTAMINATED}",
             "- One training seed, 20 validation engines, FD001 only.",
             "- Latency was measured once per condition on one laptop and depends on hardware, power and load.", "",
             "## 1. Research questions and hypotheses", "",
             "1. How does RUL accuracy change as sensors are removed? (H: error grows as the budget shrinks.)",
             "2. Does informed selection beat random selection? (H: ranked subsets have lower error than random ones.)",
             "3. Which architecture is most robust to aggressive reduction? (no directional hypothesis)",
             "4. How do accuracy, latency and sensor cost relate?",
             f"5. Can a much smaller sensor set stay within {cfg['analysis']['close_to_full_tolerance_pct']:g}% of the "
             "full-sensor RMSE? (tolerance declared before training)", "",
             *_short_answers(evidence, ctx["versus"], ctx["statistics"]),
             "## 2. Frozen data identity", "",
             f"FD001, RUL cap 125, engine split seed 42 ({plan['data']['n_engines']['train']} training / "
             f"{plan['data']['n_engines']['validation']} validation engines), training-only scaler, window length 30. "
             f"Arrays: train {plan['data']['shapes']['train']}, validation {plan['data']['shapes']['validation']}. Session 1 "
             f"manifest SHA-256 `{plan['data']['manifest_sha256'][:16]}...`. Sensor columns are sliced in the original "
             "Session 1 order. The official test arrays are never opened.", "",
             "## 3. Frozen architectures", "",
             table([{"model": m, "definition": ctx["definitions"][m]} for m in cfg["matrix"]["models"]],
                   ["model", "definition"]), "",
             f"Deep models: Adam {plan['training_protocol']['learning_rate']}, batch {plan['training_protocol']['batch_size']}, "
             f"max {plan['training_protocol']['max_epochs']} epochs, patience "
             f"{plan['training_protocol']['early_stopping_patience']}, min delta {plan['training_protocol']['min_delta']}, "
             f"clip {plan['training_protocol']['grad_clip']}, ReduceLROnPlateau, targets / 125, seed "
             f"{plan['training_protocol']['seed']}, CPU. Classical models: Session 3 hyperparameters, six temporal "
             "features per selected sensor (6k columns), no new search.", "",
             "## 4. Frozen sensor-selection strategies", "",
             "Correlation (training-only) and permutation (validation) rankings and five random draws per budget "
             "(subset seeds 101-105) are read from the Session 3 artifacts. Subset seeds select sensors; the training "
             "seed is 42 for every run. Note that the permutation ranking was built from validation data, so its "
             "validation results are not independent of the selection step.", "",
             table(ctx["subsets"], ["budget", "strategy", "sensors"]), "",
             "## 5. Experiment matrix", "",
             f"{plan['n_conditions']} conditions: per model one 14-sensor reference and, for k in {budgets}, correlation, "
             f"permutation and five random subsets. Identical sensor sets are trained once. Status counts: {counts}.", "",
             "## 6. Full-sensor reproducibility gate", "",
             table(ctx["gate"]["checks"], ["model", "rmse", "canonical_rmse", "predictions_identical",
                                            "weights_identical", "n_parameters", "passed"],
                   {"rmse": ".4f", "canonical_rmse": ".4f"}), "",
             f"Gate passed: {ctx['gate']['passed']}. Each full-sensor run was retrained from its frozen definition and "
             "compared with the canonical validation predictions of its session (weights too for deep models).", "",
             *_gate_history(ctx.get("gate_diagnosis")),
             *_incident_evidence(ctx.get("gate_diagnosis") or {}),
             "## 7. All model-condition validation metrics", "",
             table(all_rows, ["model", "budget", "subset", "val_rmse", "val_mae", "val_r2", "val_nasa_score",
                              "relative_pct", "n_parameters", "train_time_s", "inference_latency_ms_per_batch",
                              "status", "run"], num), "",
             "## 8. Sensor-budget effect", "", CONTAMINATED, "", *[f"- {t}" for t in con["questions"]["accuracy_versus_sensor_count"]], "",
             f"![rmse vs sensors]({figures_link}/{FIGURES[0]})", "",
             "## 9. Correlation versus permutation ranking", "", CONTAMINATED + " This comparison was not planned "
             "and has no test.", "",
             table(ctx["ranking_pairs"], ["model", "budget", "correlation_rmse", "permutation_rmse",
                                          "permutation_minus_correlation"],
                   {"correlation_rmse": ".3f", "permutation_rmse": ".3f", "permutation_minus_correlation": "+.3f"}), "",
             "## 10. Ranked versus random", "", CONTAMINATED + " Correlation and random subsets did not use "
             "validation labels.", "", *[f"- {t}" for t in con["questions"]["ranked_versus_random"]], "",
             table(ctx["versus"], ["model", "budget", "strategy", "ranked_rmse", "random_mean_rmse", "random_min_rmse",
                                   "random_max_rmse", "random_draws_better"],
                   {"ranked_rmse": ".3f", "random_mean_rmse": ".3f", "random_min_rmse": ".3f", "random_max_rmse": ".3f"}),
             "", "Random controls summarised (five draws each):", "",
             table(ctx["random"], ["model", "budget", "n_draws", "rmse_mean", "rmse_median", "rmse_std", "rmse_min",
                                   "rmse_max"], {k: ".3f" for k in ("rmse_mean", "rmse_median", "rmse_std",
                                                                    "rmse_min", "rmse_max")}), "",
             f"![ranked vs random]({figures_link}/{FIGURES[2]})", "",
             "## 11. Relative degradation", "",
             "`100 x (RMSE_reduced - RMSE_full) / RMSE_full` against the same model's Session 7 full-sensor run; negative "
             "values are kept.", "", f"![degradation]({figures_link}/{FIGURES[3]})", "",
             "## 12. Architecture robustness", "", "Descriptive. Relative increases depend on each model's own "
             "full-sensor error, and the permutation rows are selection-contaminated.", "", *[f"- {t}" for t in con["questions"]["architecture_robustness"]], "",
             table(ctx["robustness"], ["model", "strategy", "full_rmse", *[f"increase_pct_k{k}" for k in budgets],
                                       "mean_increase_pct", "smallest_budget_within_tolerance"],
                   {"full_rmse": ".3f", "mean_increase_pct": "+.1f", **{f"increase_pct_k{k}": "+.1f" for k in budgets}}), "",
             f"Smaller sensor sets (tolerance {cfg['analysis']['close_to_full_tolerance_pct']:g}%):", "",
             *[f"- {t}" for t in con["questions"]["smaller_sensor_sets"]], "",
             "## 13. Inference and model-size trade-offs", "",
             table(ctx["efficiency"], ["model", "budget", "conditions", "median_latency_ms", "median_train_time_s",
                                       "median_model_size_bytes", "median_n_parameters"],
                   {"median_latency_ms": ".2f", "median_train_time_s": ".1f", "median_model_size_bytes": ".0f",
                    "median_n_parameters": ".0f"}), "",
             "Classical latency includes computing the temporal features; tree 'parameters' are node counts.", "",
             "## 14. Pareto analysis", "",
             "A condition is Pareto-efficient if no other is at least as good in all objectives and strictly better in "
             "one. Computed programmatically over all completed conditions.", "",
             "Objectives (validation RMSE, latency per batch):", "",
             table(ctx["pareto_rows"], ["rank", "model", "budget", "subset", "val_rmse", "inference_latency_ms_per_batch"], num),
             "", f"With sensor count as a third objective, {len(ctx['pareto3_ids'])} conditions are efficient "
             f"(listed in `{ctx['artifacts_ref']}/pareto.json`).", "",
             *_latency_caveat(ctx["pareto_rows"]),
             "**Numerical versus practical membership**", "",
             *[f"- {t}" for t in evidence["pareto_numerical_vs_practical"]],
             *[f"- {t}" for t in _hardware_notes(ctx.get("execution_conditions"))],
             "- Membership among conditions of similar latency can change with timing noise. Efficient random or "
             "reduced subsets are not shown to be faster.", "", f"![pareto]({figures_link}/{FIGURES[1]})", "",
             "## 15. Planned paired statistical comparisons", "",
             "Unit of analysis: validation engine (per-engine RMSE over its windows; overlapping windows are never "
             "treated as independent). Two-sided Wilcoxon signed-rank, Holm correction within each of the two "
             "predeclared families, effect size = matched-pairs rank-biserial, 95% bootstrap interval of the median "
             "difference (engines resampled). Negative difference = lower error for the ranked subset. With one "
             "training seed, these tests describe variation across engines only, not training randomness.", "",
             table(ctx["statistics"], ["family", "model", "budget", "strategy", "n_engines", "median_difference",
                                       "ci95_low", "ci95_high", "rank_biserial", "p_value", "p_holm",
                                       "significant_after_correction"],
                   {"median_difference": "+.3f", "ci95_low": "+.3f", "ci95_high": "+.3f", "rank_biserial": "+.2f",
                    "p_value": ".2g", "p_holm": ".2g"}), "",
             f"![per engine]({figures_link}/{FIGURES[4]})", "",
             "## 16. Interpretation and limitations", "",
             "### Observed validation results", "", *[f"- {t}" for t in evidence["observed_validation_results"]], "",
             "### Selection-contaminated comparisons", "", CONTAMINATED, "",
             *[f"- {t}" for t in evidence["selection_contaminated"]], "",
             "### Statistical conclusions (20 engines)", "",
             *[f"- {t}" for t in evidence["statistical_20_engines"]], "",
             "### Exploratory and post-hoc findings", "",
             *[f"- {t}" for t in evidence["exploratory_post_hoc"]], "",
             "### Pareto membership", "", *[f"- {t}" for t in evidence["pareto_numerical_vs_practical"]], "",
             "### Hardware-dependent latency", "",
             *[f"- {t}" for t in _hardware_notes(ctx.get("execution_conditions"))],
             "- Latency and training time depend on this machine, its load and library threading; they compare model "
             "families coarsely and say nothing reliable about sensor subsets.", "",
             "### Limitations", "",
             "- One training seed: differences of a few percent can come from training randomness and are not evidence "
             "of superiority.",
             "- Validation windows overlap heavily and come from only 20 engines.",
             "- The same validation engines were used to select architectures and hyperparameters (Sessions 3-6), for "
             "early stopping, to build the permutation ranking and to report these results. Absolute performance is "
             "optimistic for unseen engines.",
             "- Evidence for generalization would need engines that played no part in any selection, for example "
             "engine-level cross-validation with the ranking rebuilt inside each fold. That was not done.",
             "- Families differ in inputs (temporal features vs raw windows), original search budgets and size measures.",
             "- Architectures were selected with all 14 sensors; they were deliberately not retuned for smaller sets.",
             "- Only FD001 (one operating condition, one fault mode); generalisation is Session 8.", "",
             "## 17. Failed or unstable conditions", "",
             (table([r for r in rows if r["status"] != OK], ["experiment_id", "status", "attempt", "error"])
              if any(r["status"] != OK for r in rows) else f"None: all {len(rows)} conditions completed."), "",
             "## 18. Validation / test separation", "",
             "Training uses the training windows; early stopping and every reported metric use the validation windows. "
             "The test files are not opened by any Session 7 code, no condition has a test metric, and no test result "
             "influenced the plan or the analysis. Historical official-test results are unchanged. The reproducibility "
             "gate reads the historical results file and keeps only its validation rows.", "",
             "## 19. Reproducibility metadata", "",
             table([{"item": k, "value": v} for k, v in {
                 "python": env["python"]["version"], "platform": env["os"]["platform"],
                 "hardware": env["hardware"]["description"], "torch": env["libraries"].get("torch", ""),
                 "scikit-learn": env["libraries"].get("scikit-learn", ""), "git_commit": git["git_commit"],
                 "git_dirty": git["git_dirty"], "total_condition_time_h": f"{ctx['total_wall_s'] / 3600:.2f}",
                 "conditions_with_more_than_one_attempt": sum(int(r["attempt"] or 1) > 1 for r in rows),
                 "execution_code_fingerprint": ", ".join(f[:16] for f in produced["code_fingerprints"]),
                 "current_source_equals_execution_source": produced["source_matches_current"],
                 "executed_between_utc": f"{produced['first_started_at']} and {produced['last_ended_at']}",
                 "official_test_evaluation": "not performed"}.items()],
                 ["item", "value"]), "",
             "Software and hardware rows describe the machine that wrote this report; the execution environment of the "
             "runs is in the manifest (`execution_provenance`), with hashes of the sources that wrote this report "
             "(`report_source_hashes`)."
             + ("" if produced["source_matches_current"] else " **The source has changed since the runs were executed: "
                "the code that trained them is not the code listed there, and it was never committed.**"), "",
             "## 20. Session 8 handoff", "",
             f"- `{ctx['artifacts_ref']}/experiment_plan.json`: every condition with sensors, indices and definition "
             "hashes.",
             "- Everything below is git-ignored and exists only locally; back it up before any cleanup.",
             "- `results/session7/conditions/<id>.json`: status, MLflow run ID, run folder and file hashes.",
             "- `results/session7/runs/<run_id>/`: model files and window-level validation predictions "
             "(`predictions.npz`).",
             "- `results/session7/validation_index.npz`: engine ID, end cycle and target of each validation window.",
             "- `results/session7/per_engine_metrics.csv`: per-engine metrics of every condition.", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_package(frozen_dir, ctx):
    """Versionable evidence plus a manifest of hashes and identities."""
    frozen = Path(frozen_dir)
    rows = ctx["rows"]
    write_json(frozen / "experiment_plan.json", ctx["plan"])
    summary = [c for c in rows[0] if c not in ("run_dir", "error")]
    write_csv(frozen / "master_summary.csv", [{k: r[k] for k in summary} for r in rows], summary)
    for name, data in (("degradation.csv", ctx["degradation"]), ("random_controls.csv", ctx["random"]),
                       ("ranked_vs_random.csv", ctx["versus"]), ("robustness.csv", ctx["robustness"]),
                       ("statistics.csv", ctx["statistics"])):
        columns = []
        for d in data:
            columns += [k for k in d if k not in columns]
        write_csv(frozen / name, [{k: d.get(k, "") for k in columns} for d in data], columns or ["empty"])
    write_json(frozen / "pareto.json", {"dominance": "<= in all objectives and < in at least one; all minimised",
                                         "rmse_latency": ctx["pareto_ids"], "rmse_latency_sensors": ctx["pareto3_ids"]})
    write_json(frozen / "reproducibility_gate.json", ctx["gate"])
    write_json(frozen / "conclusions.json", ctx["conclusions"])
    files = {p.name: sha256_file(p) for p in sorted(frozen.iterdir()) if p.name != "manifest.json"}
    write_json(frozen / "manifest.json", {**ctx["manifest"], "files": files})
