"""The Session 8 research report. Numbers come from the tables passed in; sentences that state a
direction or a ranking are wrapped in `claim` so that they cannot outlive the data."""
import csv
import statistics as st

from .benchmark_config import resolve_path
from .session3_outputs import markdown_table
from .session8_reporting import DATASET_LABEL, FIGURES, claim

NAMES = {"random_forest": "random forest", "gradient_boosting": "gradient boosting", "lstm": "LSTM", "gru": "GRU",
         "tcn": "TCN", "transformer": "Transformer"}
FULL = {"random_forest": "rf_full", "gradient_boosting": "gb_full", "lstm": "lstm_full", "gru": "gru_full",
        "tcn": "tcn_full", "transformer": "transformer_full"}


def _csv(path):
    with open(resolve_path(path), newline="") as f:
        return list(csv.DictReader(f))


def _table(rows, columns, formats=None, headers=None):
    formats = formats or {}
    clean = [{c: (format(r[c], formats[c]) if c in formats and isinstance(r[c], (int, float)) else
                  ("-" if r.get(c, "") == "" else r[c])) for c in columns} for r in rows]
    if headers:
        clean = [{headers.get(k, k): v for k, v in r.items()} for r in clean]
        columns = [headers.get(c, c) for c in columns]
    return markdown_table(clean, columns)


def facts(cfg, t):
    """The derived quantities the narrative uses."""
    s = {(r["candidate"], r["dataset"]): r for r in t["candidate_summary"]}
    gap = {r["candidate"]: r for r in t["generalization_gap"]}
    tests = {(r["family"], r["a"]): r for r in t["statistics"]}
    stage = {(r["candidate"], r["dataset"], r["stage"]): r for r in t["stage_metrics"]}
    compare = {(r["candidate"], r["dataset"], r["stage"]): r for r in t["stage_full_vs_reduced"]}
    asym = {(r["candidate"], r["dataset"]): r for r in t["asymmetry"]}
    family_c = [r for r in t["statistics"] if r["family"] == "C_ranked_vs_random_mean"]
    ranked = {}
    for strategy in ("correlation", "permutation"):
        c = [r for r in family_c if r["strategy"] == strategy]
        v = [r for r in t["sensitivity_ranking"] if r["strategy"] == strategy]
        ranked[strategy] = {
            "n": len(c),
            "sig_lower": sum(r["significant_after_correction"] and r["median_difference"] < 0 for r in c),
            "sig_higher": sum(r["significant_after_correction"] and r["median_difference"] > 0 for r in c),
            **{f"{d}_below": sum(r[f"{d}_ranked_rmse"] < r[f"{d}_random_mean_rmse"] for r in v)
               for d in ("fd001_val", "fd003_train", "fd003_test")},
            "fd003_test_below_without_tcn": sum(r["fd003_test_ranked_rmse"] < r["fd003_test_random_mean_rmse"]
                                                for r in v if r["model"] != "tcn")}
    agreement = {d: [r for r in t["family_agreement"] if r["dataset"] == d] for d in DATASET_LABEL}
    return {"s": s, "gap": gap, "tests": tests, "stage": stage, "compare": compare, "asym": asym, "ranked": ranked,
            "agreement": agreement}


def write_report(cfg, t, datasets):
    f = facts(cfg, t)
    s, gap, tests, stage, compare, asym, ranked = (f[k] for k in ("s", "gap", "tests", "stage", "compare", "asym",
                                                                  "ranked"))
    models = cfg["error_analysis"]["full_models"]
    test, val, train = "fd003_test", "fd001_validation", "fd003_train"
    rmse = lambda k, d=test: s[k, d]["rmse"]
    A = lambda k: tests["A_reduced_vs_full", k]
    B = lambda k: tests["B_full_vs_reference", k]
    by_test = sorted(models, key=lambda m: rmse(FULL[m]))
    by_val = sorted(models, key=lambda m: rmse(FULL[m], val))
    master = t["master_results"]
    full_rows = {r["model"]: r for r in master if r["strategy"] == "full"}
    best_overall = min(master, key=lambda r: r["fd003_test_rmse"])
    robustness = _csv("artifacts/session7/robustness.csv")
    mean_increase = {m: st.mean(float(r["mean_increase_pct"]) for r in robustness
                                if r["model"] == m and r["strategy"] != "random") for m in models}
    least_sensitive = min(mean_increase, key=mean_increase.get)
    smallest = {st_: {r["model"]: int(r["smallest_budget_within_tolerance"]) for r in robustness
                      if r["strategy"] == st_ and r["smallest_budget_within_tolerance"]}
                for st_ in ("correlation", "permutation", "random")}
    caps = t["sensitivity_cap"]
    cap_values = sorted({r["training_cap"] for r in caps})
    cap_order = {c: [r["candidate"] for r in sorted(
        (r for r in caps if r["training_cap"] == c and r["rank_among_full_models"] != ""),
        key=lambda r: r["rank_among_full_models"])] for c in cap_values}
    cap_same = len({tuple(v) for v in cap_order.values()}) == 1
    cap = {(r["candidate"], r["training_cap"]): r for r in caps}
    exposure = {(r["model"], r["sensor"]): r for r in t["exposure"]}
    counts = {r["stage"]: r["n_engines"] for r in t["stage_counts_fd003_test"]}

    def worst_list(dataset, criterion, view="all_full_models"):
        return [r["unit_id"] for r in t["worst_engines"]
                if r["dataset"] == dataset and r["criterion"] == criterion and r["view"] == view]

    test_worst = worst_list(test, "absolute_error")
    test_worst_no_tcn = worst_list(test, "absolute_error", "without_tcn")
    order = t["architecture_order"]
    rank = {(r["protocol"], r["model"]): r for r in order if r["kind"] == "rmse"}
    rho = {r["protocol"]: r["value"] for r in order if r["kind"] == "spearman"}
    rho_test = rho["fd001_test_previously_inspected vs fd003_test"]
    rho_val = rho["fd001_validation vs fd003_test"]
    rho_fd001 = rho["fd001_validation vs fd001_test_previously_inspected"]
    posthoc = {(r["a"], r["b"]): r for r in t["posthoc_checks"] if r["check"] == "joint_holm_and_sign_test"}
    top5 = {r["a"]: r["value"] for r in t["posthoc_checks"] if r["check"] == "nasa_top5_share"}
    non_tcn = [m for m in models if m != "tcn"]
    ratios = [gap[FULL[m]]["fd003_over_fd001_test_ratio"] for m in non_tcn]
    ag = f["agreement"]

    claim(by_val[0] == cfg["evaluation"]["reference_model"] == "transformer",
          "the Transformer has the lowest FD001 validation RMSE")
    claim(all(rmse(FULL[m]) > rmse(FULL[m], val) for m in models), "every full model is worse on FD003 test")
    claim(by_test[-1] == "tcn" and by_test[0] == "random_forest", "random forest best, TCN worst on FD003 test")
    claim(rho_test > 0.7 and rho_val < 0.3 and rank["fd001_test_previously_inspected", "random_forest"]["rank"] == 1,
          "on matched terminal protocols the architecture order is largely similar, with random forest first on both")
    claim(not set(test_worst) & set(test_worst_no_tcn), "removing the TCN changes the whole FD003 test worst list")
    claim(all(stage[FULL[m], train, "intermediate"]["engine_mean_mae"] >
              stage[FULL[m], train, "near_failure"]["engine_mean_mae"]
              for m in ("random_forest", "gradient_boosting", "lstm", "gru")),
          "on FD003 trajectories intermediate error exceeds near-failure error for four models")
    claim(posthoc["gru_k02_perm", "gru_full"]["p_holm_all_tests"] < 0.05
          < posthoc["gru_full", "transformer_full"]["p_holm_all_tests"], "joint correction pattern")
    claim(all(B(FULL[m])["significant_after_correction"] and B(FULL[m])["median_difference"] < 0
              for m in ("gru", "gradient_boosting", "random_forest", "lstm")) and B("tcn_full")["median_difference"] > 0,
          "four architectures beat the Transformer on FD003 test and the TCN is worse")
    claim(not A("gru_k10_perm")["significant_after_correction"], "preferred reduced vs full is not significant")
    claim(rmse("gru_k10_perm") > 2 * rmse("gru_k10_perm", val), "preferred reduced error more than doubles")
    claim(all(A(k)["median_difference"] < 0 and A(k)["significant_after_correction"]
              for k in ("tcn_k07_perm", "gru_k02_perm")), "two reduced candidates beat their full model on FD003")
    claim(ranked["correlation"]["fd001_val_below"] < 12 < ranked["permutation"]["fd001_val_below"],
          "ranking direction on FD001")
    claim(ranked["permutation"]["sig_higher"] == 0 and ranked["correlation"]["sig_higher"] >
          ranked["correlation"]["sig_lower"], "family C pattern")
    for m in models:
        claim(stage[FULL[m], test, "near_failure"]["mean_signed"] > 0
              and stage[FULL[m], test, "near_failure"]["mae"] > stage[FULL[m], test, "early_life"]["mae"],
              "near failure is the worst and optimistic stage on FD003 test")
        claim(0.45 < asym[FULL[m], val]["over_fraction"] < 0.55, "FD001 validation is balanced in count")
        claim(asym[FULL[m], test]["over_fraction"] > 0.5 and asym[FULL[m], test]["nasa_late_share"] > 0.9,
              "optimistic errors dominate on FD003 test")
        claim(stage[FULL[m], val, "near_failure"]["engine_mean_mae"] <
              stage[FULL[m], val, "intermediate"]["engine_mean_mae"],
              "near failure is easier than intermediate on FD001")
    for k in ("gru_k02_perm", "gru_k02_corr"):
        c = {x: compare[k, val, x] for x in ("near_failure", "early_life")}
        claim(c["near_failure"]["mae_change_pct"] > c["early_life"]["mae_change_pct"] > 0,
              "two sensors hurt near failure relatively more on FD001")
        claim(compare[k, test, "near_failure"]["mae_change"] < 0, "the same reduction helps near failure on FD003")
    claim(least_sensitive == "tcn", "TCN has the smallest mean relative increase on FD001")
    for sensor in ("sensor_7", "sensor_12"):
        claim(all(exposure[m, sensor]["median_rmse_with"] > exposure[m, sensor]["median_rmse_without"] for m in models),
              "subsets containing sensors 7 and 12 transfer worse for every model")

    lat = lambda m: full_rows[m]["inference_latency_ms_per_batch"]
    stage_name = {"near_failure": "RUL<=50", "intermediate": "51-100", "early_life": ">100"}

    L = ["# Reliability, generalisation and conclusions: sensor-reduced RUL models on NASA C-MAPSS", "",
         "Session 8. Protocol: [session8_evaluation_protocol.md](session8_evaluation_protocol.md), configuration "
         "`configs/session8_evaluation.yaml`. Reproduction: "
         "[session8_reproducibility.md](session8_reproducibility.md).", "",
         "**Read first.** All data are simulated. FD001 validation results are affected by selection. FD003 official-test "
         "results are the primary transfer evidence; according to the local access log and hashes (self-recorded, not "
         "externally attested) the candidates and test families were fixed before the labels were read. "
         "Permutation-ranked subsets were chosen with the FD001 validation labels. All narrative, figures and anything "
         "marked *post hoc* or *exploratory* were produced after seeing results. The results come from an uncommitted "
         "working tree and are not canonical. This text was revised after an independent audit "
         "([session8_audit_remediation.md](session8_audit_remediation.md)).", "",
         "## 1. Research objectives", "",
         "Determine whether the FD001 sensor-efficiency findings hold on FD003, where the models fail, how their errors "
         "are directed, and what can be concluded.", "",
         "## 2. Dataset and protocol", "",
         "| Property | FD001 | FD003 |", "|---|---|---|",
         "| Operating conditions (NASA readme) | one, sea level | one, sea level |",
         "| Fault modes (NASA readme) | HPC degradation | HPC degradation and fan degradation |",
         f"| Engines in training file | {datasets['fd001_train_file']['engines']} (80 train / 20 validation) | "
         f"{datasets['fd003_train_file']['engines']} (all evaluation only) |",
         f"| Lifetime, min / median / max cycles | {datasets['fd001_train_file']['min_cycles']} / "
         f"{datasets['fd001_train_file']['median_cycles']:.0f} / {datasets['fd001_train_file']['max_cycles']} | "
         f"{datasets['fd003_train_file']['min_cycles']} / {datasets['fd003_train_file']['median_cycles']:.0f} / "
         f"{datasets['fd003_train_file']['max_cycles']} |", "",
         "The files give no fault label per engine. Transfer is zero-shot: the 14 FD001 sensors in order, the "
         "FD001-fitted scaler applied unchanged, 30-cycle windows, target min(RUL, 125). Three sets are scored and "
         "are not interchangeable:", "",
         f"- FD001 validation: {datasets['windows'][val]['n_windows']} windows from 20 engines.",
         f"- FD003 training trajectories: {datasets['windows'][train]['n_windows']} windows from 100 engines.",
         f"- FD003 official test: 100 engines, one window each ({counts['near_failure']} with true RUL<=50, "
         f"{counts['intermediate']} with 51-100, {counts['early_life']} above 100).", "",
         "The FD001 official-test arrays were not opened. FD001 test figures quoted below are the rows recorded in "
         "Sessions 3-6 and had already been inspected.", "",
         "## 3. Frozen models and subsets", "",
         "The 174 canonical Session 7 models (6 architectures; 14 sensors and ranked or random subsets of 10, 7, 4, 2) "
         "were used for inference only, after each reproduced its stored FD001 predictions exactly. No model used for "
         "the results was trained or changed. Candidates, fixed from FD001 validation results:", "",
         _table(cfg["candidates"], ["key", "role"]), "",
         "## 4. FD001 baseline", "",
         _table([{"model": NAMES[m], "val": rmse(FULL[m], val), "test": gap[FULL[m]]["fd001_test_rmse_previously_inspected"],
                  "latency": lat(m), "parameters": full_rows[m]["n_parameters"],
                  "size_kb": full_rows[m]["model_size_bytes"] / 1000} for m in by_val],
                ["model", "val", "test", "latency", "parameters", "size_kb"],
                {"val": ".3f", "test": ".3f", "latency": ".1f", "size_kb": ".0f"},
                {"val": "validation RMSE", "test": "test RMSE (already inspected)", "latency": "ms per 256 windows",
                 "parameters": "parameters or tree nodes", "size_kb": "size, kB"}), "",
         "## 5. FD003 generalisation", "",
         _table([{**s[k, test], "val": rmse(k, val), "train": rmse(k, train),
                  "ci": f"{s[k, test]['ci95_low']:.1f} to {s[k, test]['ci95_high']:.1f}"}
                 for k in gap], ["candidate", "val", "train", "rmse", "ci", "mae", "r2", "nasa_score"],
                {"val": ".2f", "train": ".2f", "rmse": ".2f", "mae": ".2f", "r2": ".2f", "nasa_score": ".3g"},
                {"val": "FD001 val RMSE", "train": "FD003 trajectory RMSE", "rmse": "FD003 test RMSE",
                 "ci": "95% interval", "mae": "test MAE", "r2": "test R2", "nasa_score": "test NASA"}), "",
         f"- Every full-sensor model has a higher error on FD003 test than on the FD001 official test, which uses the "
         f"same one-window-per-engine protocol (already inspected in earlier sessions): {min(ratios):.1f} to "
         f"{max(ratios):.1f} times without the TCN (*post hoc* grouping), {gap['tcn_full']['fd003_over_fd001_test_ratio']:.1f} "
         "times for the TCN. FD001 validation and FD003 test RMSE are different protocols and their difference is not a "
         "pure generalisation gap.",
         f"- The TCN fails: it predicts values far above the cap on FD003 (R2 {s['tcn_full', test]['r2']:.1f}).",
         "", "**Architecture order under three protocols** (RMSE, rank in brackets):", "",
         _table([{"model": NAMES[m], **{p: f"{rank[p, m]['value']:.2f} ({rank[p, m]['rank']})" for p in
                                        ("fd001_validation", "fd001_test_previously_inspected", "fd003_test")}}
                 for m in by_val], ["model", "fd001_validation", "fd001_test_previously_inspected", "fd003_test"],
                headers={"fd001_validation": "FD001 validation (all windows)",
                         "fd001_test_previously_inspected": "FD001 test (terminal; already inspected)",
                         "fd003_test": "FD003 test (terminal)"}), "",
         f"Spearman correlation of the six ranks: FD001 test with FD003 test {rho_test:.2f}; FD001 validation with FD003 "
         f"test {rho_val:.2f}; FD001 validation with FD001 test {rho_fd001:.2f}. On the matched terminal protocol the "
         "order is largely similar and random forest is first on both. The difference between the FD001 validation "
         "order and the FD003 order therefore cannot be attributed to domain shift alone: much of it already appears "
         "between FD001 validation and FD001 test. Six models and one seed make these correlations very uncertain "
         "(*post hoc* analysis).", "",
         "- Gap on trajectories (engine-mean RMSE, FD003 training minus FD001 validation, independent engines):", "",
         _table(t["generalization_gap"], ["candidate", "trajectory_engine_rmse_difference", "ci95_low", "ci95_high"],
                {k: "+.2f" for k in ("trajectory_engine_rmse_difference", "ci95_low", "ci95_high")}), "",
         f"![transfer](figures/{FIGURES['transfer']})", "",
         "**Associated input shift (exploratory, post hoc).** FD003 moves several channels far outside the FD001 range, "
         "most clearly sensors 7 and 12, in both the training file and the test windows (figure 32). For every "
         "architecture, reduced subsets that contain either sensor transfer worse than those that do not:",
         "", _table([{"model": NAMES[m], **{f"{x}_{y}": exposure[m, x][f"median_rmse_{y}"]
                                          for x in ("sensor_7", "sensor_12") for y in ("with", "without")}}
                    for m in models], ["model", "sensor_7_with", "sensor_7_without", "sensor_12_with",
                                       "sensor_12_without"],
                    {k: ".1f" for k in ("sensor_7_with", "sensor_7_without", "sensor_12_with", "sensor_12_without")}),
         "", "Subsets differ in several sensors at once, so this is an association, not a demonstrated cause of the TCN "
         "failure or of any other degradation; no controlled intervention was part of the protocol. The shift is "
         "consistent with the second fault mode but cannot be attributed to it without per-engine fault labels.", "",
         f"![shift](figures/{FIGURES['shift']})", "",
         "## 6. Error by RUL stage", "",
         "Stages use the uncapped true RUL (identical bins under the cap). Two aggregations are used and always named: "
         "**pooled** (over all predictions of the stage) and **per engine, then averaged**. FD003 test has one "
         "prediction per engine, so there pooled is the only meaningful form.", "",
         "FD003 official test (one terminal prediction per engine):", "",
         _table([{**stage[FULL[m], test, x], "model": NAMES[m], "stage": stage_name[x]}
                 for m in models for x in stage_name],
                ["model", "stage", "n_engines", "rmse", "mae", "median_abs", "mean_signed", "under_fraction",
                 "over_fraction"],
                {"rmse": ".2f", "mae": ".2f", "median_abs": ".2f", "mean_signed": "+.2f", "under_fraction": ".2f",
                 "over_fraction": ".2f"},
                {"rmse": "RMSE", "mae": "MAE", "median_abs": "median abs error", "mean_signed": "mean signed"}), "",
         "Trajectory sets, per engine and then averaged over engines (FD001 validation: 20 engines; FD003 training: "
         "100):", "",
         _table([{**stage[FULL[m], d, x], "model": NAMES[m], "set": d, "stage": stage_name[x]}
                 for d in (val, train) for m in models for x in stage_name],
                ["set", "model", "stage", "n_engines", "n_predictions", "engine_mean_rmse", "engine_mean_mae",
                 "engine_mean_median_abs", "engine_mean_signed"],
                {"n_predictions": ".0f", "engine_mean_rmse": ".2f", "engine_mean_mae": ".2f",
                 "engine_mean_median_abs": ".2f", "engine_mean_signed": "+.2f"},
                {"engine_mean_rmse": "mean of engine RMSE", "engine_mean_mae": "mean of engine MAE",
                 "engine_mean_median_abs": "mean of engine median abs", "engine_mean_signed": "mean of engine bias"}), "",
         "Pooled values of the trajectory sets are in `artifacts/session8/stage_metrics.csv`.", "",
         "Full against reduced (MAE, paired by engine; bootstrap interval of the mean change; no test):", "",
         _table([{**r, "stage": stage_name[r["stage"]]} for r in t["stage_full_vs_reduced"]],
                ["candidate", "dataset", "stage", "n_engines", "full_mae", "reduced_mae", "mae_change_pct", "ci95_low",
                 "ci95_high"], {"full_mae": ".2f", "reduced_mae": ".2f", "mae_change_pct": "+.0f", "ci95_low": "+.2f",
                                "ci95_high": "+.2f"}), "",
         "**Does sensor reduction affect near-failure accuracy differently from early life?**", "",
         f"- On FD001 validation, ten GRU sensors change little at any stage. Two sensors raise near-failure MAE by "
         f"{compare['gru_k02_perm', val, 'near_failure']['mae_change_pct']:+.0f}% (permutation) and "
         f"{compare['gru_k02_corr', val, 'near_failure']['mae_change_pct']:+.0f}% (correlation), against "
         f"{compare['gru_k02_perm', val, 'early_life']['mae_change_pct']:+.0f}% and "
         f"{compare['gru_k02_corr', val, 'early_life']['mae_change_pct']:+.0f}% in early life. The relative harm is "
         "larger near failure; the absolute increases are of similar size, because the full model starts lower there.",
         f"- On FD003 the direction reverses: the same two-sensor models are far better near failure than the "
         f"full GRU (test MAE {compare['gru_k02_corr', test, 'near_failure']['reduced_mae']:.1f} against "
         f"{compare['gru_k02_corr', test, 'near_failure']['full_mae']:.1f}, "
         f"{compare['gru_k02_corr', test, 'near_failure']['n_engines']} engines).",
         "- So the answer depends on the domain. A conclusion drawn on FD001 alone would be wrong for FD003.", "",
         f"![stages](figures/{FIGURES['stages']})", "", f"![bias](figures/{FIGURES['bias']})", "",
         "## 7. Worst engines", "",
         "Predeclared rule: over the six full-sensor models, rank engines by the mean of absolute error, of late error, "
         "of early error and of near-failure RMSE (for FD003 test, with one prediction, the absolute error); list the top "
         "five. The second view applies the same rule **without the TCN** and was added after the results were seen "
         "(*post hoc*).", "",
         _table([{"dataset": d, "criterion": c,
                  "six": ", ".join(map(str, worst_list(d, c))),
                  "five": ", ".join(map(str, worst_list(d, c, "without_tcn")))}
                 for d in DATASET_LABEL for c in ("absolute_error", "late_error", "early_error", "near_failure")],
                ["dataset", "criterion", "six", "five"],
                headers={"six": "six models (predeclared)", "five": "without TCN (post hoc)"}), "",
         f"One architecture that fails catastrophically dominates a pooled score. On FD003 test the six-model list "
         f"({', '.join(map(str, test_worst))}) and the list without the TCN ({', '.join(map(str, test_worst_no_tcn))}) "
         "share no engine, so the first mainly describes where the TCN fails. The TCN failure itself is real and stays "
         "in every model comparison.", "",
         "Do families fail on the same engines? Spearman correlation of per-engine absolute error over the 15 model "
         "pairs, and the overlap of each pair's ten worst engines:", "",
         _table([{"dataset": d, "median_rho": st.median(r["spearman_rho"] for r in ag[d]),
                  "min_rho": min(r["spearman_rho"] for r in ag[d]), "max_rho": max(r["spearman_rho"] for r in ag[d]),
                  "shared": f"{min(r['shared_worst'] for r in ag[d])} to {max(r['shared_worst'] for r in ag[d])}"}
                 for d in DATASET_LABEL], ["dataset", "median_rho", "min_rho", "max_rho", "shared"],
                {"median_rho": ".2f", "min_rho": ".2f", "max_rho": ".2f"}), "",
         "Difficulty is largely shared (positive correlation for every pair), but not identical.", "",
         "The cycle-wise plots use FD001 validation and FD003 training trajectories. FD003 test gives one prediction "
         "per engine, so no trajectory is drawn from it. The plotted engines follow the six-model rule; choosing "
         "trajectories by outcome is exploratory. Figure 29 orders engines by the six-model mean, which the TCN weighs "
         "heavily.", "",
         *[x for k in ("trajectories", "errors", "sensors", "engines") for x in (f"![{k}](figures/{FIGURES[k]})", "")],
         "## 8. Late versus early predictions", "",
         "error = predicted - true. Positive: the model says more life remains than does (optimistic, the dangerous "
         "direction for maintenance planning). Negative: it says less (conservative, costs early maintenance). The NASA "
         "score uses exp(d/10)-1 for d>=0 and exp(-d/13)-1 for d<0, so an optimistic error costs more.", "",
         "**Pooled over predictions** (used in figure 28 for all three sets, so that they are comparable; long engines "
         "weigh more):", "",
         _table([{**asym[FULL[m], d], "model": NAMES[m], "set": d} for d in DATASET_LABEL for m in models]
                + [{**asym[k, d], "model": k, "set": d} for d in DATASET_LABEL
                   for k in ("gru_k10_perm", "gru_k02_perm", "gru_k02_corr")],
                ["set", "model", "mean_signed", "median_signed", "over_fraction", "mean_late_magnitude",
                 "mean_early_magnitude", "nasa_sum", "nasa_late_share"],
                {"mean_signed": "+.2f", "median_signed": "+.2f", "over_fraction": ".2f", "mean_late_magnitude": ".1f",
                 "mean_early_magnitude": ".1f", "nasa_sum": ".3g", "nasa_late_share": ".3f"}), "",
         "**Per engine, then averaged** (trajectory sets; each engine counts once):", "",
         _table([{**asym[FULL[m], d], "model": NAMES[m], "set": d} for d in (val, train) for m in models],
                ["set", "model", "engine_mean_signed", "engine_mean_over_fraction", "engine_mean_late_magnitude",
                 "engine_mean_early_magnitude", "engine_mean_nasa_late_share"],
                {"engine_mean_signed": "+.2f", "engine_mean_over_fraction": ".2f", "engine_mean_late_magnitude": ".1f",
                 "engine_mean_early_magnitude": ".1f", "engine_mean_nasa_late_share": ".3f"}), "",
         "- FD001 validation: close to balanced in count (about half the predictions late), small positive bias.",
         "- FD003 test: all six full models are mainly optimistic, and more than 90% of each pooled NASA score comes from "
         "late predictions. Near failure the bias is strongly positive.",
         f"- The test NASA sums are concentrated: the five largest engine terms give {min(top5.values()):.0%} to "
         f"{max(top5.values()):.0%} of each full model's total (*post hoc*).",
         "- NASA sums grow with the number of predictions and exponentially with large errors; compare them only within "
         "one set. The TCN's FD003 values reflect a failure, not a meaningful cost.", "",
         f"![asymmetry](figures/{FIGURES['asymmetry']})", "",
         "## 9. Sensitivity", "", "### Ranking method", "",
         _table([{"strategy": k, **v} for k, v in ranked.items()],
                ["strategy", "fd001_val_below", "fd003_train_below", "fd003_test_below", "fd003_test_below_without_tcn",
                 "sig_lower", "sig_higher"], headers={
                    "fd001_val_below": "FD001 val: below random mean (of 24)", "fd003_train_below": "FD003 trajectories",
                    "fd003_test_below": "FD003 test",
                    "fd003_test_below_without_tcn": "FD003 test without TCN, of 20 (post hoc)",
                    "sig_lower": "FD003 test: significantly lower", "sig_higher": "significantly higher"}), "",
         "The conclusion depends on the ranking method. Permutation subsets beat the random mean on all three sets; its "
         "FD001 figure is selection-contaminated, while FD003 labels played no part in choosing it. Correlation subsets "
         "are worse than random on FD001 and mixed on FD003. *Exploratory:* the correlation top-10 contains sensors 7, "
         "12, 20 and 21, which shift in FD003; the permutation top-10 omits 7, 20 and 21 but does contain sensor 12.",
         "",
         "### RUL cap in training", "",
         "Seven configurations refitted at caps 100 and 150, plus a control at 125 that reproduced the canonical "
         "predictions exactly. Scored on FD001 validation windows with true RUL<=100, where targets are identical.", "",
         _table(caps, ["candidate", "training_cap", "common_subset_rmse", "common_subset_mae",
                       "common_subset_mean_signed", "own_cap_rmse", "rank_among_full_models"],
                {k: ".2f" for k in ("common_subset_rmse", "common_subset_mae", "common_subset_mean_signed",
                                    "own_cap_rmse")}), "",
         "Order of the full-sensor models by common-subset RMSE: " + "; ".join(
             f"cap {c}: {', '.join(x.replace('_full', '') for x in v)}" for c, v in cap_order.items()) + ".",
         ("The order is the same under all three caps." if cap_same else
          "The order changes with the cap, so close architecture rankings on FD001 are not robust to it."),
         "Own-cap RMSE values are not comparable across caps. The order also depends on the evaluation subset: at cap 125 "
         "the Transformer is first on all validation windows (section 4) but not on windows with RUL<=100. A lower cap "
         "reduces error on this subset by construction, so the levels do not show that one cap is better. One seed, FD001 "
         "validation only; no cap is recommended or selected, and this does not show robustness in general.", "",
         f"![caps](figures/{FIGURES['caps']})", "",
         "### Evaluation target on FD003 test", "",
         "Scoring against the uncapped official RUL instead (this changes the question, not the training cap):", "",
         _table(t["sensitivity_target"], ["candidate", "fd003_test_rmse_capped_target",
                                          "fd003_test_rmse_uncapped_target", "n_engines_above_cap"],
                {"fd003_test_rmse_capped_target": ".2f", "fd003_test_rmse_uncapped_target": ".2f"}), "",
         "## 10. Statistical evidence", "",
         "Wilcoxon signed-rank on per-engine absolute error, FD003 test, 100 paired engines, zeros dropped, Holm within "
         "family. Difference = first minus second (negative: first is better).", "",
         _table([r for r in t["statistics"] if r["family"] != "C_ranked_vs_random_mean"],
                ["family", "a", "b", "n_engines", "median_difference", "ci95_low", "ci95_high", "rank_biserial",
                 "p_value", "p_holm", "significant_after_correction"],
                {"median_difference": "+.2f", "ci95_low": "+.2f", "ci95_high": "+.2f", "rank_biserial": "+.2f",
                 "p_value": ".2g", "p_holm": ".2g"}), "",
         f"Family C (48 tests, in `artifacts/session8/statistics.csv`): permutation {ranked['permutation']['sig_lower']} of "
         f"24 significantly lower than the random mean and {ranked['permutation']['sig_higher']} higher; correlation "
         f"{ranked['correlation']['sig_lower']} lower and {ranked['correlation']['sig_higher']} higher.", "",
         "**Caveats.** The tests concern per-engine absolute error, not RMSE, and the two can disagree. Wilcoxon assumes "
         "roughly symmetric paired differences, which these skewed differences only approximate. Holm correction was "
         "declared per family. *Post hoc* checks, one Holm correction over all 58 declared tests and an exact sign test:",
         "",
         _table([{**posthoc[r["a"], r["b"]], "family_p": r["p_holm"]} for r in t["statistics"]
                 if r["family"] != "C_ranked_vs_random_mean"],
                ["a", "b", "family_p", "p_holm_all_tests", "sign_test_p"],
                {k: ".2g" for k in ("family_p", "p_holm_all_tests", "sign_test_p")},
                {"family_p": "Holm within family", "p_holm_all_tests": "Holm over 58", "sign_test_p": "sign test"}), "",
         "The two-sensor permutation GRU result survives the joint correction; the GRU against Transformer difference "
         "does not.", "",
         "These tests describe differences across the 100 FD003 engines for one trained instance of each model. They "
         "say nothing about training-seed variation, and comparisons share engines. Non-significance is not equivalence. "
         "No test pairs FD001 with FD003 engines. Session 7's FD001 tests are unchanged.", "",
         "## 11. Efficiency and trade-offs", "",
         _table([{"candidate": c["key"], "sensors": c["budget"], "val": rmse(c["key"], val), "test": rmse(c["key"]),
                  "near": stage[c["key"], test, "near_failure"]["mae"],
                  **{k: st.mean(r[k] for r in master if (r["model"], r["budget"], r["strategy"]) ==
                                (c["model"], c["budget"], c["strategy"]))
                     for k in ("inference_latency_ms_per_batch", "throughput_samples_per_s", "train_time_s",
                               "model_size_bytes", "n_parameters")}} for c in cfg["candidates"]],
                ["candidate", "sensors", "val", "test", "near", "inference_latency_ms_per_batch",
                 "throughput_samples_per_s", "train_time_s", "model_size_bytes", "n_parameters"],
                {"val": ".2f", "test": ".2f", "near": ".1f", "inference_latency_ms_per_batch": ".1f",
                 "throughput_samples_per_s": ".0f", "train_time_s": ".1f", "model_size_bytes": ".0f",
                 "n_parameters": ".0f"},
                {"val": "FD001 val RMSE", "test": "FD003 test RMSE", "near": "FD003 near-failure MAE",
                 "inference_latency_ms_per_batch": "latency ms", "throughput_samples_per_s": "windows/s",
                 "train_time_s": "train s", "model_size_bytes": "size bytes", "n_parameters": "params or nodes"}), "",
         "Latency and throughput are the single Session 7 measurements (see "
         "`session7_followup/power_latency_provenance.md`); nothing was re-timed. They separate families only coarsely "
         "and support no sensor-level speed claim, whatever numerical Pareto membership says. Parameter counts and "
         "tree nodes are not the same kind of complexity.", "",
         f"![tradeoffs](figures/{FIGURES['tradeoffs']})", "",
         "## 12. Conclusions", "", "### The ten questions", "",
         f"1. **Best architecture with 14 sensors.** It depends on the protocol. On FD001 validation the Transformer "
         f"({rmse('transformer_full', val):.2f}; descriptive, selection-affected). On the terminal test sets random forest "
         f"is first on both FD001 (already inspected, {gap['rf_full']['fd001_test_rmse_previously_inspected']:.2f}) and FD003 "
         f"({rmse('rf_full'):.2f}). On FD003 test the Transformer's absolute errors are significantly larger than those of "
         "GRU, gradient boosting, random forest and LSTM (family B, these trained instances).",
         f"2. **Accuracy against inference cost.** On FD001, GRU: {rmse('gru_full', val):.2f} at {lat('gru'):.1f} ms against "
         f"{rmse('transformer_full', val):.2f} at {lat('transformer'):.1f} ms. If transfer matters, gradient boosting is "
         f"more defensible: {rmse('gb_full', val):.2f} on FD001, {rmse('gb_full'):.2f} on FD003, {lat('gradient_boosting'):.1f} ms "
         "(*post hoc* recommendation, made after seeing FD003; these test results cannot also confirm it).",
         f"3. **Smallest budget within 5% on FD001.** Seven sensors, only for the TCN with permutation ranking "
         f"(selection-contaminated, and the TCN's absolute error is the highest). Ten sensors for "
         f"{', '.join(NAMES[m] for m in smallest['permutation'] if m != 'tcn')} with permutation ranking. Without "
         f"ranking, the mean of random draws stays within 5% at ten sensors for "
         f"{', '.join(NAMES[m] for m in smallest['random'])}. No correlation subset does.",
         f"4. **Least sensitive to reduction.** TCN on FD001 (mean increase over ranked subsets "
         f"{mean_increase['tcn']:+.1f}%), a relative measure on the weakest full model; the same architecture fails on "
         "FD003. Relative robustness to sensor removal says nothing about robustness to domain shift.",
         f"5. **Does ranking beat random?** Not in general. Permutation: yes on all three sets "
         f"({ranked['permutation']['fd001_val_below']}/24 contaminated on FD001, "
         f"{ranked['permutation']['fd003_test_below']}/24 on FD003 test). Correlation: no "
         f"({ranked['correlation']['fd001_val_below']}/24 on FD001, {ranked['correlation']['fd003_test_below']}/24 on FD003 test).",
         f"6. **Does the preferred reduced set (GRU, 10, permutation) generalise?** Not well. Its FD003 test RMSE is "
         f"{rmse('gru_k10_perm'):.2f} ({rmse('gru_k10_perm', val):.2f} on FD001 validation, a different protocol). It is "
         f"not distinguishable from the full GRU "
         f"({rmse('gru_full'):.2f}; adjusted p = {A('gru_k10_perm')['p_holm']:.2f}), which also fails to transfer well, and "
         f"its near-failure MAE is higher ({compare['gru_k10_perm', test, 'near_failure']['reduced_mae']:.1f} against "
         f"{compare['gru_k10_perm', test, 'near_failure']['full_mae']:.1f}).",
         f"7. **Largest weaknesses.** On FD003 test, near failure (all six full models strongly optimistic). On FD003 "
         "training trajectories the largest errors of random forest, gradient boosting, LSTM and GRU are instead in the "
         f"intermediate stage. Hardest FD003 test engines: {', '.join(map(str, test_worst_no_tcn))} without the TCN "
         f"(post hoc), {', '.join(map(str, test_worst))} with it.",
         "8. **Optimistic or conservative?** Nearly balanced on FD001 validation; mainly optimistic on FD003 test for every "
         "full model.",
         f"9. **What is statistically supported?** Only these FD003 test differences in per-engine absolute error, for "
         "these trained instances on this dataset: family B (all five against the Transformer, four lower and the TCN "
         "higher); family A for the two-sensor permutation GRU and the seven-sensor TCN against their full models; and "
         f"in family C {ranked['permutation']['sig_lower']} permutation and {ranked['correlation']['sig_lower']} "
         f"correlation comparisons lower, {ranked['correlation']['sig_higher']} correlation higher. The ten-sensor GRU, the "
         "gradient boosting random draws and the two-sensor correlation GRU are **not** shown to differ from their full "
         "models, which is not equivalence. Session 7's FD001 reduced-against-full tests are selection-affected. FD001 "
         "architecture ranking is descriptive.",
         "10. **When do the conclusions stop holding?** Under the FD003 shift; for the TCN on "
         "FD003; possibly under other seeds or caps; and for any real engine.", "",
         "### Defensible conclusions", "",
         f"1. Every full-sensor model has clearly larger errors on FD003 test than on the matched FD001 terminal test. "
         f"The FD001 *validation* order did not carry over to FD003 test (Spearman {rho_val:.2f}), but it did not carry "
         f"over to FD001 test either ({rho_fd001:.2f}), while the two terminal tests order the models similarly "
         f"({rho_test:.2f}). A change of architecture order caused by domain shift is therefore not established.",
         f"2. Fewer sensors can transfer better, but not reliably. The GRU with two permutation-ranked sensors beats the "
         f"full GRU on FD003 test ({rmse('gru_k02_perm'):.2f} against {rmse('gru_full'):.2f}; adjusted p = "
         f"{A('gru_k02_perm')['p_holm']:.3f}), although the same reduction was clearly harmful on FD001. The two-sensor "
         f"correlation set (adjusted p = {A('gru_k02_corr')['p_holm']:.2f}) and ten sensors (GRU and gradient boosting) "
         "show no significant difference, and the seven-sensor TCN's advantage only reflects the failure of the full TCN. "
         f"The two-sensor gain is concentrated near failure (MAE {compare['gru_k02_perm', test, 'near_failure']['reduced_mae']:.1f} "
         f"against {compare['gru_k02_perm', test, 'near_failure']['full_mae']:.1f}); in early life it is slightly worse "
         f"({compare['gru_k02_perm', test, 'early_life']['reduced_mae']:.1f} against "
         f"{compare['gru_k02_perm', test, 'early_life']['full_mae']:.1f}). "
         f"The lowest FD003 test RMSE among all 174 conditions is {best_overall['fd003_test_rmse']:.2f} "
         f"({NAMES[best_overall['model']]}, k={best_overall['budget']}, {best_overall['strategy']}); picking it would be "
         "selection on the test set, so it is descriptive only.",
         "3. On FD003 test the full-sensor models' errors are predominantly optimistic and, in that terminal sample, largest "
         "near failure. The FD003 training trajectories show the optimistic bias less uniformly and do not show "
         "near failure as the worst stage for most models.",
         "4. Permutation-ranked subsets did better than random controls on FD003 as well, but this does not establish a "
         "general ranking advantage: correlation ranking did not, and the mechanism is untested.", "",
         "### Evidence classes", "",
         "| Class | Content |", "|---|---|",
         "| Preselected FD003 evaluation (primary; per local records) | Candidate metrics, families A and B |",
         "| Secondary, planned | Family C, all 174 conditions, trajectory analyses, cap study |",
         "| FD001 validation observations | Sections 4, FD001 rows elsewhere (selection-affected) |",
         "| Selection-contaminated | Every permutation result on FD001 validation |",
         "| Post hoc / exploratory | All narrative and figures; architecture-order analysis; views without the TCN; ratios "
         "without the TCN; sensor-exposure and shift remarks; joint correction, sign tests, NASA concentration; preferring "
         "gradient boosting after FD003 |", "",
         "## 13. Limitations and threats to validity", "",
         "1. **Simulated data**, two related subsets. Removing a model input does not show that a physical sensor can "
         "safely be removed from an aircraft.",
         "2. **One training seed.** Differences of a few cycles may be training randomness; engine-level tests do not "
         "cover it.",
         "3. **Selection on FD001 validation** (architectures, hyperparameters, early stopping, permutation ranking, "
         "candidate choice).",
         "4. **Scaling is part of the transferred model.** Another normalisation or domain adaptation might change every "
         "FD003 result; none was tried, by design.",
         "5. **Different protocols**: FD001 validation windows, FD003 trajectories and FD003 terminal windows differ in "
         "engine count, lifetime and stage mix.",
         "6. **No per-engine fault labels**, so errors cannot be split by fault mode.",
         "7. **Candidates were chosen with all FD001 results known**, including the earlier official-test rows.",
         "8. **Latency** is a single, coarse measurement on one laptop.",
         "9. The cap study covers one dataset, one seed and two alternative caps.",
         "10. **Chronology is self-attested.** The access log, hashes and file times were produced by the same workflow; "
         "nothing was externally time-stamped or committed. The first draft of the protocol (before the cap-125 control "
         "was added) was not preserved, and unlogged reads cannot be excluded by these records.",
         "11. The protocol says the test set is evaluated once; the log shows two label reads. The second rebuilt the "
         "tables from the same cached predictions to check determinism; it did not run inference again and the master "
         "table did not change.",
         "12. Multiple-testing correction is per declared family.",
         "13. The TCN failure dominates any score averaged over all six architectures.",
         "14. FD003 test has now been seen and cannot serve as untouched data.", "",
         "## 14. Reproducibility", "",
         "See [session8_reproducibility.md](session8_reproducibility.md). Tables are in `artifacts/session8/`; "
         "predictions and fits in `results/session8/` (git-ignored).", "",
         "## 15. Recommended future work", "",
         "- Repeat training over several seeds before ranking architectures.",
         "- Select sensors inside engine-level cross-validation, then test once.",
         "- Study normalisation and domain adaptation with a new, untouched evaluation set; FD003 test is now seen.",
         "- Test FD002 and FD004 (varying operating conditions).",
         "- Check whether penalising optimistic errors reduces near-failure bias.",
         "- Time inference repeatedly on controlled hardware.", ""]
    path = resolve_path(cfg["outputs"]["report"])
    path.write_text("\n".join(L))
