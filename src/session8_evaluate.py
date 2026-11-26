"""Session 8 evaluation: join the stored predictions with the labels and build every result table.
Nothing is trained; the FD003 test labels are read through the guarded ledger."""
import csv
import json

import numpy as np

from .benchmark_config import resolve_path
from .session7_analysis import holm_adjust
from .session8_analysis import (both_aggregations, bootstrap_ci, bootstrap_mean_rmse, dataset_metrics,
                                family_agreement, mean_rows, paired_family, per_engine, stage_metrics,
                                worst_engine_views)
from .session8_data import DATASETS, TEST_SET, TRAJECTORY_SETS, load_test_labels


def targets(cfg, inputs, raw_test_rul=None):
    """{dataset: (capped target, raw true RUL)}. Without test labels the test set is left out."""
    cap = cfg["evaluation"]["rul_cap"]
    raw = {k: np.asarray(inputs[k].y_raw, dtype=float) for k in TRAJECTORY_SETS}
    if raw_test_rul is not None:
        order = np.sort(np.unique(inputs[TEST_SET].unit_ids))
        if not np.array_equal(inputs[TEST_SET].unit_ids, order) or len(raw_test_rul) != len(order):
            raise ValueError("test windows must be one per engine in unit order, matching the RUL file")
        raw[TEST_SET] = np.asarray(raw_test_rul, dtype=float)
    return {k: (np.minimum(v, cap), v) for k, v in raw.items()}


def candidate_members(cfg, conditions):
    """{candidate key: [experiment_id, ...]} from the frozen matrix (five draws for a random one)."""
    out = {}
    for c in cfg["candidates"]:
        ids = sorted(k for k, v in conditions.items()
                     if (v["model"], v["budget"], v["strategy"]) == (c["model"], c["budget"], c["strategy"]))
        expected = len(cfg_random_seeds(conditions)) if c["strategy"] == "random" else 1
        if len(ids) != expected:
            raise ValueError(f"candidate {c['key']} matches {len(ids)} conditions, expected {expected}")
        out[c["key"]] = ids
    return out


def cfg_random_seeds(conditions):
    return sorted({v["subset_seed"] for v in conditions.values() if v["strategy"] == "random"})


def member_engine_values(ids, engines, field="mae"):
    """{unit: value} of a candidate: the mean over its member conditions."""
    units = sorted(engines[ids[0]])
    return {u: float(np.mean([engines[i][u][field] for i in ids])) for u in units}


def evaluate_all(cfg, canonical, inputs, predictions, raw_test_rul):
    """Every table of the study except the cap sensitivity, as plain rows."""
    conditions, records = canonical.conditions, canonical.records
    truth = targets(cfg, inputs, raw_test_rul)
    stages, stats = cfg["evaluation"]["stages"], cfg["statistics"]
    metrics = {d: {} for d in DATASETS}
    engines = {d: {} for d in DATASETS}
    for dataset in DATASETS:
        y, raw = truth[dataset]
        for key in conditions:
            metrics[dataset][key], engines[dataset][key] = dataset_metrics(y, predictions[dataset][key],
                                                                           inputs[dataset].unit_ids, raw)
    for key in conditions:                      # the stored canonical metrics must be reproduced
        for name, value in records[key]["metrics"].items():
            if not np.isclose(metrics["fd001_validation"][key][name], value, rtol=1e-9, atol=1e-9):
                raise RuntimeError(f"{key}: recomputed FD001 {name} differs from the canonical record")

    master = []
    for key, c in sorted(conditions.items()):
        r = records[key]
        row = {"experiment_id": key, "mlflow_run_id": r["mlflow_run_id"], "model": c["model"], "budget": c["budget"],
               "strategy": c["strategy"], "subset_seed": c["subset_seed"] or "",
               "sensors": "|".join(c["sensors"]), "identity_sha256": c["identity_sha256"],
               "selection_contaminated_on_fd001_validation": c["strategy"] == "permutation",
               "n_parameters": r["n_parameters"], "model_size_bytes": r["model_size_bytes"],
               "train_time_s": r["train_time_s"],
               "inference_latency_ms_per_batch": r["inference_latency_ms_per_batch"],
               "throughput_samples_per_s": r["throughput_samples_per_s"]}
        for dataset in DATASETS:
            m = metrics[dataset][key]
            for name in ("rmse", "mae", "r2", "nasa_score", "engine_mean_rmse", "engine_mean_mae", "n_predictions",
                         "n_engines"):
                row[f"{dataset}_{name}"] = m[name]
        row["fd003_train_minus_fd001_val_engine_rmse"] = (row["fd003_train_engine_mean_rmse"]
                                                          - row["fd001_validation_engine_mean_rmse"])
        master.append(row)

    members = candidate_members(cfg, conditions)
    meta = {c["key"]: c for c in cfg["candidates"]}
    historical = {r["model"]: r for r in json.loads(
        resolve_path(cfg["upstream"]["historical_results"]).read_text())["results"]
        if r["eval_set"] == "test" and str(r["git_dirty"]).lower() == "false"}

    summary, candidate_engines = [], []
    seed, resamples = int(stats["bootstrap_seed"]), int(stats["bootstrap_resamples"])
    for key, ids in members.items():
        c = meta[key]
        for dataset in DATASETS:
            row = {"candidate": key, "model": c["model"], "budget": c["budget"], "strategy": c["strategy"],
                   "dataset": dataset, "n_conditions": len(ids), "role": c["role"]}
            for name in ("rmse", "mae", "r2", "nasa_score", "engine_mean_rmse", "engine_mean_mae"):
                values_ = [metrics[dataset][i][name] for i in ids]      # empty: undefined for this dataset
                row[name] = "" if values_[0] == "" else float(np.mean(values_))
            row["n_predictions"] = metrics[dataset][ids[0]]["n_predictions"]
            row["n_engines"] = metrics[dataset][ids[0]]["n_engines"]
            unit = member_engine_values(ids, engines[dataset], "mae" if dataset == TEST_SET else "rmse")
            values = np.array([unit[u] for u in sorted(unit)])
            if dataset == TEST_SET:       # the interval is of the quantity shown: mean over members of RMSE
                low, high = bootstrap_mean_rmse([[engines[dataset][i][u]["mae"] for u in sorted(unit)] for i in ids],
                                                resamples, seed)
            else:
                low, high = bootstrap_ci(values, resamples, seed, "mean")
            row["ci_quantity"] = "rmse" if dataset == TEST_SET else "engine_mean_rmse"
            row["ci95_low"], row["ci95_high"] = low, high
            summary.append(row)
            for u in sorted(unit):
                rows = [engines[dataset][i][u] for i in ids]
                row_ = {"candidate": key, "dataset": dataset, "unit_id": u, **mean_rows(rows)}
                if dataset == TEST_SET:        # one prediction: only the absolute and signed error exist
                    row_.update(rmse="", median_abs="", median_signed="", near_rmse="")
                candidate_engines.append(row_)

    # cross-domain differences (independent engines: never paired)
    rng = np.random.default_rng(seed)
    gap = []
    for key, ids in members.items():
        a = np.array(list(member_engine_values(ids, engines["fd001_validation"], "rmse").values()))
        b = np.array(list(member_engine_values(ids, engines["fd003_train"], "rmse").values()))
        boot = (b[rng.integers(0, b.size, (resamples, b.size))].mean(axis=1)
                - a[rng.integers(0, a.size, (resamples, a.size))].mean(axis=1))
        s = {(r["candidate"], r["dataset"]): r for r in summary}
        row = {"candidate": key,
               "fd001_val_rmse": s[key, "fd001_validation"]["rmse"], "fd003_train_rmse": s[key, "fd003_train"]["rmse"],
               "fd003_test_rmse": s[key, TEST_SET]["rmse"],
               "trajectory_engine_rmse_difference": float(b.mean() - a.mean()),
               "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)),
               "fd001_test_rmse_previously_inspected": "", "fd003_over_fd001_test_ratio": ""}
        if meta[key]["strategy"] == "full":
            old = historical[meta[key]["model"]]["rmse"]
            row["fd001_test_rmse_previously_inspected"] = old
            row["fd003_over_fd001_test_ratio"] = s[key, TEST_SET]["rmse"] / old
        gap.append(row)

    # RUL stages: candidates, and the reduced candidates against their full-sensor model
    stage_rows, stage_values = [], {}
    for key, ids in members.items():
        for dataset in DATASETS:
            y, raw = truth[dataset]
            per_member = [stage_metrics(y, predictions[dataset][i], inputs[dataset].unit_ids, raw, stages) for i in ids]
            for j, stage in enumerate(stages):
                mine = [m[j] for m in per_member]
                row = mean_rows(mine) if mine[0]["n_engines"] else dict(mine[0])
                stage_rows.append({"candidate": key, "dataset": dataset, **row})
                if mine[0]["n_engines"]:
                    units = sorted(mine[0]["engine_values"])
                    stage_values[key, dataset, stage["name"]] = {
                        u: float(np.mean([m["engine_values"][u]["mae"] for m in mine])) for u in units}
    full_key = {c["model"]: c["key"] for c in cfg["candidates"] if c["strategy"] == "full"}
    stage_compare = []
    for key in cfg["statistics"]["families"]["A_reduced_vs_full"]:
        reference = full_key[meta[key]["model"]]
        for dataset in DATASETS:
            for stage in stages:
                a, b = stage_values.get((key, dataset, stage["name"])), stage_values.get((reference, dataset, stage["name"]))
                if a is None:
                    continue
                units = sorted(a)
                d = np.array([a[u] - b[u] for u in units])
                low, high = bootstrap_ci(d, resamples, seed, "mean") if d.size > 1 else (float("nan"),) * 2
                full_mae, reduced_mae = np.mean([b[u] for u in units]), np.mean([a[u] for u in units])
                stage_compare.append({"candidate": key, "reference": reference, "dataset": dataset,
                                      "stage": stage["name"], "n_engines": len(units), "full_mae": float(full_mae),
                                      "reduced_mae": float(reduced_mae),
                                      "mae_change": float(reduced_mae - full_mae),
                                      "mae_change_pct": float(100 * (reduced_mae / full_mae - 1)),
                                      "ci95_low": low, "ci95_high": high,
                                      "engines_worse": int((d > 0).sum()), "engines_better": int((d < 0).sum())})

    # asymmetry: both aggregations, named
    asymmetry = []
    for key, ids in members.items():
        for dataset in DATASETS:
            y, _ = truth[dataset]
            out = mean_rows([both_aggregations(y, predictions[dataset][i], engines[dataset][i]) for i in ids])
            asymmetry.append({"candidate": key, "dataset": dataset,
                              **{k: v for k, v in out.items() if k not in ("rmse", "mae", "median_abs",
                                                                          "engine_mean_rmse", "engine_mean_mae",
                                                                          "engine_mean_median_abs")}})

    # worst engines among the six full-sensor models
    worst, agreement = [], []
    full_ids = {m: members[full_key[m]][0] for m in cfg["error_analysis"]["full_models"]}
    for dataset in DATASETS:
        tables = {m: engines[dataset][i] for m, i in full_ids.items()}
        worst += [{"dataset": dataset, **r} for r in worst_engine_views(tables, cfg["error_analysis"]["worst_n"])]
        agreement += [{"dataset": dataset, **r} for r in family_agreement(tables, cfg["error_analysis"]["overlap_n"])]

    # declared tests on FD003 test, one absolute error per engine
    test = engines[TEST_SET]
    values = lambda key: member_engine_values(members[key], test)
    families = cfg["statistics"]["families"]
    reference = full_key[cfg["evaluation"]["reference_model"]]
    statistics = paired_family("A_reduced_vs_full", [
        ({"a": k, "b": full_key[meta[k]["model"]]}, values(k), values(full_key[meta[k]["model"]]))
        for k in families["A_reduced_vs_full"]], stats)
    statistics += paired_family("B_full_vs_reference", [
        ({"a": k, "b": reference}, values(k), values(reference)) for k in families["B_full_vs_reference"]], stats)
    lookup = {(c["model"], c["budget"], c["strategy"], c["subset_seed"]): k for k, c in conditions.items()}
    seeds = cfg_random_seeds(conditions)
    comparisons, ranking = [], []
    for model in cfg["error_analysis"]["full_models"]:
        for k in sorted({c["budget"] for c in conditions.values() if c["strategy"] != "full"}, reverse=True):
            random_ids = [lookup[model, k, "random", s] for s in seeds]
            for strategy in ("correlation", "permutation"):
                i = lookup[model, k, strategy, None]
                comparisons.append(({"a": f"{model}_k{k:02d}_{strategy}", "b": f"{model}_k{k:02d}_random_mean",
                                     "model": model, "budget": k, "strategy": strategy},
                                    member_engine_values([i], test), member_engine_values(random_ids, test)))
                row = {"model": model, "budget": k, "strategy": strategy}
                for dataset, short in (("fd001_validation", "fd001_val"), (TEST_SET, "fd003_test"),
                                       ("fd003_train", "fd003_train")):
                    draws = [metrics[dataset][r]["rmse"] for r in random_ids]
                    row[f"{short}_ranked_rmse"] = metrics[dataset][i]["rmse"]
                    row[f"{short}_random_mean_rmse"] = float(np.mean(draws))
                    row[f"{short}_random_draws_better"] = int(sum(d < metrics[dataset][i]["rmse"] for d in draws))
                ranking.append(row)
    statistics += paired_family("C_ranked_vs_random_mean", comparisons, stats)

    # evaluation-target sensitivity on the test set (not a training-cap sensitivity)
    y_test, raw_test = truth[TEST_SET]
    target_rows = []
    for key, ids in members.items():
        capped = float(np.mean([metrics[TEST_SET][i]["rmse"] for i in ids]))
        raw = float(np.mean([np.sqrt(np.mean((predictions[TEST_SET][i] - raw_test) ** 2)) for i in ids]))
        target_rows.append({"candidate": key, "fd003_test_rmse_capped_target": capped,
                            "fd003_test_rmse_uncapped_target": raw,
                            "n_engines_above_cap": int((raw_test > cfg["evaluation"]["rul_cap"]).sum())})

    order = architecture_order(cfg, summary, gap)
    checks = posthoc_checks(statistics, engines[TEST_SET], full_ids, y_test, predictions[TEST_SET])

    long_engines = [{"dataset": d, "experiment_id": i, "unit_id": u, **v}
                    for d in DATASETS for i, table in engines[d].items() for u, v in table.items()]
    test_predictions = [{"experiment_id": i, "unit_id": int(u), "end_cycle": int(e), "true_rul_raw": float(r),
                         "target": float(t), "prediction": float(p), "error": float(p - t)}
                        for i in sorted(conditions)
                        for u, e, r, t, p in zip(inputs[TEST_SET].unit_ids, inputs[TEST_SET].end_cycles, raw_test,
                                                 y_test, predictions[TEST_SET][i])]
    for row in stage_rows:
        row.pop("engine_values", None)
    return {"master_results": master, "candidate_summary": summary, "generalization_gap": gap,
            "per_engine_candidates": candidate_engines, "stage_metrics": stage_rows,
            "stage_full_vs_reduced": stage_compare, "asymmetry": asymmetry, "worst_engines": worst,
            "family_agreement": agreement, "statistics": statistics, "sensitivity_ranking": ranking,
            "sensitivity_target": target_rows, "per_engine_all": long_engines,
            "architecture_order": order, "posthoc_checks": checks,
            "fd003_test_predictions": test_predictions,
            "stage_counts_fd003_test": [{"stage": s["name"], "n_engines": int(np.sum(
                (raw_test >= s["low"]) & (raw_test <= (np.inf if s["high"] is None else s["high"]))))}
                for s in stages]}


def architecture_order(cfg, summary, gap):
    """Full-sensor RMSE of the six architectures under three protocols, and the rank correlations.
    FD001 official-test values are the rows recorded in Sessions 3-6 (already inspected)."""
    from scipy.stats import spearmanr

    models = cfg["error_analysis"]["full_models"]
    key = {c["model"]: c["key"] for c in cfg["candidates"] if c["strategy"] == "full"}
    s = {(r["candidate"], r["dataset"]): r["rmse"] for r in summary}
    old = {r["candidate"]: r["fd001_test_rmse_previously_inspected"] for r in gap}
    series = {"fd001_validation": [s[key[m], "fd001_validation"] for m in models],
              "fd001_test_previously_inspected": [old[key[m]] for m in models],
              "fd003_test": [s[key[m], TEST_SET] for m in models]}
    rows = []
    for name, values in series.items():
        rows += [{"kind": "rmse", "protocol": name, "model": m, "value": v,
                  "rank": 1 + sorted(values).index(v)} for m, v in zip(models, values)]
    for a, b in (("fd001_validation", "fd003_test"), ("fd001_test_previously_inspected", "fd003_test"),
                 ("fd001_validation", "fd001_test_previously_inspected")):
        rows.append({"kind": "spearman", "protocol": f"{a} vs {b}", "model": "all six",
                     "value": float(spearmanr(series[a], series[b])[0]), "rank": ""})
    return rows


def posthoc_checks(statistics, test_engines, full_ids, y_test, test_predictions):
    """Checks added after the results were seen: one Holm correction over all declared tests, an exact
    sign test (no symmetry assumption), and how concentrated each NASA score is."""
    from scipy.stats import binomtest

    joint = holm_adjust([r["p_value"] for r in statistics])
    rows = [{"check": "joint_holm_and_sign_test", "family": r["family"], "a": r["a"], "b": r["b"],
             "p_holm_all_tests": p,
             "sign_test_p": float(binomtest(r["engines_worse"], r["engines_worse"] + r["engines_better"]).pvalue)}
            for r, p in zip(statistics, joint)]
    for model, key in full_ids.items():
        d = test_predictions[key] - y_test
        terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
        rows.append({"check": "nasa_top5_share", "a": model, "value": float(np.sort(terms)[-5:].sum() / terms.sum())})
    return rows


def write_rows(path, rows):
    """CSV with the union of the keys, written atomically."""
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    columns = []
    for row in rows:
        columns += [k for k in row if k not in columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path):
    """CSV rows with numbers and booleans restored; empty cells stay ''."""
    def value(text):
        if text in ("True", "False"):
            return text == "True"
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    with open(path, newline="") as f:
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]
