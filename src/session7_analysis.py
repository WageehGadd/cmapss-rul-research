"""Aggregation of the Session 7 condition records: master table, degradation, random controls,
Pareto efficiency, per-engine metrics and the predeclared paired statistics. Every number is
computed from the stored records and validation predictions."""
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .metrics import mae, nasa_score, rmse

OK = "completed"


def load_predictions(record):
    return np.load(Path(record["run_dir"]) / "predictions.npz")["validation"]


def master_rows(plan, records):
    """One row per planned condition (missing conditions stay visible with status 'not_run')."""
    rows = []
    for c in plan["conditions"]:
        r = records.get(c["experiment_id"]) or {}
        m = r.get("metrics", {})
        rows.append({
            "experiment_id": c["experiment_id"], "model": c["model"], "family": c["family"], "budget": c["budget"],
            "strategy": c["strategy"], "subset_seed": c["subset_seed"] or "", "training_seed": c["training_seed"],
            "memberships": "|".join(c["memberships"]), "sensors": "|".join(c["sensors"]),
            "sensor_indices": "|".join(map(str, c["sensor_indices"])),
            "n_parameters": r.get("n_parameters", ""), "val_rmse": m.get("rmse", ""), "val_mae": m.get("mae", ""),
            "val_r2": m.get("r2", ""), "val_nasa_score": m.get("nasa_score", ""),
            "train_time_s": r.get("train_time_s", ""),
            "inference_latency_ms_per_batch": r.get("inference_latency_ms_per_batch", ""),
            "inference_latency_ms_per_sample": r.get("inference_latency_ms_per_sample", ""),
            "throughput_samples_per_s": r.get("throughput_samples_per_s", ""),
            "model_size_bytes": r.get("model_size_bytes", ""), "best_epoch": r.get("best_epoch", ""),
            "epochs_run": r.get("epochs_run", ""), "stop_reason": r.get("stop_reason", ""),
            "mlflow_run_id": r.get("mlflow_run_id", "") or "", "run_dir": r.get("run_dir", ""),
            "predictions_sha256": r.get("predictions_sha256", ""), "weights_sha256": r.get("weights_sha256", ""),
            "git_commit": r.get("git_commit", ""), "git_dirty": r.get("git_dirty", ""),
            "status": r.get("status", "not_run"), "attempt": r.get("attempt", ""), "error": r.get("error") or ""})
    return rows


def full_references(rows):
    return {r["model"]: r for r in rows if r["strategy"] == "full" and r["status"] == OK}


def relative_rmse_increase_pct(reduced, full):
    """100 * (RMSE_reduced - RMSE_full) / RMSE_full. Negative means the subset did better."""
    return 100.0 * (reduced - full) / full


def degradation_rows(rows):
    """Changes of every completed condition against the same model's Session 7 full-sensor run."""
    full = full_references(rows)
    out = []
    for r in rows:
        ref = full.get(r["model"])
        if r["status"] != OK or ref is None:
            continue
        removed = ref["budget"] - r["budget"]
        out.append({
            "experiment_id": r["experiment_id"], "model": r["model"], "budget": r["budget"], "strategy": r["strategy"],
            "subset_seed": r["subset_seed"], "sensors_removed": removed,
            "sensor_reduction_pct": 100.0 * removed / ref["budget"], "val_rmse": r["val_rmse"],
            "full_rmse": ref["val_rmse"],
            "relative_rmse_increase_pct": relative_rmse_increase_pct(r["val_rmse"], ref["val_rmse"]),
            "mae_change": r["val_mae"] - ref["val_mae"], "nasa_change": r["val_nasa_score"] - ref["val_nasa_score"],
            "latency_change_pct": 100.0 * (r["inference_latency_ms_per_batch"] / ref["inference_latency_ms_per_batch"] - 1),
            "throughput_change_pct": 100.0 * (r["throughput_samples_per_s"] / ref["throughput_samples_per_s"] - 1),
            "model_size_change_bytes": r["model_size_bytes"] - ref["model_size_bytes"]})
    return out


def summarise(values):
    v = np.asarray(list(values), dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max())}


def random_control_rows(rows):
    """Summary of the random-subset runs of each model and budget (never shown as a single run)."""
    out = []
    groups = {}
    for r in rows:
        if r["strategy"] == "random" and r["status"] == OK:
            groups.setdefault((r["model"], r["budget"]), []).append(r)
    full = full_references(rows)
    for (model, budget), members in sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        s = summarise(m["val_rmse"] for m in members)
        row = {"model": model, "budget": budget, "n_draws": s["n"], **{f"rmse_{k}": s[k] for k in
               ("mean", "median", "std", "min", "max")}}
        if model in full:
            row["relative_increase_pct_mean"] = relative_rmse_increase_pct(s["mean"], full[model]["val_rmse"])
        row["mae_mean"] = float(np.mean([m["val_mae"] for m in members]))
        row["nasa_mean"] = float(np.mean([m["val_nasa_score"] for m in members]))
        out.append(row)
    return out


def ranked_vs_random_rows(rows):
    """For each model, budget and ranked strategy: position against the random draws."""
    random = {}
    for r in rows:
        if r["strategy"] == "random" and r["status"] == OK:
            random.setdefault((r["model"], r["budget"]), []).append(r["val_rmse"])
    out = []
    for r in rows:
        draws = random.get((r["model"], r["budget"]))
        if r["strategy"] not in ("correlation", "permutation") or r["status"] != OK or not draws:
            continue
        out.append({"model": r["model"], "budget": r["budget"], "strategy": r["strategy"], "ranked_rmse": r["val_rmse"],
                    "random_mean_rmse": float(np.mean(draws)), "random_min_rmse": float(min(draws)),
                    "random_max_rmse": float(max(draws)), "difference_from_random_mean": r["val_rmse"] - float(np.mean(draws)),
                    "n_draws": len(draws), "random_draws_better": int(sum(d < r["val_rmse"] for d in draws))})
    return sorted(out, key=lambda x: (x["model"], -x["budget"], x["strategy"]))


def dominates(a, b, objectives):
    """a dominates b: at least as good (<=) in every objective and strictly better in one."""
    return all(a[o] <= b[o] for o in objectives) and any(a[o] < b[o] for o in objectives)


def pareto_front(rows, objectives):
    """Experiment IDs of the non-dominated completed conditions (all objectives minimised)."""
    pool = [r for r in rows if r["status"] == OK]
    return [r["experiment_id"] for r in pool if not any(dominates(o, r, objectives) for o in pool if o is not r)]


def per_engine_metrics(y_true, y_pred, unit_ids):
    """Aggregate the overlapping windows of each engine into one row per engine."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    out = []
    for unit in np.unique(unit_ids):
        mask = unit_ids == unit
        out.append({"unit_id": int(unit), "n_windows": int(mask.sum()), "rmse": rmse(y_true[mask], y_pred[mask]),
                    "mae": mae(y_true[mask], y_pred[mask]), "bias": float(np.mean(y_pred[mask] - y_true[mask])),
                    "nasa_score": nasa_score(y_true[mask], y_pred[mask])})
    return out


def paired_test(differences, *, resamples=10000, seed=0):
    """Wilcoxon signed-rank test on paired per-engine differences, with effect sizes and a
    bootstrap interval (engines resampled) for the median difference."""
    d = np.asarray(differences, dtype=float)
    nonzero = d[d != 0]
    if nonzero.size:
        statistic, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        ranks = rankdata(np.abs(nonzero))
        effect = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    else:
        statistic, p, effect = 0.0, 1.0, 0.0
    rng = np.random.default_rng(seed)
    medians = np.median(d[rng.integers(0, d.size, size=(resamples, d.size))], axis=1)
    return {"n_engines": int(d.size), "n_nonzero": int(nonzero.size), "median_difference": float(np.median(d)),
            "mean_difference": float(d.mean()), "ci95_low": float(np.quantile(medians, 0.025)),
            "ci95_high": float(np.quantile(medians, 0.975)), "engines_worse": int((d > 0).sum()),
            "engines_better": int((d < 0).sum()), "statistic": float(statistic), "p_value": float(p),
            "rank_biserial": effect}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted, running = np.empty_like(p), 0.0
    for i, idx in enumerate(order):
        running = max(running, (p.size - i) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def planned_comparisons(rows, engine_rmse, settings, planned_models, budgets):
    """The two predeclared comparison families. engine_rmse: {experiment_id: {unit_id: rmse}}.
    A comparison whose inputs are incomplete is listed as unavailable, never silently dropped."""
    by_key = {(r["model"], r["budget"], r["strategy"], r["subset_seed"]): r for r in rows if r["status"] == OK}
    full = full_references(rows)
    seeds = sorted({r["subset_seed"] for r in rows if r["strategy"] == "random"})
    out = []
    for family in settings["families"]:
        members = []
        for model in planned_models:
            for k in budgets:
                for strategy in ("correlation", "permutation"):
                    entry = {"family": family, "model": model, "budget": k, "strategy": strategy}
                    ranked = by_key.get((model, k, strategy, ""))
                    if family == "reduced_ranked_vs_full":
                        others = [full[model]] if model in full else []
                        needed = 1
                    else:
                        others = [by_key[(model, k, "random", s)] for s in seeds if (model, k, "random", s) in by_key]
                        needed = len(seeds)
                    if ranked is None or len(others) != needed or needed == 0:
                        members.append({**entry, "available": False})
                        continue
                    units = sorted(engine_rmse[ranked["experiment_id"]])
                    reference = np.mean([[engine_rmse[o["experiment_id"]][u] for u in units] for o in others], axis=0)
                    diff = np.array([engine_rmse[ranked["experiment_id"]][u] for u in units]) - reference
                    members.append({**entry, "available": True,
                                    **paired_test(diff, resamples=int(settings["bootstrap_resamples"]),
                                                  seed=int(settings["bootstrap_seed"]))})
        available = [m for m in members if m["available"]]
        for m, adjusted in zip(available, holm_adjust([m["p_value"] for m in available])):
            m["p_holm"] = adjusted
            m["significant_after_correction"] = bool(adjusted < float(settings["alpha"]))
        for m in members:
            m["family_size"] = len(available)
        out.extend(members)
    return out


def robustness_rows(rows, tolerance_pct, budgets):
    """Per model: ranked-subset RMSE and relative increase at each budget, and the smallest budget
    that stays within the predeclared tolerance of the full-sensor run."""
    full = full_references(rows)
    out = []
    for model, ref in full.items():
        for strategy in ("correlation", "permutation", "random"):
            per_budget = {}
            for k in budgets:
                values = [r["val_rmse"] for r in rows if r["model"] == model and r["budget"] == k
                          and r["strategy"] == strategy and r["status"] == OK]
                if values:
                    per_budget[k] = relative_rmse_increase_pct(float(np.mean(values)), ref["val_rmse"])
            if not per_budget:
                continue
            within = [k for k, v in per_budget.items() if v <= tolerance_pct]
            out.append({"model": model, "strategy": strategy, "full_rmse": ref["val_rmse"],
                        **{f"increase_pct_k{k}": per_budget.get(k, "") for k in budgets},
                        "mean_increase_pct": float(np.mean(list(per_budget.values()))),
                        "smallest_budget_within_tolerance": min(within) if within else "",
                        "n_budgets": len(per_budget)})
    return out
