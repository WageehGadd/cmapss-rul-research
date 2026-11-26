"""Session 8 analysis functions: metrics per engine and per RUL stage, error asymmetry, worst
engines, bootstrap intervals and the declared paired tests. Pure functions of arrays.

error = predicted - true (cycles); positive = optimistic (late), negative = conservative (early).
Overlapping windows are never treated as independent: every uncertainty statement and test works on
one value per engine.
"""
import numpy as np

from .metrics import compute_metrics, nasa_score
from .session7_analysis import holm_adjust, paired_test


def stage_of(raw_rul, stages):
    """Stage name of each true uncapped RUL value."""
    raw = np.asarray(raw_rul, dtype=float)
    if raw.size == 0 or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("raw RUL must be non-empty, finite and non-negative")
    out = np.empty(raw.shape, dtype=object)
    for stage in stages:
        high = np.inf if stage["high"] is None else stage["high"]
        out[(raw >= stage["low"]) & (raw <= high)] = stage["name"]
    if any(v is None for v in out):
        raise ValueError("a RUL value belongs to no stage (non-integer between bins?)")
    return out


def signed_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("targets and predictions must be non-empty and of equal length")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("targets and predictions must be finite")
    return y_pred - y_true


def error_summary(y_true, y_pred):
    """Summary of one group of predictions (one engine, or one terminal prediction per engine)."""
    d = signed_error(y_true, y_pred)
    late, early = d[d > 0], d[d < 0]
    return {"n_predictions": int(d.size), "mae": float(np.abs(d).mean()), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mean_signed": float(d.mean()), "median_signed": float(np.median(d)),
            "median_abs": float(np.median(np.abs(d))), "over_fraction": float((d > 0).mean()),
            "under_fraction": float((d < 0).mean()), "late_part": float(np.maximum(d, 0).mean()),
            "early_part": float(np.maximum(-d, 0).mean()),
            "mean_late_magnitude": float(late.mean()) if late.size else 0.0,
            "mean_early_magnitude": float(-early.mean()) if early.size else 0.0,
            "nasa_sum": float(nasa_score(y_true, y_pred)),
            "nasa_mean": float(nasa_score(y_true, y_pred) / d.size),
            "nasa_late_share": _late_share(d)}


def _late_share(d):
    """Share of the NASA score that comes from optimistic (late) predictions."""
    terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(terms[d >= 0].sum() / terms.sum()) if terms.sum() > 0 else 0.0


def per_engine(y_true, y_pred, unit_ids, y_raw, near_failure_max=50):
    """{unit_id: summary} for one model; `near_rmse` is NaN when the engine has no window there."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    unit_ids, y_raw = np.asarray(unit_ids), np.asarray(y_raw, dtype=float)
    if not (y_true.shape == y_pred.shape == unit_ids.shape == y_raw.shape):
        raise ValueError("targets, predictions, units and raw RUL differ in length")
    out = {}
    for unit in np.unique(unit_ids):
        mine = unit_ids == unit
        near = mine & (y_raw <= near_failure_max)
        out[int(unit)] = {**error_summary(y_true[mine], y_pred[mine]),
                          "near_rmse": float(np.sqrt(np.mean((y_pred[near] - y_true[near]) ** 2)))
                          if near.any() else float("nan")}
    return out


# Two aggregations, always named. Plain names ("rmse", "mae", "median_abs", ...) are POOLED: computed over
# every prediction of the group, so long engines weigh more. "engine_*" names are computed per engine and
# then averaged over engines.
MEAN_OVER_ENGINES = {"rmse": "engine_mean_rmse", "mae": "engine_mean_mae", "median_abs": "engine_mean_median_abs",
                     "mean_signed": "engine_mean_signed", "median_signed": "engine_mean_median_signed",
                     "over_fraction": "engine_mean_over_fraction", "under_fraction": "engine_mean_under_fraction",
                     "late_part": "engine_mean_late_part", "early_part": "engine_mean_early_part",
                     "mean_late_magnitude": "engine_mean_late_magnitude",
                     "mean_early_magnitude": "engine_mean_early_magnitude", "nasa_mean": "engine_mean_nasa",
                     "nasa_late_share": "engine_mean_nasa_late_share"}
# With one prediction per engine these per-engine quantities are not what their names say (a mean of
# one-sample RMSEs is the MAE), so they are left empty.
UNDEFINED_FOR_ONE_PREDICTION = ("engine_mean_rmse", "engine_mean_median_abs", "engine_mean_median_signed",
                                "engine_mean_nasa_late_share")
POOLED = ("rmse", "mae", "median_abs", "mean_signed", "median_signed", "over_fraction", "under_fraction",
          "mean_late_magnitude", "mean_early_magnitude", "nasa_sum", "nasa_mean", "nasa_late_share")


def across_engines(engines):
    """Engine-weighted means of the per-engine summaries, under their `engine_*` names."""
    if not engines:
        raise ValueError("no engines")
    out = {"n_engines": len(engines), "n_predictions": int(sum(e["n_predictions"] for e in engines.values())),
           **{name: float(np.mean([e[k] for e in engines.values()])) for k, name in MEAN_OVER_ENGINES.items()}}
    if all(e["n_predictions"] == 1 for e in engines.values()):
        out.update({name: "" for name in UNDEFINED_FOR_ONE_PREDICTION})
    return out


def both_aggregations(y_true, y_pred, engines):
    """Pooled (prediction-weighted) and per-engine-averaged summaries of one group."""
    pooled = error_summary(y_true, y_pred)
    return {**{k: pooled[k] for k in POOLED}, **across_engines(engines)}


def dataset_metrics(y_true, y_pred, unit_ids, y_raw):
    """Benchmark metrics over all predictions (the Session 1-7 definition) plus engine-weighted ones."""
    engines = per_engine(y_true, y_pred, unit_ids, y_raw)
    mean = across_engines(engines)
    return {**compute_metrics(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)),
            "n_predictions": mean["n_predictions"], "n_engines": mean["n_engines"],
            "engine_mean_rmse": mean["engine_mean_rmse"], "engine_mean_mae": mean["engine_mean_mae"]}, engines


def stage_metrics(y_true, y_pred, unit_ids, y_raw, stages):
    """One row per RUL stage with both aggregations and the counts."""
    labels = stage_of(y_raw, stages)
    y_true, y_pred, unit_ids = (np.asarray(a) for a in (y_true, y_pred, unit_ids))
    rows = []
    for stage in stages:
        mine = labels == stage["name"]
        if not mine.any():
            rows.append({"stage": stage["name"], "n_engines": 0, "n_predictions": 0})
            continue
        engines = per_engine(y_true[mine], y_pred[mine], unit_ids[mine], np.asarray(y_raw)[mine])
        rows.append({"stage": stage["name"], **both_aggregations(y_true[mine], y_pred[mine], engines),
                     "engine_values": engines})
    return rows


def mean_rows(rows):
    """Average several summaries of the same shape (the five random draws of one candidate)."""
    if not rows:
        raise ValueError("nothing to average")
    return {k: (float(np.mean([r[k] for r in rows])) if isinstance(rows[0][k], float) else rows[0][k])
            for k in rows[0] if k != "engine_values"}


def bootstrap_ci(values, resamples, seed, statistic="rmse"):
    """95% interval over engines. `values` is one number per engine: absolute errors for 'rmse'
    (root of the resampled mean square) or any quantity for 'mean'."""
    v = np.asarray(values, dtype=float)
    if v.size < 2 or not np.isfinite(v).all():
        raise ValueError("a bootstrap interval needs at least two finite engine values")
    idx = np.random.default_rng(seed).integers(0, v.size, size=(resamples, v.size))
    stat = np.sqrt((v[idx] ** 2).mean(axis=1)) if statistic == "rmse" else v[idx].mean(axis=1)
    return float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))


def bootstrap_mean_rmse(errors, resamples, seed):
    """95% interval, engines resampled, of the MEAN OVER MODELS OF EACH MODEL'S RMSE. `errors` is
    [n_models, n_engines], one prediction per engine. With one model this is the RMSE interval."""
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2 or e.shape[1] < 2 or not np.isfinite(e).all():
        raise ValueError("errors must be finite [models, engines] with at least two engines")
    idx = np.random.default_rng(seed).integers(0, e.shape[1], size=(resamples, e.shape[1]))
    stat = np.mean([np.sqrt((m[idx] ** 2).mean(axis=1)) for m in e], axis=0)
    return float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))


def paired_family(name, comparisons, settings):
    """Declared comparisons, Holm-corrected together. comparisons: [(labels, a, b)] where a and b are
    {unit_id: value}; the difference is a - b (positive: a has the larger error)."""
    members = []
    for labels, a, b in comparisons:
        if set(a) != set(b) or not a:
            raise ValueError(f"{name}: paired values must cover the same engines")
        units = sorted(a)
        difference = np.array([a[u] - b[u] for u in units])
        members.append({"family": name, **labels,
                        **paired_test(difference, resamples=int(settings["bootstrap_resamples"]),
                                      seed=int(settings["bootstrap_seed"]))})
    for m, adjusted in zip(members, holm_adjust([m["p_value"] for m in members])):
        m["p_holm"] = adjusted
        m["significant_after_correction"] = bool(adjusted < float(settings["alpha"]))
        m["family_size"] = len(members)
    return members


def worst_engines(engine_tables, n):
    """Frozen worst-engine rule over several models: engine score = mean over models.
    engine_tables: {model: {unit_id: summary}}. Returns rows ordered by criterion and rank."""
    criteria = {"absolute_error": "mae", "late_error": "late_part", "early_error": "early_part",
                "near_failure": "near_rmse"}
    units = sorted(next(iter(engine_tables.values())))
    if any(sorted(t) != units for t in engine_tables.values()):
        raise ValueError("models were not evaluated on the same engines")
    rows = []
    for criterion, field in criteria.items():
        scores = {u: float(np.mean([t[u][field] for t in engine_tables.values()])) for u in units}
        eligible = [u for u in units if np.isfinite(scores[u]) and scores[u] > 0]
        for rank, unit in enumerate(sorted(eligible, key=lambda u: (-scores[u], u))[:n], 1):
            rows.append({"criterion": criterion, "rank": rank, "unit_id": unit, "score": scores[unit],
                         **{f"score_{m}": t[unit][field] for m, t in engine_tables.items()}})
    return rows


def worst_engine_views(engine_tables, n, excluded="tcn"):
    """The predeclared rule over all models, and the same rule without one model. The second view was
    added after the results were seen: one architecture that fails badly can dominate a pooled score."""
    views = [("all_full_models", "predeclared", engine_tables),
             (f"without_{excluded}", "post_hoc", {m: t for m, t in engine_tables.items() if m != excluded})]
    return [{"view": view, "status": status, "models": "|".join(tables), **row}
            for view, status, tables in views for row in worst_engines(tables, n)]


def family_agreement(engine_tables, overlap_n):
    """Do the model families fail on the same engines? Pairwise Spearman correlation of per-engine
    absolute error and the overlap of each model's `overlap_n` worst engines."""
    from scipy.stats import spearmanr

    models = list(engine_tables)
    units = sorted(engine_tables[models[0]])
    errors = {m: np.array([engine_tables[m][u]["mae"] for u in units]) for m in models}
    worst = {m: set(np.array(units)[np.argsort(-errors[m], kind="stable")[:overlap_n]].tolist()) for m in models}
    return [{"model_a": a, "model_b": b, "spearman_rho": float(spearmanr(errors[a], errors[b])[0]),
             "shared_worst": len(worst[a] & worst[b]), "worst_set_size": min(overlap_n, len(units))}
            for i, a in enumerate(models) for b in models[i + 1:]]
