"""Leakage-safe sensor-level rankings, ranking comparison and frozen sensor subsets.

Hard gate: every function takes training and/or validation arrays only. Nothing here can see
the official test set.

Ranking A (correlation, TRAINING data only)
    For each sensor, Pearson correlation between the sensor value at the last cycle of each
    training window (feature `last`; one sample per window, i.e. per engine cycle from cycle
    `sequence_length` on) and the capped RUL target of that window. Score = signed r; ranked
    by |r| descending.

Ranking B (permutation importance, VALIDATION data only)
    Feature-level permutation importance of a fitted classical model on the validation features:
    the increase in validation RMSE (shared Session 2 metric) when one tabular feature column
    is shuffled, averaged over `n_repeats` seeded shuffles. The six feature importances of each
    sensor are aggregated to the sensor with `aggregation` (sum by default). Ranked descending.

Ties are broken by the sensor's position in the Session 1 sensor list, so rankings are
fully deterministic.
"""
import numpy as np
from scipy.stats import spearmanr

from .metrics import rmse


def _rank_descending(scores):
    """1-based ranks, highest score = rank 1, ties broken by input position."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for position, i in enumerate(order, start=1):
        ranks[i] = position
    return ranks


def correlation_ranking(F_train, y_train, transformer, feature="last"):
    """Ranking A. F_train: training tabular features [n, n_features] from `transformer`."""
    if feature not in transformer.features:
        raise ValueError(f"correlation feature '{feature}' is not a configured temporal feature")
    y = np.asarray(y_train, dtype=np.float64)
    if len(F_train) != len(y):
        raise ValueError("training features and targets differ in length")
    rows = []
    for sensor, columns in transformer.sensor_to_columns().items():
        x = F_train[:, columns[transformer.features.index(feature)]]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            r = 0.0
        else:
            r = float(np.corrcoef(x, y)[0, 1])
        rows.append({"sensor": sensor, "correlation": r, "abs_correlation": abs(r)})
    ranks = _rank_descending([row["abs_correlation"] for row in rows])
    for row, rank in zip(rows, ranks):
        row["rank"] = rank
    return sorted(rows, key=lambda row: row["rank"])


def feature_permutation_importance(model, F_val, y_val, n_repeats, seed):
    """Mean RMSE increase per feature column when that column is shuffled (validation only).

    Returns (importance_mean [n_features], importance_std [n_features], baseline_rmse).
    """
    rng = np.random.default_rng(seed)
    baseline = rmse(y_val, model.predict_features(F_val))
    n_features = F_val.shape[1]
    deltas = np.empty((n_features, n_repeats))
    work = F_val.copy()
    for j in range(n_features):
        original = F_val[:, j].copy()
        for r in range(n_repeats):
            work[:, j] = original[rng.permutation(len(original))]
            deltas[j, r] = rmse(y_val, model.predict_features(work)) - baseline
        work[:, j] = original
    return deltas.mean(axis=1), deltas.std(axis=1), float(baseline)


def aggregate_to_sensors(importance, transformer, method="sum"):
    """Feature-level importances [n_features] -> per-sensor {sensor: (aggregate, components)}."""
    if method not in ("sum", "mean"):
        raise ValueError("aggregation must be 'sum' or 'mean'")
    importance = np.asarray(importance, dtype=np.float64)
    if importance.shape != (transformer.n_features,):
        raise ValueError(f"expected {transformer.n_features} importances, got {importance.shape}")
    out = {}
    for sensor, columns in transformer.sensor_to_columns().items():
        values = importance[columns]
        total = float(values.sum() if method == "sum" else values.mean())
        out[sensor] = (total, {f: float(v) for f, v in zip(transformer.features, values)})
    return out


def permutation_ranking(model, F_val, y_val, transformer, n_repeats, seed, aggregation="sum"):
    """Ranking B from a fitted model and VALIDATION features/targets."""
    mean, std, baseline = feature_permutation_importance(model, F_val, y_val, n_repeats, seed)
    per_sensor = aggregate_to_sensors(mean, transformer, aggregation)
    rows = [{"sensor": s, "importance": total, "components": components}
            for s, (total, components) in per_sensor.items()]
    for row, rank in zip(rows, _rank_descending([r["importance"] for r in rows])):
        row["rank"] = rank
    detail = {"feature_names": transformer.feature_names, "importance_mean": mean.tolist(),
              "importance_std": std.tolist(), "baseline_validation_rmse": baseline}
    return sorted(rows, key=lambda row: row["rank"]), detail


def compare_rankings(ranking_a, ranking_b, budgets, top_n_disagreements=5):
    """Spearman rank correlation, per-sensor rank differences and top-k overlap."""
    rank_a = {r["sensor"]: r["rank"] for r in ranking_a}
    rank_b = {r["sensor"]: r["rank"] for r in ranking_b}
    if set(rank_a) != set(rank_b):
        raise ValueError("rankings cover different sensors")
    sensors = list(rank_a)
    rho, p_value = spearmanr([rank_a[s] for s in sensors], [rank_b[s] for s in sensors])
    differences = [{"sensor": s, "rank_correlation_method": rank_a[s], "rank_permutation_method": rank_b[s],
                    "rank_difference": rank_b[s] - rank_a[s], "abs_rank_difference": abs(rank_b[s] - rank_a[s])}
                   for s in sensors]
    differences.sort(key=lambda d: (-d["abs_rank_difference"], sensors.index(d["sensor"])))
    overlap = {}
    for k in budgets:
        top_a = {s for s in sensors if rank_a[s] <= k}
        top_b = {s for s in sensors if rank_b[s] <= k}
        overlap[f"top_{k}"] = {"shared": sorted(top_a & top_b), "only_correlation": sorted(top_a - top_b),
                               "only_permutation": sorted(top_b - top_a),
                               "jaccard": len(top_a & top_b) / len(top_a | top_b)}
    return {"spearman_rho": float(rho), "spearman_p_value": float(p_value), "n_sensors": len(sensors),
            "largest_disagreements": differences[:top_n_disagreements],
            "all_rank_differences": differences, "top_k_overlap": overlap,
            "note": "Neither ranking is declared correct: A is a univariate linear association, "
                    "B is the contribution to one fitted model."}


def top_k_subsets(ranking, budgets, sensor_order):
    """{top_k: {sensors in rank order, sensors in Session 1 order}} for each budget."""
    ordered = [r["sensor"] for r in sorted(ranking, key=lambda r: r["rank"])]
    out = {}
    for k in budgets:
        if not 1 <= k <= len(ordered):
            raise ValueError(f"budget k={k} outside 1..{len(ordered)}")
        chosen = ordered[:k]
        out[f"top_{k}"] = {"k": k, "sensors_by_rank": chosen,
                           "sensors": [s for s in sensor_order if s in chosen]}
    return out


def random_k_subsets(sensor_order, budgets, seeds):
    """Reproducible random control subsets: {random_k: {seed: sensors}} (Session 1 order).

    Drawn from numpy's Generator seeded with (seed, k); performance is never consulted.
    """
    out = {}
    for k in budgets:
        if not 1 <= k <= len(sensor_order):
            raise ValueError(f"budget k={k} outside 1..{len(sensor_order)}")
        draws = {}
        for seed in seeds:
            rng = np.random.default_rng([int(seed), int(k)])
            picked = set(rng.choice(len(sensor_order), size=k, replace=False).tolist())
            draws[str(seed)] = [s for i, s in enumerate(sensor_order) if i in picked]
        out[f"random_{k}"] = {"k": k, "draws": draws}
    return out
