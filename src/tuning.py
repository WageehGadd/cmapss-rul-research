"""Compact, pre-declared hyperparameter search with validation-only selection.

The grid comes from the Session 3 config. Every candidate is fitted on the TRAINING features
and scored on the VALIDATION features with the shared Session 2 metrics. The functions here
receive training and validation arrays only: the official test set cannot influence selection.

Selection rule (declared in the config):
  1. rank candidates by the validation `selection_metric` (lower is better, rmse);
  2. every candidate within `tie_tolerance_rel` (relative) of the best is "equivalent";
  3. among those, choose the smallest model (fewest `n_parameters`), then the first in grid order.
"""
import itertools
import time

import numpy as np

from .metrics import compute_metrics


def expand_grid(grid):
    """{param: [values]} -> list of {param: value}, in deterministic (declaration) order."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def search(build_candidate, grid, F_train, y_train, F_val, y_val):
    """Fit every candidate on train, score on validation. Returns a list of records."""
    records = []
    for index, params in enumerate(expand_grid(grid)):
        model = build_candidate(params)
        start = time.perf_counter()
        model.fit_features(F_train, y_train)
        fit_s = time.perf_counter() - start
        metrics = compute_metrics(y_val, model.predict_features(F_val))
        records.append({"candidate": index, "params": params, "validation": metrics,
                        "n_parameters": model.n_parameters(), "fit_time_s": fit_s})
    return records


def select(records, metric="rmse", tie_tolerance_rel=0.0):
    """Apply the selection rule to the search records; returns (chosen record, rule description)."""
    if not records:
        raise ValueError("no candidates to select from")
    best = min(r["validation"][metric] for r in records)
    equivalent = [r for r in records if r["validation"][metric] <= best * (1.0 + tie_tolerance_rel)]
    chosen = min(equivalent, key=lambda r: (r["n_parameters"] if r["n_parameters"] is not None else np.inf,
                                            r["candidate"]))
    rule = {"metric": metric, "best_validation_value": best, "tie_tolerance_rel": tie_tolerance_rel,
            "n_equivalent": len(equivalent), "tiebreak": "fewest n_parameters, then grid order"}
    return chosen, rule
