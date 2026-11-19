"""Load and validate the Session 3 configuration.

Session 3 reuses the Session 2 benchmark engine. `to_benchmark_config` combines the Session 2
configuration (timing protocol, MLflow store, results table) with the Session 3 settings into
the BenchmarkConfig that `run_experiment` expects, so classical models are logged to the same
experiment and appended to the same results table.
"""
import copy
from pathlib import Path

import yaml

from .benchmark_config import BenchmarkConfig, load_benchmark_config, resolve_path
from .classical_models import FAMILIES
from .tabular_features import FEATURES

_REQUIRED = {
    "session1": ["config", "processed_dir"],
    "session2": ["config"],
    "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
    "features": ["temporal"],
    "execution": ["n_jobs_train", "n_jobs_predict"],
    "models": list(FAMILIES),
    "selection": ["metric", "tie_tolerance_rel"],
    "ranking": ["correlation", "permutation", "top_disagreements"],
    "budgets": ["sensor_counts", "random_control_seeds"],
    "outputs": ["frozen_dir", "working_dir", "report", "figures_dir"],
}


def load_session3_config(config_path):
    config_path = Path(config_path).resolve()
    text = config_path.read_text()
    cfg = yaml.safe_load(text)
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Config {config_path} is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config {config_path} is missing '{section}.{key}'")

    temporal = cfg["features"]["temporal"]
    if not temporal or set(temporal) - set(FEATURES) or len(set(temporal)) != len(temporal):
        raise ValueError(f"features.temporal must use only {FEATURES}")
    for family in FAMILIES:
        model = cfg["models"][family]
        for key in ("name", "search_space"):
            if key not in model:
                raise ValueError(f"models.{family}.{key} is missing")
        if not all(isinstance(v, list) and v for v in model["search_space"].values()):
            raise ValueError(f"models.{family}.search_space must map each parameter to a non-empty list")
    if cfg["selection"]["metric"] not in ("rmse", "mae", "nasa_score"):
        raise ValueError("selection.metric must be one of rmse, mae, nasa_score (lower is better)")
    if cfg["ranking"]["permutation"]["source_model"] not in FAMILIES:
        raise ValueError(f"ranking.permutation.source_model must be one of {FAMILIES}")
    if cfg["ranking"]["permutation"]["aggregation"] not in ("sum", "mean"):
        raise ValueError("ranking.permutation.aggregation must be 'sum' or 'mean'")
    if cfg["ranking"]["correlation"]["feature"] not in temporal:
        raise ValueError("ranking.correlation.feature must be one of features.temporal")
    counts = cfg["budgets"]["sensor_counts"]
    if sorted(set(counts), reverse=True) != list(counts) or min(counts) < 1:
        raise ValueError("budgets.sensor_counts must be unique, positive and descending")
    if not cfg["budgets"]["random_control_seeds"]:
        raise ValueError("budgets.random_control_seeds must not be empty")
    return cfg, text, config_path


def to_benchmark_config(cfg, text, config_path):
    """Session 2 config + Session 3 overrides, in the form run_experiment expects."""
    base = load_benchmark_config(resolve_path(cfg["session2"]["config"]))
    values = copy.deepcopy(base.values)
    values["session1"] = copy.deepcopy(cfg["session1"])
    values["benchmark"] = copy.deepcopy(cfg["benchmark"])
    return BenchmarkConfig(values=values, path=config_path, text=text)
