"""Load and validate the Session 4 (recurrent architectures) configuration."""
from pathlib import Path

import yaml

from .recurrent_models import CELLS, DEVICE_CHOICES

_REQUIRED = {
    "session1": ["config", "processed_dir"],
    "session2": ["config"],
    "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
    "training": ["optimizer", "learning_rate", "batch_size", "max_epochs", "early_stopping_patience",
                 "min_delta", "grad_clip", "scheduler", "target_scale"],
    "execution": ["device"],
    "reference": ["family", "hidden_size", "num_layers", "dropout", "sequence_length",
                  "tiny_batch_samples", "tiny_batch_epochs", "tiny_batch_learning_rate", "min_loss_drop"],
    "ablation": ["families", "stage_a", "stage_b"],
    "selection": ["metric", "tie_tolerance_rel", "stage_b_comparison"],
    "outputs": ["frozen_dir", "checkpoint_dir", "working_dir", "report", "figures_dir"],
}


def load_session4_config(config_path):
    config_path = Path(config_path).resolve()
    text = config_path.read_text()
    cfg = yaml.safe_load(text)
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Config {config_path} is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config {config_path} is missing '{section}.{key}'")
    if cfg["training"]["optimizer"] != "adam":
        raise ValueError("training.optimizer must be 'adam'")
    if cfg["execution"]["device"] not in DEVICE_CHOICES:
        raise ValueError(f"execution.device must be one of {DEVICE_CHOICES}")
    if not cfg["ablation"]["families"] or set(cfg["ablation"]["families"]) - set(CELLS):
        raise ValueError(f"ablation.families must be a non-empty subset of {CELLS}")
    a = cfg["ablation"]["stage_a"]
    for key in ("sequence_length", "hidden_size", "num_layers", "dropout"):
        if key not in a:
            raise ValueError(f"ablation.stage_a.{key} is missing")
    if cfg["selection"]["metric"] not in ("rmse", "mae", "nasa_score"):
        raise ValueError("selection.metric must be one of rmse, mae, nasa_score (lower is better)")
    if cfg["selection"]["stage_b_comparison"] != "common_endpoints":
        raise ValueError("selection.stage_b_comparison must be 'common_endpoints'")
    if cfg["reference"]["family"] not in CELLS:
        raise ValueError(f"reference.family must be one of {CELLS}")
    return cfg, text, config_path
