"""Load and validate the Session 6 (Transformer) configuration."""
from pathlib import Path

import yaml

from .recurrent_models import DEVICE_CHOICES
from .session5_config import check_protocol_matches_reference  # same guard: training, seed, device
from .transformer_models import POOLINGS, require_positive_int, validate_architecture

__all__ = ["load_session6_config", "check_protocol_matches_reference"]

_REQUIRED = {
    "session1": ["config", "processed_dir"],
    "session2": ["config"],
    "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
    "input": ["n_sensors", "sequence_length"],
    "training": ["optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
                 "early_stopping_patience", "min_delta", "grad_clip", "scheduler", "target_scale"],
    "protocol_reference": ["config", "require_identical_training"],
    "execution": ["device"],
    "model": ["positional_encoding", "attention_mask", "normalization", "activation", "ff_multiplier", "dropout"],
    "study": ["stage_a", "stage_b", "sensor_aware_variant"],
    "stability_gate": ["samples", "epochs", "learning_rate", "min_loss_drop", "session7_sensor_counts"],
    "selection": ["metric", "tie_tolerance_rel"],
    "outputs": ["frozen_dir", "checkpoint_dir", "working_dir", "report", "figures_dir"],
}
_FIXED = {"positional_encoding": "sinusoidal", "attention_mask": "none", "normalization": "pre_norm"}


def load_session6_config(config_path):
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
    if cfg["selection"]["metric"] not in ("rmse", "mae", "nasa_score"):
        raise ValueError("selection.metric must be one of rmse, mae, nasa_score (lower is better)")
    require_positive_int("input.n_sensors", cfg["input"]["n_sensors"])
    require_positive_int("input.sequence_length", cfg["input"]["sequence_length"])
    model = cfg["model"]
    for key, value in _FIXED.items():
        if model[key] != value:
            raise ValueError(f"model.{key} must be '{value}' (the only implemented policy)")
    a = cfg["study"]["stage_a"]
    for key in ("d_model", "num_heads", "num_layers"):
        if not a.get(key):
            raise ValueError(f"study.stage_a.{key} must be a non-empty list")
    if a.get("pooling") not in POOLINGS:
        raise ValueError(f"unsupported pooling '{a.get('pooling')}', expected one of {POOLINGS}")
    poolings = cfg["study"]["stage_b"].get("pooling") or []
    if len(set(poolings)) < 2 or set(poolings) - set(POOLINGS):
        raise ValueError(f"study.stage_b.pooling must list at least two of {POOLINGS}")
    if a["pooling"] not in poolings:
        raise ValueError("study.stage_b.pooling must include the Stage A pooling")
    for d_model in a["d_model"]:
        for heads in a["num_heads"]:
            for layers in a["num_layers"]:
                validate_architecture(d_model, heads, layers, model["ff_multiplier"], model["dropout"],
                                      a["pooling"], model["activation"])
    for count in cfg["stability_gate"]["session7_sensor_counts"]:
        require_positive_int("stability_gate.session7_sensor_counts", count)
    if cfg["study"]["sensor_aware_variant"]:
        raise ValueError("study.sensor_aware_variant is not implemented; set it to false")
    return cfg, text, config_path
