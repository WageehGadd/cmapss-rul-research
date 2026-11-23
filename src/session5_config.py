"""Load and validate the Session 5 (convolutional architectures) configuration."""
from pathlib import Path

import yaml

from .benchmark_config import resolve_path
from .convolutional_models import KINDS, cnn_output_length
from .recurrent_models import DEVICE_CHOICES

_REQUIRED = {
    "session1": ["config", "processed_dir"],
    "session2": ["config"],
    "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
    "input": ["n_sensors", "sequence_length"],
    "training": ["optimizer", "learning_rate", "batch_size", "max_epochs", "early_stopping_patience",
                 "min_delta", "grad_clip", "scheduler", "target_scale"],
    "protocol_reference": ["config", "require_identical_training"],
    "execution": ["device"],
    "architecture": ["dropout", "cnn", "tcn"],
    "selection": ["metric", "tie_tolerance_rel"],
    "outputs": ["frozen_dir", "checkpoint_dir", "working_dir", "report", "figures_dir"],
}


def load_session5_config(config_path):
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
    if not 0.0 <= float(cfg["architecture"]["dropout"]) < 1.0:
        raise ValueError("architecture.dropout must be in [0, 1)")
    cnn, tcn = cfg["architecture"]["cnn"], cfg["architecture"]["tcn"]
    for key in ("channels", "kernel_size", "depth"):
        if not cnn.get(key):
            raise ValueError(f"architecture.cnn.{key} must be a non-empty list")
    for key in ("channels", "kernel_size", "dilations"):
        if not tcn.get(key):
            raise ValueError(f"architecture.tcn.{key} must be a non-empty list")
    if any(not d or any(int(v) < 1 for v in d) for d in tcn["dilations"]):
        raise ValueError("architecture.tcn.dilations must be lists of positive integers")
    L = int(cfg["input"]["sequence_length"])
    for k in cnn["kernel_size"]:
        for d in cnn["depth"]:
            if cnn_output_length(L, k, d) < 1:
                raise ValueError(f"cnn kernel {k} x depth {d} does not fit sequence_length={L}")
    if set(KINDS) - set(cfg["architecture"]):
        raise ValueError(f"architecture must define {KINDS}")
    return cfg, text, config_path


def check_protocol_matches_reference(cfg):
    """The training protocol must equal the frozen recurrent protocol (Session 4) when required."""
    ref = cfg["protocol_reference"]
    if not ref["require_identical_training"]:
        return
    other = yaml.safe_load(resolve_path(ref["config"]).read_text())
    if other["training"] != cfg["training"]:
        diff = sorted(k for k in set(other["training"]) | set(cfg["training"])
                      if other["training"].get(k) != cfg["training"].get(k))
        raise ValueError(f"training protocol differs from {ref['config']} in {diff}")
    if other["benchmark"]["seed"] != cfg["benchmark"]["seed"]:
        raise ValueError(f"seed differs from {ref['config']}")
    if other["execution"]["device"] != cfg["execution"]["device"]:
        raise ValueError(f"execution.device differs from {ref['config']}")
