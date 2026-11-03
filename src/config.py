"""Load and validate the Session 1 YAML configuration."""
from pathlib import Path

import yaml

SUPPORTED_SCALERS = ("standard", "minmax")

_REQUIRED = {
    "dataset": ["subset", "raw_dir", "expected_train_engines", "expected_test_engines",
                "expected_train_rows", "expected_test_rows"],
    "rul": ["cap"],
    "split": ["val_fraction", "seed"],
    "features": ["selected"],
    "scaler": ["type"],
    "windows": ["sequence_length"],
    "diagnostics": ["constant_std_threshold", "near_constant_std_threshold",
                    "near_constant_max_unique", "example_units", "example_sensors",
                    "figure_dpi"],
    "outputs": ["processed_dir", "figures_dir", "report_path", "diagnostics_csv"],
}


def load_config(config_path):
    """Return (config dict, project root). The root is the parent of the configs/ folder."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Config {config_path} is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config {config_path} is missing '{section}.{key}'")

    if int(cfg["rul"]["cap"]) < 1:
        raise ValueError("rul.cap must be a positive integer")
    if int(cfg["windows"]["sequence_length"]) < 1:
        raise ValueError("windows.sequence_length must be a positive integer")
    if not 0.0 < float(cfg["split"]["val_fraction"]) < 1.0:
        raise ValueError("split.val_fraction must be strictly between 0 and 1")
    if cfg["scaler"]["type"] not in SUPPORTED_SCALERS:
        raise ValueError(f"scaler.type must be one of {SUPPORTED_SCALERS}")
    if not cfg["features"]["selected"]:
        raise ValueError("features.selected must not be empty")
    return cfg, config_path.parent.parent
