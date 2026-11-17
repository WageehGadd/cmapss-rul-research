"""Load and validate the Session 2 benchmark configuration."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_BASELINES = ("mean_rul", "linear_regression")
TARGET_TYPES = ("capped", "raw")
EVAL_SETS = ("validation", "test")

_REQUIRED = {
    "session1": ["config", "processed_dir"],
    "benchmark": ["target_type", "seed", "evaluation_sets", "sensor_set_name"],
    "timing": ["warmup_runs", "timed_runs", "batch_size", "latency_unit", "timing_set",
               "hardware_description"],
    "tracking": ["tracking_dir", "experiment_name"],
    "outputs": ["results_csv", "results_json", "runs_dir"],
    "environment": ["library_names"],
}


@dataclass
class BenchmarkConfig:
    values: dict
    path: Path
    text: str

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()

    @property
    def ref(self):
        """Config path (relative to the project root when inside it)."""
        try:
            return str(self.path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(self.path)

    def __getitem__(self, key):
        return self.values[key]


def resolve_path(path):
    """Paths in configs are relative to the project root unless absolute."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_benchmark_config(config_path):
    config_path = Path(config_path).resolve()
    text = config_path.read_text()
    cfg = yaml.safe_load(text)
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise ValueError(f"Config {config_path} is missing section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise ValueError(f"Config {config_path} is missing '{section}.{key}'")

    bench, timing = cfg["benchmark"], cfg["timing"]
    if bench["target_type"] not in TARGET_TYPES:
        raise ValueError(f"benchmark.target_type must be one of {TARGET_TYPES}")
    if not bench["evaluation_sets"] or any(s not in EVAL_SETS for s in bench["evaluation_sets"]):
        raise ValueError(f"benchmark.evaluation_sets must be a non-empty subset of {EVAL_SETS}")
    if timing["timing_set"] not in bench["evaluation_sets"]:
        raise ValueError("timing.timing_set must be one of benchmark.evaluation_sets")
    if timing["latency_unit"] != "ms":
        raise ValueError("timing.latency_unit is fixed to 'ms' by the benchmark protocol")
    if int(timing["warmup_runs"]) < 0 or int(timing["timed_runs"]) < 1 or int(timing["batch_size"]) < 1:
        raise ValueError("timing needs warmup_runs >= 0, timed_runs >= 1 and batch_size >= 1")

    baselines = cfg.get("baselines") or []
    if not baselines:
        raise ValueError("Config defines no baselines")
    names = [b.get("name") for b in baselines]
    if len(set(names)) != len(names):
        raise ValueError(f"Baseline names must be unique, got {names}")
    for b in baselines:
        if b.get("type") not in SUPPORTED_BASELINES:
            raise ValueError(f"Baseline {b.get('name')}: type must be one of {SUPPORTED_BASELINES}")
    return BenchmarkConfig(values=cfg, path=config_path, text=text)
