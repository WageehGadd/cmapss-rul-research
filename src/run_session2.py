"""Session 2 entry point: run the baselines through the shared benchmark engine.

Usage (from the project root):
    python -m src.run_session2 --config configs/session2_benchmark.yaml
"""
import argparse
import sys

from .baselines import build_baseline
from .benchmark_config import PROJECT_ROOT, load_benchmark_config
from .benchmark_data import load_benchmark_data
from .environment import capture_environment
from .experiment import run_experiment
from .tracking import MlflowTracker


def run(config_path):
    cfg = load_benchmark_config(config_path)
    data = load_benchmark_data(cfg["session1"]["processed_dir"], cfg["session1"]["config"],
                               cfg["benchmark"]["target_type"])
    environment = capture_environment(
        PROJECT_ROOT, cfg["environment"]["library_names"],
        {"session1_config": cfg["session1"]["config"],
         "session2_config": f"{cfg.ref}@{cfg.sha256}"},
        hardware_note=cfg["timing"]["hardware_description"])
    tracker = MlflowTracker(cfg["tracking"]["tracking_dir"], cfg["tracking"]["experiment_name"])

    results = []
    for spec in cfg["baselines"]:
        results.append(run_experiment(build_baseline(spec), data, cfg, tracker, environment))
    return cfg, data, tracker, results


def main():
    parser = argparse.ArgumentParser(description="Run the Session 2 baseline benchmark.")
    parser.add_argument("--config", default="configs/session2_benchmark.yaml")
    args = parser.parse_args()
    try:
        cfg, data, tracker, results = run(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"SESSION 2 FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Session 2 complete: {len(results)} runs, target={data.target_type}, "
          f"{len(data.sensors)} sensors")
    print(f"{'model':<20}{'set':<12}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'NASA':>14}")
    for r in results:
        for eval_set, m in r.metrics.items():
            print(f"{r.model_name:<20}{eval_set:<12}{m['rmse']:>9.3f}{m['mae']:>9.3f}"
                  f"{m['r2']:>9.3f}{m['nasa_score']:>14.1f}")
    print(f"  run IDs: {', '.join(r.run_id for r in results)}")
    print(f"  results: {cfg['outputs']['results_csv']}, {cfg['outputs']['results_json']}")
    print(f"  MLflow: {tracker.tracking_uri}")


if __name__ == "__main__":
    main()
