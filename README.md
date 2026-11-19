# C-MAPSS Remaining Useful Life (RUL) Research

Research project on aircraft-engine RUL prediction with the NASA C-MAPSS turbofan dataset.

## Dataset

Raw NASA C-MAPSS files (FD001-FD004) are in `data/raw/`. Each train/test file has 26
whitespace-separated columns: `unit_id`, `cycle`, 3 operational settings and 21 sensors.
FD001 (100 train engines, 100 test engines, one operating condition) is the working subset.
FD003 is present but not used yet.

## Session 1: RUL learning problem

Session 1 turns raw FD001 into a deterministic, leakage-safe supervised dataset:

- canonical schema, parsed with a whitespace regex (26 columns validated)
- training RUL = `max_cycle - cycle`, with a configurable piecewise-linear cap (`rul.cap`)
- train/validation split by `unit_id` only (seeded, `split.val_fraction`, `split.seed`)
- scaler fitted on training engines only, then applied to validation and test
- sequence windows of `windows.sequence_length` consecutive cycles from one engine, stride 1;
  the target is the RUL of the final timestep. Test data gets one window per engine (its last
  cycles), labelled with the official `RUL_FD001.txt` value
- automated sanity checks (the run stops with an error if any invariant fails)
- diagnostics, 5 figures and a Markdown validation report

No models are trained in Session 1.

## Environment setup

Python 3.11 is used.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Session 1

From the project root:

```bash
python -m src.run_session1 --config configs/session1_fd001.yaml
```

All settings (RUL cap, sequence length, split fraction and seed, selected features, scaler,
output paths) live in `configs/session1_fd001.yaml`.

## Outputs

| Path | Content |
|---|---|
| `reports/session1_validation_report.md` | counts, split summary, window shapes, constant sensors, sanity-check results |
| `reports/session1_channel_diagnostics.csv` | per-channel variance, status and correlation with capped RUL |
| `reports/figures/fig1..fig5_*.png` | lifetime distribution, sensor trajectories, channel variance, correlation with capped RUL, RUL cap example |
| `data/processed/fd001/` (git-ignored) | window arrays (`*_windows.npz`), `split.json`, `scaler.joblib`, `resolved_config.yaml`, `manifest.json` |

Re-running with the same config reproduces identical splits, window counts, shapes, figures
and report.

## Session 2: benchmark and experiment engine

Session 2 provides one evaluation and tracking layer that every later model is judged under:
shared metrics, a model-agnostic evaluator, a fixed timing protocol, MLflow tracking and an
append-only results table. It consumes the frozen Session 1 arrays in `data/processed/fd001/`
(checked against the Session 1 manifest) and runs two baselines. Run Session 1 first if that
folder does not exist.

### Run

```bash
python -m src.run_session2 --config configs/session2_benchmark.yaml
python -m pytest            # unit and integration tests
```

Settings (target type, seed, baselines, timing protocol, MLflow experiment, output paths) are in
`configs/session2_benchmark.yaml`. Dependencies, including `mlflow` and `pytest`, are in
`requirements.txt`.

### Baselines

| Name | Definition |
|---|---|
| `mean_rul` | Predicts one constant: the mean of the training targets (validation/test labels are never used). |
| `linear_regression` | Ordinary least squares on the sensor values of the last cycle of each Session 1 window: `[30, 14]` -> `[14]` features. No temporal summary features. |

### Target and metrics

All models are trained and scored on the same target, set by `benchmark.target_type`
(default `capped`: Session 1's `min(RUL, 125)`), on every split. Metrics live in `src/metrics.py`:

- RMSE and MAE (cycles) and R² (`1 - SS_res / SS_tot`; for a constant target it is 1.0 if the predictions are perfect and 0.0 otherwise).
- NASA asymmetric score, the sum over samples of `exp(-d/13) - 1` for `d < 0` and `exp(d/10) - 1` for `d >= 0`, where `d = predicted - true`. `d < 0` is an early (conservative) prediction and `d > 0` a late (optimistic) one; late predictions are penalised more. Lower is better.

The NASA score is a sum, so it grows with the number of samples and should only be compared between
sets of the same size. The validation set has 3272 windows and the test set 100 (one per engine).

### Timing protocol

- Training time: wall-clock seconds of one `fit` call (`time.perf_counter`).
- Inference: one call predicts a batch of `timing.batch_size` windows (the first ones of `timing.timing_set`), including the model's input preparation from the Session 1 window. `timing.warmup_runs` un-timed calls are followed by `timing.timed_runs` timed calls.
- Reported: median latency per batch (ms), the same divided by the batch size (latency per sample), and throughput = batch size / median latency (samples per second). The 95th percentile per batch is logged to MLflow.
- Timing values vary between runs; predictions and metrics do not.
- Model size is the total size in bytes of the saved model files.

### Results and tracking

| Path | Content |
|---|---|
| `results/benchmark_results.csv` / `.json` | one row per run and evaluation set; new runs are appended, never overwritten |
| `results/runs/<run_id>/` | model files, `predictions.npz`, the config used, `environment.json`, `sensors.json` |
| `mlruns/` | local MLflow store (SQLite database plus artifacts, no server needed) |

`run_id` is the MLflow run ID and appears in every results row. The `results/` and `mlruns/`
folders are git-ignored. To browse runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Each run logs its configuration, seed, sensor list and count, metrics, timing, model size, model
artifacts, the Git commit (and whether the working tree had uncommitted changes) and the
hardware and library versions. Later models run through `src/experiment.py::run_experiment`
and `src/evaluation.py::evaluate_model` instead of writing their own metric or logging code.

## Session 3: classical ML and sensor intelligence

Session 3 builds a non-neural benchmark on the Session 1 windows and creates leakage-safe sensor
rankings and frozen sensor subsets for later experiments. Models run through the Session 2
engine (`run_experiment` / `evaluate_model`), so they use the same metrics, timing protocol,
MLflow experiment and results table.

```bash
python -m src.run_session3 --config configs/session3_classical_ml.yaml   # about 2 minutes
```

- **Features** (`src/tabular_features.py`): per sensor `last, mean, std, min, max, slope` over the
  30-cycle window (slope = OLS slope over the cycle index, std = population std), giving
  14 x 6 = 84 columns in sensor-major order with names `<sensor>__<feature>`.
- **Models** (`src/classical_models.py`): random forest and histogram gradient boosting
  (`HistGradientBoostingRegressor`; XGBoost is not installed in this environment).
- **Selection** (`src/tuning.py`): a 16-candidate grid per family, declared in the config. Candidates
  are fitted on training windows and scored on validation windows only; within 1% of the best
  validation RMSE the smallest model is chosen. The test set is used once, for the selected models.
- **Ranking A**: |Pearson r| between each sensor's last-cycle value and capped RUL, training windows only.
- **Ranking B**: validation permutation importance (increase in validation RMSE) of the fitted
  random forest per feature column, summed over each sensor's six features.
- **Comparison and subsets**: Spearman rank correlation with the largest disagreements; top-k
  subsets (k = 14, 10, 7, 4, 2) per ranking; random-k controls with fixed seeds.

| Path | Content |
|---|---|
| `artifacts/session3/` (versioned) | `sensor_ranking_correlation.csv`, `sensor_ranking_permutation.csv`, `sensor_ranking_comparison.json`, `sensor_subsets_topk.json`, `sensor_subsets_randomk.json`, `permutation_feature_importance.json`, `manifest.json` (SHA-256 of each file) |
| `results/session3/` (git-ignored) | search tables for both families, `selection.json`, `model_comparison.csv` |
| `reports/session3_classical_ml_report.md`, `reports/figures/fig6..fig7_*.png` | review report and figures |
| `results/benchmark_results.*` | Session 2 table, extended with optional `n_parameters`, `model_family`, `feature_set`, `selected_params` columns |

Predictions, rankings and subsets are reproducible; timings vary. Permutation importance on
individual features understates sensors whose features are mutually redundant (for example
`last`, `mean`, `min`, `max` of the same sensor).

## Layout

```
configs/   YAML configuration (Session 1 and Session 2)
src/       Session 1 pipeline; Session 2 metrics, timing, evaluation, baselines, tracking, results;
           Session 3 features, classical models, tuning, sensor rankings
artifacts/ frozen Session 3 sensor rankings and subsets (version-controlled)
tests/     pytest suite (Session 2 benchmark layer, Session 3 features, models, rankings)
data/raw/  NASA C-MAPSS files (not modified)
reports/   validation report, diagnostics CSV and figures
results/   benchmark results table and per-run artifacts (git-ignored)
```
