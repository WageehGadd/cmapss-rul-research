# Session 1 Validation Report: FD001

## Raw data
- Raw train rows: 20631
- Raw test rows: 13096
- Total training engines: 100
- Test engines: 100

## Engine-level split
- Split seed: 42; validation fraction: 0.2
- Training engines after split: 80
- Validation engines: 20
- Engines present in both train and validation: **0** (zero overlap confirmed)
- Validation engine IDs: [3, 5, 19, 22, 25, 26, 28, 34, 43, 51, 52, 53, 57, 60, 62, 72, 82, 93, 98, 100]

## Row counts after loading and splitting
| Set | Engines | Rows |
|---|---|---|
| train | 80 | 16779 |
| validation | 20 | 3852 |
| test | 100 | 13096 |

## Sequence windows
- Sequence length: 30 cycles (stride 1)
- Selected feature count: 14
- Window tensor shape: [n_windows, 30, 14]
- Target: RUL of the final timestep of each window

| Set | Windows | X shape | y shape |
|---|---|---|---|
| train | 14459 | 14459x30x14 | 14459 |
| validation | 3272 | 3272x30x14 | 3272 |
| test | 100 | 100x30x14 | 100 |

Test windows: one per engine (its last cycles), labelled with the official RUL file.

## Target and preprocessing settings
- RUL cap: 125
- Scaler type: standard (fitted on training engines only)
- Selected features: sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21

## Constant and near-constant channels (training engines, raw units)
- constant: operational_setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
- near-constant: operational_setting_1, operational_setting_2, sensor_6
- Thresholds: constant std <= 1e-08; near-constant std < 0.01 or <= 2 distinct values
- Correlation with capped RUL is reported as `undefined` for constant channels.

| Channel | Variance | Distinct values | Status | Corr. with capped RUL |
|---|---|---|---|---|
| operational_setting_1 | 4.77854e-06 | 154 | near-constant | -0.003 |
| operational_setting_2 | 8.55169e-08 | 13 | near-constant | -0.007 |
| operational_setting_3 | 0 | 1 | constant | undefined |
| sensor_1 | 0 | 1 | constant | undefined |
| sensor_2 | 0.250081 | 304 | ok | -0.680 |
| sensor_3 | 37.4678 | 2922 | ok | -0.655 |
| sensor_4 | 80.7914 | 3894 | ok | -0.761 |
| sensor_5 | 0 | 1 | constant | undefined |
| sensor_6 | 1.93379e-06 | 2 | near-constant | -0.111 |
| sensor_7 | 0.788536 | 509 | ok | 0.737 |
| sensor_8 | 0.00497691 | 50 | ok | -0.638 |
| sensor_9 | 442.427 | 5731 | ok | -0.457 |
| sensor_10 | 0 | 1 | constant | undefined |
| sensor_11 | 0.071415 | 158 | ok | -0.779 |
| sensor_12 | 0.545868 | 423 | ok | 0.754 |
| sensor_13 | 0.00511237 | 54 | ok | -0.640 |
| sensor_14 | 328.331 | 5447 | ok | -0.358 |
| sensor_15 | 0.00140563 | 1878 | ok | -0.722 |
| sensor_16 | 0 | 1 | constant | undefined |
| sensor_17 | 2.3981 | 13 | ok | -0.682 |
| sensor_18 | 0 | 1 | constant | undefined |
| sensor_19 | 0 | 1 | constant | undefined |
| sensor_20 | 0.0327053 | 117 | ok | 0.706 |
| sensor_21 | 0.0117519 | 4539 | ok | 0.712 |

## Figures
- `reports/figures/fig1_engine_lifetime_distribution.png`
- `reports/figures/fig2_sensor_trajectories.png`
- `reports/figures/fig3_channel_variance.png`
- `reports/figures/fig4_correlation_with_capped_rul.png`
- `reports/figures/fig5_rul_cap_example.png`

## Sanity checks
| Check | Result | Detail |
|---|---|---|
| B. train_FD001.txt has exactly 26 columns | PASS | train_FD001.txt: 26 columns |
| A. train_FD001.txt canonical columns | PASS | train_FD001.txt: all 26 canonical columns present, in order |
| C. train_FD001.txt no missing values | PASS | train_FD001.txt: no missing values |
| E/F. train_FD001.txt cycles | PASS | train_FD001.txt: every engine starts at cycle 1 and cycles increase by 1 |
| H. train_FD001.txt expected engine/row counts | PASS | train_FD001.txt: 100 engines, 20631 rows |
| B. test_FD001.txt has exactly 26 columns | PASS | test_FD001.txt: 26 columns |
| A. test_FD001.txt canonical columns | PASS | test_FD001.txt: all 26 canonical columns present, in order |
| C. test_FD001.txt no missing values | PASS | test_FD001.txt: no missing values |
| E/F. test_FD001.txt cycles | PASS | test_FD001.txt: every engine starts at cycle 1 and cycles increase by 1 |
| H. test_FD001.txt expected engine/row counts | PASS | test_FD001.txt: 100 engines, 13096 rows |
| H. official RUL file matches test engines | PASS | RUL file: 100 values, one per test engine |
| RUL derivation and cap | PASS | raw RUL = max_cycle - cycle (0 at failure); capped RUL = min(raw, 125) |
| G. no train/validation unit_id overlap | PASS | 80 train / 20 validation engines, 0 engines in both, together they cover all 100 |
| Selected features vary on training engines | PASS | all 14 selected features vary on the training engines |
| L. scaler fitted on training engines only | PASS | scaler fitted on 16779 rows of the 80 training engines; statistics match the training rows; no validation/test engine used |
| D. no NaN/inf after scaling (train) | PASS | scaled train features: all values finite |
| D. no NaN/inf after scaling (validation) | PASS | scaled validation features: all values finite |
| D. no NaN/inf after scaling (test) | PASS | scaled test features: all values finite |
| I. window shapes (train) | PASS | train: X (14459, 30, 14), y (14459,) |
| D. no NaN/inf in windows (train) | PASS | train X: all values finite |
| J/K. window boundaries and target alignment (train) | PASS | train: all 14459 windows are 30 consecutive cycles of one engine and their targets equal the RUL of the final timestep |
| I. window shapes (validation) | PASS | validation: X (3272, 30, 14), y (3272,) |
| D. no NaN/inf in windows (validation) | PASS | validation X: all values finite |
| J/K. window boundaries and target alignment (validation) | PASS | validation: all 3272 windows are 30 consecutive cycles of one engine and their targets equal the RUL of the final timestep |
| I. window shapes (test) | PASS | test: X (100, 30, 14), y (100,) |
| D. no NaN/inf in windows (test) | PASS | test X: all values finite |
| J/K. window boundaries and target alignment (test) | PASS | test: all 100 windows are 30 consecutive cycles of one engine and their targets equal the RUL of the final timestep |
| G. no engine contributes windows to both train and validation | PASS | train and validation windows come from disjoint engines |

**All 28 sanity checks passed.**

## Reproducibility fingerprints (SHA-256 of array contents)
- `train_X`: `34917872c55cd9baef24bf19e2e40debe97d85e75f4a761d0a3ddcaa59d46189`
- `train_y`: `2a04b8e7b65e2bc500b94ff66e27db9e6b1ac357f1266c21f6a9ed752be4faf4`
- `train_y_raw`: `19f0da094957fb6bb4d0b04549cda6df2ee1513a9962e10da9e0d00121453297`
- `train_unit_ids`: `1fc0884488e92d2edb77a891236fe6dd89f3b71d5744ed56949f02c49762042d`
- `train_end_cycles`: `aedeb69164eeecd97e7382aa31729c9d0bb8728ff033896f7ba579d6cf7d6d0a`
- `val_X`: `0918a4a77b90546328bd4dd073248dac49ed273c487ee03925c8eece812ecef3`
- `val_y`: `a2c142968f1b1ee70f1e1bc034759a49880842e32fddc51ba937e1abd47e3bb7`
- `val_y_raw`: `d63c726cfa880fa24fba8c86ed11d4ed018a2d0d66d0f8c87807d3460bb5a079`
- `val_unit_ids`: `0cf458a45b83f6cbb78fe371e5e3a2eaadf889e393390bd14d0345711a1fc554`
- `val_end_cycles`: `ec3c8ef767d20c2581fb0691ee24371fc45d98e7c432b8b6b0d346468b93f72c`
- `test_X`: `d5a77270663bf1c023ec74bcd0dfc8159bad6467e60c2c1889e573a32e8ee873`
- `test_y`: `b86e07b9cfde953087c318120122e422f5c51645aa3f9353885348f11e239f0a`
- `test_y_raw`: `2039f434cb333405e8c65058d9679687afa56a65f72c1971283d5b5c3af8aeee`
- `test_unit_ids`: `d6a25caee0839129c601a96b6f4af48f38808a971371c55e3174f2584110b597`
- `test_end_cycles`: `f71ab9252f9b1ba2ad5e817663bb1fdf87cb376b9d949b7b6298a7fc37cb4670`
