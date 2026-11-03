# C-MAPSS Remaining Useful Life (RUL) Research

Research project on aircraft-engine RUL prediction with the NASA C-MAPSS turbofan dataset.

## Dataset

Raw NASA C-MAPSS files (FD001-FD004) are in `data/raw/`. Each train/test file has 26
whitespace-separated columns: `unit_id`, `cycle`, 3 operational settings and 21 sensors.
FD001 (100 train engines, 100 test engines, one operating condition) is the working subset.
FD003 is present but not used yet.

## Current status: Session 1 (RUL learning problem)

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

## Layout

```
configs/   YAML configuration
src/       Session 1 pipeline (loading, rul, splitting, scaling, windows, checks, diagnostics, report)
data/raw/  NASA C-MAPSS files (not modified)
reports/   validation report, diagnostics CSV and figures
```
