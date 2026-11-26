# Session 8 evaluation protocol (version 1)

Configuration: `configs/session8_evaluation.yaml`. Written on 2026-11-26, after Sessions 1-7 were
complete and published, and **before any FD003 prediction was computed or any FD003 test label was
read by this project**. Nothing here was registered before Session 7.

## Origin of each element

| From the roadmap | Specified here for the first time |
|---|---|
| FD001 to FD003 test with frozen models and subsets | Zero-shot use of the FD001 scaler; the candidate list; FD003 training trajectories as a secondary set |
| Per-engine error by RUL stage (<=50, 51-100, >100) | Stages by uncapped true RUL; per-engine aggregation |
| Largest absolute and late errors, sensor trajectories | Selection rules below |
| Late versus early asymmetry and the NASA score | Sign convention and summaries |
| Sensitivity checks (RUL cap, ranking method) | Caps 100 and 150 on seven configurations; ranking comparison from frozen results |
| Statistical comparison | The three test families below |

## Research questions

R1 Do models and sensor subsets developed on FD001 transfer to FD003 without retraining?
R2 Does sensor reduction change near-failure error differently from early-life error?
R3 Which engines expose the largest errors, and do model families fail on the same ones?
R4 Are models mainly optimistic or conservative, and how does the NASA score reflect it?
R5 Do conclusions depend on the ranking method or the RUL cap?

## Frozen inputs

All 174 Session 7 canonical models (commit `0b7d0b3`), loaded through `canonical_runs`. Same 14
sensors and order, same FD001-fitted scaler (applied, never refitted), window 30, cap 125, same
metric code (`src/metrics.py`). No FD003 data is used for training, scaling, ranking, calibration,
threshold or model choice. Before FD003 is touched, every loaded model must reproduce its stored
FD001 validation predictions exactly.

## Datasets

| Set | Content | Role |
|---|---|---|
| FD003 official test | 100 engines, one window (last 30 cycles), target min(official RUL, 125) | **Primary**; evaluated once under this protocol |
| FD003 training trajectories | 100 run-to-failure engines, all stride-1 windows | Secondary, descriptive; never used for any choice |
| FD001 validation | 20 engines, 3,272 windows | Reference; affected by selection (architectures, early stopping, permutation ranking) |
| FD001 official test | not opened | Earlier recorded rows for the six full-sensor models are quoted as **previously inspected** context only |

FD001 validation windows and FD003 terminal windows are different protocols; their difference is
never called a pure generalisation gap. The trajectory-to-trajectory comparison (FD001 validation
against FD003 training engines) uses the same windowing but still differs in lifetimes and stage mix.

## Primary analyses (FD003 official test)

P1 RMSE, MAE, R2 and NASA score of the candidates, with engine bootstrap intervals.
P2 Family A: each reduced candidate against the full-sensor model of the same architecture
   (for `gb_k10_random`, per-engine mean error of the five draws). 5 tests.
P3 Family B: each other full-sensor model against the Transformer. 5 tests.

## Secondary analyses

S1 Family C: each ranked subset against the mean of the five random draws (48 tests) on FD003 test.
S2 All 174 conditions on both FD003 sets (descriptive).
S3 Stage, worst-engine and asymmetry analysis on all three evaluated sets.
S4 RUL-cap sensitivity (below).
Everything not listed here is exploratory and labelled so.

## Statistics

Unit: engine. Two-sided Wilcoxon signed-rank on paired per-engine absolute errors (zeros dropped,
`zero_method="wilcox"`, average ranks for ties), Holm correction within each family, matched-pairs
rank-biserial effect size, 95% bootstrap interval of the median difference (engines resampled). The
test assumes roughly symmetric paired differences; the effect size and interval are reported with
it. A non-significant result is not equivalence. No test pairs FD001 engines with FD003 engines.

## RUL stages and errors

Error = predicted - true (cycles); positive = optimistic (late), negative = conservative (early).
Predictions are compared with the capped target; stages use the uncapped true RUL. On trajectory
sets, metrics are computed per engine and then averaged over engines; pooled window counts are
reported but windows are never treated as independent.

## Worst-engine rule

For each evaluated set, over the six full-sensor models: engine score = mean across models of
(a) absolute error, (b) positive part of the error, (c) negative part, (d) RMSE over windows with
raw RUL <= 50. The five highest engines per criterion are listed (ties: lower unit ID). The three
highest by (a) are plotted for the trajectory sets. No cycle-wise plot is made from FD003 test.

## Sensitivity

- Ranking method: frozen Session 7 results (FD001) and family C (FD003). The permutation ranking was
  built on the FD001 validation engines.
- RUL cap in training: refit the six full-sensor definitions and `gru_k10_perm` with caps 100 and
  150, otherwise frozen (target scale follows the cap, as at 125), plus a control refit at 125 that
  must reproduce the canonical predictions exactly. 21 fits, under half an hour on CPU. Scored on FD001 validation only, mainly on windows with raw RUL <= 100 whose target is the
  same under every cap. This does not select a cap and does not replace any canonical model.
- Evaluation target: FD003 test also scored against the uncapped official RUL. This is not a cap
  sensitivity of the models.

## Interpretation limits

Simulated data; two datasets; one training seed; FD003 differs from FD001 by a second fault mode
(fan degradation) with the same single sea-level condition, and the files give no per-engine fault
label. Removing a model input does not show that a physical sensor can be removed from an aircraft.
Latency is the coarse Session 7 measurement; nothing is re-timed.

## One-time rule

The first read of the FD003 test labels is recorded in `results/session8/fd003_test_access.json`
with the hash of this document and the config. The code refuses to evaluate those labels under a
different protocol or config. Re-running the identical protocol reproduces the same numbers and is
logged.
