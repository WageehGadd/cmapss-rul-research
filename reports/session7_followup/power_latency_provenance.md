# Power and latency provenance of the canonical Session 7 run

Applies to the 174 runs of commit `0b7d0b3` published in `21ec708`. The published report states
that power source and load "were not recorded". That is true of the run metadata: no record or
manifest holds them. The observations below come from outside the pipeline.

## Sources

- The macOS power log (`pmset -g log`), read after the run. An excerpt is kept in the canonical
  evidence backup as `provenance/power_source_log_20261126.txt`
  (SHA-256 `b474a18e8b31972743cca71aece27a79cd3c17ac21d3e1403d75d65fd96c7602`).
- Two manual snapshots, at the start and the end of the run
  (`results/session7_canonical_observations.txt`, git-ignored). They show mains power at both
  moments and must not be read as evidence of continuous mains power.
- Start and end timestamps in the 174 condition records.

## Observed power source

Date 2026-11-26, local time UTC+2. The records span 15:20:35 to 17:15:53. Times are those of the
first log summary line showing the new source, so each boundary is approximate.

| Battery interval | Duration | Charge when first seen on battery, then on mains again |
|---|---|---|
| 15:50:49 to 15:51:34 | 45 s | 24%, 24% |
| 16:08:55 to 16:09:16 | 21 s | 34%, 34% |
| 16:10:38 to 16:10:40 | 2 s | 35%, 35% |
| 16:26:20 to 16:41:47 | 15 min 27 s | 43%, 36% |

Otherwise the log shows mains power. The charge was 7-8% when the run started.

## Conditions whose execution overlapped a battery interval

Ten conditions: 8 TCN and 2 Transformer.

| Condition | Ran | Overlaps interval starting | Latency (ms per batch) |
|---|---|---|---|
| `s7_transformer_k07_permutation_1c7c895ecb` | 15:50:41 to 15:51:25 | 15:50:49 | 6.44 |
| `s7_transformer_k04_correlation_078416578f` | 15:51:25 to 15:51:57 | 15:50:49 | 9.21 |
| `s7_tcn_k10_random101_7d1cbd139a` | 16:08:03 to 16:10:43 | 16:08:55 and 16:10:38 | 48.08 |
| `s7_tcn_k07_random103_f12ba82edd` | 16:26:16 to 16:28:51 | 16:26:20 | 50.12 |
| `s7_tcn_k07_random104_529d6d9131` | 16:28:51 to 16:30:53 | 16:26:20 | 47.63 |
| `s7_tcn_k07_random105_926b274e40` | 16:30:53 to 16:32:54 | 16:26:20 | 50.54 |
| `s7_tcn_k04_random101_132fa94704` | 16:32:54 to 16:35:33 | 16:26:20 | 47.56 |
| `s7_tcn_k04_random102_7d064ee82e` | 16:35:33 to 16:38:10 | 16:26:20 | 48.10 |
| `s7_tcn_k04_random103_2b8f96226a` | 16:38:10 to 16:41:14 | 16:26:20 | 47.50 |
| `s7_tcn_k04_random104_21839c4929` | 16:41:14 to 16:43:00 | 16:26:20 | 46.92 |

Overlap is by whole condition (training, evaluation and timing). The records do not timestamp the
inference timing block, so for conditions at the edge of an interval it is unknown whether the
latency itself was measured on battery. None of the ten is on either published Pareto front.

## Which measurements may be affected

- **Possibly affected:** training time, inference latency (per batch and per sample), throughput and
  wall time. Model size and parameter counts are not timing measurements.
- **Not affected:** predictions, RMSE, MAE, R2 and NASA score. Training and inference ran on CPU
  with fixed seeds; power source changes speed, not the computation. This is observed, not only
  argued: all 174 prediction hashes of the canonical run equal those of the verification run, which
  ran under different power and load conditions and from earlier, uncommitted code.

## Why latency supports only coarse comparisons

- Each condition was timed once, in one block of repeated calls, on one laptop.
- Power source, thermal state and concurrent load were not controlled.
- Within one model, latency drifts by more than any plausible sensor effect. Gradient boosting has a
  median of 6.05 ms over its random subsets and 3.39 ms over its correlation and permutation subsets.
  Those groups ran at different times (stages 3 and 2), so the gap cannot be attributed to sensors.
- Differences between families of an order of magnitude (TCN near 48 ms against a few ms for the
  others) are the kind of comparison these measurements can support, on this machine only.

## Why sensor-level Pareto membership is unstable

The published (RMSE, latency) front has 8 conditions, 7 of them GRU within 0.07 ms of each other.
The verification run produced 12 for the same predictions, with 5 in common. With sensor count as a
third objective the figures are 15 and 19, with 12 in common. Since accuracy is identical between
the two runs, the changes in membership come from timing.

- **Numerical membership** is what the algorithm returns on one set of measurements.
- **Practical reading** (made after seeing the results): only the full-sensor GRU and Transformer
  are clearly distinct. No smaller or random subset is shown to be faster.
- **Deployment:** these measurements do not justify choosing a sensor subset for speed, or any
  latency budget on other hardware.

## Not measured or recorded

Per-run power source, battery charge, power mode, CPU frequency, thermal state or throttling,
load during each condition (only the two snapshots: load averages 2.15/2.81/3.39 at the start and
3.09/3.27/3.26 at the end), other applications, and repeated timing of any canonical condition.
Latency was not re-benchmarked for this note.
