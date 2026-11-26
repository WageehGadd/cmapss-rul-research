# Session 8: remediation of the independent audit

A second re-review found that the evidence seal, the manifest check and the authorization guard were
weaker than described below. Those fixes are recorded in
[session8_integrity_review_note.md](session8_integrity_review_note.md).

Audit: `session8-final-audit.md` (2026-11-26, verdict CONDITIONAL PASS), SHA-256
`758f15728f96cc6afd226c41cfbbb02339c216242ffc5c4dbabfacb64642756d`. It is kept outside the repository.

No model was trained or run, no cap fit or Session 7 experiment was repeated, no MLflow run was
created, and the config and protocol are unchanged. Tables were rebuilt from saved predictions and
the saved test records. `RUL_FD003.txt` was not parsed and the label-access ledger is unchanged;
`verify` hashes that file's bytes as part of the frozen-file snapshot. The previous versions of all
42 Session 8 files are kept in `results/session8/remediation/before/` (local, Git-ignored).

## Findings

| Finding (severity) | Status | What changed |
|---|---|---|
| FD003-test `rmse` and `median_abs` were MAE (high) | FIXED AND VERIFIED | Plain names are pooled; per-engine averages are `engine_mean_*` and empty where undefined; report tables split |
| Order-change claim confounded by protocol (high) | FIXED AND VERIFIED | Three-protocol table and Spearman 0.83 / 0.03 / 0.31; question 1, conclusion 1 and section 5 rewritten |
| FD003-test worst list driven by the TCN (high) | FIXED AND VERIFIED | Predeclared six-model view and labelled post hoc view without TCN, both shown |
| `gb_k10_random` interval estimated another quantity (medium) | FIXED AND VERIFIED | Interval of the mean of the five RMSEs: 17.97-23.32 became 18.68-24.03; other intervals unchanged |
| Mixed asymmetry aggregation (medium) | FIXED AND VERIFIED | Both aggregations tabulated and named; figure 28 is pooled throughout |
| Overstated conclusion 3, question 9, "least safe" (medium) | FIXED AND VERIFIED | Rewritten; non-significant comparisons listed |
| Clean-commit status ignored prediction provenance (medium) | FIXED AND VERIFIED | Predictions record their producer; status needs recorded clean, matching execution. Existing evidence stays "not canonical" |
| Manifest omitted local evidence (medium) | MITIGATED WITH DISCLOSURE | `evidence_manifest.json` hashes it and is enforced, but was sealed after execution |
| Control flag hard-coded; reuse unchecked (low) | FIXED AND VERIFIED | Compared with the canonical digest, also for reused records |
| `verify` skipped silently without snapshot, ignored added files (low) | MITIGATED WITH DISCLOSURE | Missing snapshot and added artifact or evidence files are problems; other new files in the tree are not detected |
| Source that ran the evaluation not on disk (low) | MITIGATED WITH DISCLOSURE | Cannot be recovered; recorded as not captured |
| Self-attested chronology; rerun did not re-infer; protocol says "once" | MITIGATED WITH DISCLOSURE | Report limitations 10 and 11 |
| Post-result analyses unlabelled | FIXED AND VERIFIED | Labelled post hoc; evidence-class table |
| Per-family correction, skewed differences, NASA concentration | MITIGATED WITH DISCLOSURE | Post hoc joint Holm, sign tests and top-5 shares reported |
| Cap order depends on evaluation subset; mechanical cap effect | MITIGATED WITH DISCLOSURE | Report section 9 and figure 31 |
| Shift shown for training file only; permutation top-10 contains s12 | FIXED AND VERIFIED | Test-window inputs added to figure 32; wording corrected |
| Causal reading of sensors 7 and 12 | FIXED AND VERIFIED | Stated as an association; the audit's own intervention diagnostic is not adopted as a project result |
| Notebook not run in a kernel | FIXED AND VERIFIED | Run with `nbclient` and `ipykernel` in a temporary environment; outputs saved |
| Reproduction guide recommended `--stage all` | FIXED AND VERIFIED | Safe stages separated; execution stages need `--authorize-execution` |
| Test coverage gaps | MITIGATED WITH DISCLOSURE | Aggregation, interval, worst views, evidence and provenance tests added. Still untested: a real-checkpoint load and end-to-end family mapping (`verify` re-derives all tables) |
| A clone cannot reproduce; 4 Session 3 tests fail and the loader message misleads without local evidence | UNRESOLVED | Documented; the code involved belongs to frozen sessions |
| No Session 8 backup; Session 7 backups on the same disk | UNRESOLVED | Owner action |
| NASA data redistribution terms unchecked | UNRESOLVED | Owner action |
| Frozen-file snapshot script is not in the repository | UNRESOLVED | The snapshot file is checked by `verify` |
| Cosmetic: raw keys as labels, no uncertainty in figure 23, small labels in figure 29, no sensor 7 panel in figure 27 | UNRESOLVED | Figure 27's sensor list is part of the locked config |

## Statistical recomputation

All 58 declared tests, the generalisation gaps, the ranking comparison and the predeclared worst
lists are numerically unchanged. Changed values are listed above and below.

## Changed files

<!-- hashes -->

| File | Old SHA-256 | New SHA-256 | Reason |
|---|---|---|---|
| `artifacts/session8/architecture_order.csv` | - | 0b9ec55a0bc3 | new file |
| `artifacts/session8/asymmetry.csv` | fecb544974b6 | c337e425125c | pooled and per-engine columns named separately |
| `artifacts/session8/candidate_summary.csv` | 15592ca66327 | 10daf9d8b32c | test engine_mean_rmse emptied; gb_k10_random interval recomputed |
| `artifacts/session8/evidence_manifest.json` | - | 6ee22d851e26 | new file |
| `artifacts/session8/input_shift.csv` | b29a1905a2c1 | 4f5306867dcd | added FD003 test-window input shift |
| `artifacts/session8/manifest.json` | e71d2dd86f11 | 2821920e552b | execution and derivation separated; points to the sealed evidence |
| `artifacts/session8/master_results.csv` | 75056f1b8d89 | d35bd0b16a14 | fd003_test_engine_mean_rmse emptied (it was the MAE) |
| `artifacts/session8/per_engine_candidates.csv` | febbb7464469 | afe4b4986ed1 | test rows: rmse, median and near_rmse emptied |
| `artifacts/session8/posthoc_checks.csv` | - | fa3197e8945d | new file |
| `artifacts/session8/sensitivity_cap.csv` | 7ddcc0d05b15 | b6bec49271c9 | control flag now compared; empty for non-control caps |
| `artifacts/session8/stage_metrics.csv` | 3ec6ccb08e87 | 600f7960a3e8 | plain names are now pooled; engine_mean_* named; undefined test columns empty |
| `artifacts/session8/worst_engines.csv` | e6121278580b | 97cf66e63274 | added the post hoc view without the TCN |
| `notebooks/session8_error_analysis.ipynb` | 46a0cde5d435 | 6272a93b3dd1 | audit fix |
| `reports/figures/fig22_session8_fd001_vs_fd003.png` | 5f1c6a153236 | 1d0e705928c2 | regenerated from corrected code |
| `reports/figures/fig28_session8_error_asymmetry.png` | 6b049d6d7e3b | c7cd324121c1 | regenerated from corrected code |
| `reports/figures/fig29_session8_cross_model_engine_errors.png` | 335831b4ab93 | ce70a985b998 | regenerated from corrected code |
| `reports/figures/fig30_session8_tradeoffs.png` | d6d1760362f1 | 0cbe475fa3f0 | regenerated from corrected code |
| `reports/figures/fig31_session8_rul_cap_sensitivity.png` | c56deb8d1e26 | 119d5d2b9098 | regenerated from corrected code |
| `reports/figures/fig32_session8_input_shift.png` | f61cf3eb99a5 | 50356e5b3706 | regenerated from corrected code |
| `reports/session8_reproducibility.md` | 516660f5569f | 621d2ae214dd | audit fix |
| `reports/session8_research_report.md` | 6eb28b63a843 | bdcf8b103661 | audit fix |
| `src/run_session8.py` | 45f00fbfffe2 | 93ae24597128 | audit fix |
| `src/session8_analysis.py` | 85bd8d14e3a3 | d71523916776 | audit fix |
| `src/session8_data.py` | e38080358598 | 54f23b89b741 | audit fix |
| `src/session8_evaluate.py` | 37f80e234e9d | dd9c7b0cd712 | audit fix |
| `src/session8_evidence.py` | - | cd7b468736c4 | new file |
| `src/session8_inference.py` | 6baecf7cf702 | 59b7536f2725 | audit fix |
| `src/session8_report_text.py` | 24740f179e85 | 9f43e30aec45 | audit fix |
| `src/session8_reporting.py` | 3ffd0fcae416 | e5c10bea1520 | audit fix |
| `src/session8_sensitivity.py` | b0ae0987576b | 2e785f6ca026 | audit fix |
| `tests/test_session8.py` | c1816472dd64 | 189d5007fc20 | audit fix |
