# Session 8 reproducibility

## Environment

Python 3.11 (3.11.15 used). From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used (NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1, PyTorch
2.15.0, MLflow 3.18.0). Runs used CPU on macOS, arm64. The notebook additionally needs `nbclient`,
`nbformat` and `ipykernel`, which are not project dependencies: install them in a separate
environment. It reads only `artifacts/session8/` and `configs/`, so it runs from a clone.

## Data

NASA C-MAPSS turbofan simulation data: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
and the NASA Prognostics Data Repository
<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>.
Expected in `data/raw/`: `train_`, `test_` and `RUL_` files for FD001 and FD003, plus `readme.txt`.
SHA-256 of the FD003 files is in `configs/session8_evaluation.yaml`. Redistribution terms of the
NASA files were not checked here; confirm them before publishing the raw files elsewhere.

## Local evidence that is not in Git

| Path | Content | How to obtain |
|---|---|---|
| `data/processed/fd001/` | Frozen FD001 arrays and scaler | Session 1 reruns it deterministically, or restore the backup |
| `results/session7/` | 174 condition records, model files, predictions | **Restore from backup.** Retraining gives new run IDs and fails the canonical check |
| `mlruns/` | MLflow store with the canonical runs | Restore from backup |
| `results/benchmark_results.json` | Earlier FD001 rows (read only) | Restore from backup |
| `results/session8/` | Predictions, per-engine tables, saved test records, refits, label-access ledger | **Restore from backup.** Primary evidence of this study |

A clone alone cannot reproduce Session 8: everything above is outside Git, and without `mlruns/` the
canonical check fails with a message that a run is not canonical. Back up `results/session8/`,
`mlruns/`, `results/session7/`, `data/processed/`, `checkpoints/` and `results/benchmark_results.json`
before any cleanup.

Model checkpoints are the `model/` folders under `results/session7/runs/<run_id>/`;
`artifacts/session8/manifest.json` lists each condition's run ID and file hashes. The evidence backup
made after Session 7 is described in `session7_followup/session8_handoff.md`. It was kept on the same
disk as the project, so copy it elsewhere before relying on it.

## Four different things

| Thing | Content | How it may change |
|---|---|---|
| Frozen execution evidence | Predictions, refits, saved test records, per-engine tables, access ledger, Session 8 MLflow runs; listed in `artifacts/session8/evidence_manifest.json` | Never |
| Analysis derived from saved evidence | Tables, figures, report, `artifacts/session8/manifest.json` | `derive` and `report`; a different result is rejected until reviewed |
| A new execution | Inference, label reads or training | Only with authorization, in its own locations; it is a different study |
| Reviewed publication metadata | `configs/session8_trusted_reference.json`, [the review note](session8_integrity_review_note.md), later the reviewed commit | By a person, after review; no code writes it |

## Commands

All commands run from the project root with `PYTHONPATH` unset, using
`python -m src.run_session8 --config configs/session8_evaluation.yaml --stage <stage>`.
None of the first three groups runs a model on data, trains, writes MLflow runs, opens
`RUL_FD003.txt` or creates an evidence manifest (`verify` hashes that file's bytes as part of the
frozen-file snapshot). They need the Git-ignored evidence listed above, restored unchanged.

**1. Read-only verification** (writes nothing)

```bash
python -m src.run_session8 --config configs/session8_evaluation.yaml --stage verify
```

Checks the 174 canonical Session 7 conditions; the evidence manifest against the trusted reference;
every sealed file; the three files recorded after sealing; unexpected files; the fields and hash of the
artifacts manifest; figures and report; the sources; that all tables re-derive byte for byte; and the
pre-session snapshot. Expected: 0 problems, with `README.md` listed as the one reviewed snapshot change.
`--stage validate` checks only the inputs.

**2. Analysis from saved predictions** (rewrites the tables in `artifacts/session8/`)

```bash
python -m src.run_session8 --config configs/session8_evaluation.yaml --stage derive
```

Test labels come from `results/session8/fd003_test_predictions.csv`, the record saved by the
evaluation, after checking it against the stored predictions.

**3. Figures and report** (rewrites figures 22-32, the report and the artifacts manifest)

```bash
python -m src.run_session8 --config configs/session8_evaluation.yaml --stage report
```

The artifacts manifest records the sources and Git state of whoever built it, so rebuilding it on
another commit changes the file and `verify` reports that it differs from the reviewed one. That is
intended: restore it, or have the new one reviewed and recorded in the reference.

Groups 2 and 3 refuse to run if the evidence manifest is missing or is not the reviewed one.

**4. A new experimental execution.** `predict`, `evaluate`, `sensitivity`, `register` and `all` run
inference, **read the FD003 test labels**, **train**, write MLflow runs or register evidence. They are
not needed to reproduce any published table or figure.

- They require `--authorize-execution`. The functions behind them require an explicit
  `ExecutionAuthorization` argument as well, so importing them does not skip the check.
- **They are refused for the completed execution.** Running it again would change the label ledger,
  tables and runs, so its seal would no longer match. A rerun needs a new configuration with its own
  `working_dir`, `artifacts_dir`, `figures_dir`, `report` and `experiment_name`, none overlapping the
  reviewed ones.
- `all` ends by registering the new evidence. `report` and `derive` for it stay refused until a
  reviewer adds that execution to the trusted reference.
- A rerun's label reads are not a fresh evaluation: FD003 test has been seen.

These checks prevent accidents. They are not a security boundary: whoever can run the code can edit it
or call the underlying libraries.

Tests: `python -m pytest`. Run them in a copy of the project if stores must not be touched.

## Sealed evidence and what it can show

`evidence_manifest.json` lists SHA-256 of 1,148 evidence files and a digest of each Session 8 MLflow
run. The trusted reference holds the SHA-256 of that manifest, of the artifacts manifest, and of three
further files that were **not** part of the seal (see the review note). A missing, replaced or edited
manifest is rejected, and no command writes one for a location the reference covers.

The limit: hashes detect change **relative to the reviewed reference**. Someone who can rewrite both
the files and the reference is not detected. The reference gains independent weight only once it is
committed and the commit is kept somewhere that person cannot rewrite. Both the manifest and the
reference were made **after** execution, so they say nothing about changes before that.

## Evaluation integrity

Label reads are recorded with the hashes of the config and protocol; a different config or protocol
is refused. These records are self-generated: they protect one working copy and cannot prove
chronology to a third party. FD003 test must be treated as seen in any later work.

## Provenance and publication

Session 8 was executed from an uncommitted working tree on commit `5416845`. The source files used
at execution were not recorded for predictions, so they cannot be reconstructed. The manifest keeps
`execution` (copied from the seal) apart from `derivation` (the sources that built the current
tables and report) and marks the results `"canonical": false`. Committing the files or rebuilding the
report changes neither fact, and no command upgrades it.

To publish: have the changed files reviewed; commit them as they are, with the honest manifest; record
the reviewer's decision and the reviewed commit in the review note.
Do not edit the manifests to claim a clean execution. A clean-commit execution would be a new,
separately authorized experiment with every output written to a new location; it would not make these
results clean, and its test-set reads would not be a fresh evaluation.

## Timing

No timing was measured in Session 8. Latency values are Session 7 measurements and depend on
hardware, power and load. Predictions are deterministic on the reference machine; other hardware or
library versions may differ in the last digits, which the exact-reproduction checks will report.
