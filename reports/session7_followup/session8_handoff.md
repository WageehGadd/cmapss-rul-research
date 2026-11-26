# Session 7 to Session 8 handoff

## Four different things

| Term | Meaning here |
|---|---|
| Verification run | Executed from an uncommitted tree (`git_dirty=True`). 180 MLflow runs tagged `s7.verification=True`, 6 of them also `s7.status=superseded`. Never canonical. |
| Clean-commit execution | Executed from a committed tree: the 174 runs of `0b7d0b3`, `s7.verification=False`. Being clean makes runs eligible, not canonical. |
| Canonical evidence | The records, predictions, models and MLflow runs of that execution, named in `artifacts/session7/manifest.json`. |
| Frozen published results | The report, figures and artifacts committed in `21ec708`. Never regenerated in place. |

The published manifest has `"canonical": null`: the pipeline does not grant that status. It was
given by review and commit. Manifests written by the current code add `execution_class`
(`verification`, `clean_commit`, `unknown`) and `canonical_note`; the published one is unchanged.

## Loading the canonical runs

```python
from src.session7_canonical import canonical_runs

runs = canonical_runs("artifacts/session7/manifest.json", "mlruns")   # {experiment_id: run_id}, 174
```

It reads the store read-only and fails unless all of the following hold:

- the manifest is the published one, for commit `0b7d0b3`, listing exactly the 174 conditions of the
  frozen matrix (IDs and scientific identities), each once and with its own run;
- every condition record under `results/session7/` has the hash the manifest lists and agrees with it,
  and its model and prediction files are unchanged;
- every run is FINISHED, active, tagged `s7.verification` = `"False"` and `s7.status` = `"completed"`,
  and carries the condition identity and the clean commit.

The expected matrix, commit and manifest hash are constants in `src/session7_canonical.py`. A third
argument points at a restored copy of `results/session7`. MLflow tags are strings: compare with
`'False'`, not a boolean. The equivalent tag search (a filter only: it checks none of the above):

```
tags.`s7.verification` = 'False' and tags.`s7.status` = 'completed'
and params.git_commit = '0b7d0b3e334d8a3453f56bc841159f373a50872c'
```

Do not filter on `s7.status = 'completed'` alone: that returns 348 runs, 174 of them verification
runs. `s7.verification='False'` alone returns the 174 today, but a later clean execution would match
it too, so select through the manifest.

Condition records are in `results/session7/conditions/`, predictions and models under
`results/session7/runs/<run_id>/`. Both, and `mlruns/`, are git-ignored.

## Earlier gate-failure record

`reports/session7_incidents/gradient_boosting_memory_layout.json` gives
`original_record.path = results/session7/gate_diagnosis.json`. That file was moved with the
verification outputs and is not in the project folder now. It is at

`~/cmapss-rul-research-session7-verification-outputs-20261126/results/session7/gate_diagnosis.json`

SHA-256 `89394e292f444407646286d18a3a3e7b50f681d657404e926435f1af7b90d535`, equal to the value in
the incident record (checked on 2026-11-26). The incident record is unchanged.

## What the hardening changed in the code

- A completed record must hold every required field with a valid value; deep models also the weight
  digest and training summary. All 174 canonical records pass.
- Git state comes from `session7_integrity.git_state`, which reports `unavailable` when git cannot
  answer; writing stages then refuse. `src/environment.py` belongs to Session 2 and is unchanged, so
  runners of Sessions 1-6 still read a failed `git status` as clean.
- The code fingerprint now includes `checks`, `data_loading`, `rul`, `scaling`, `splitting` and
  `windows`.
- `report` and `publish` require a passed full-sensor gate.
- Writing stages refuse configured outputs of another experiment or execution.

**Consequence:** the code fingerprint and the commit differ from those in the canonical
records, so `run`, `resume`, `summarize`, `report` and `all` refuse to touch `results/session7`.
That is intended. `validate`, `verify` and `publish` (into a new folder) still work. Session 8 must
use its own working directory, outputs and MLflow experiment.

## Backups

All copies known on 2026-11-26 are on the same Mac and disk, so none protects against disk loss:
`~/cmapss-rul-research-session7-canonical-backup-20261126-174908`,
`~/cmapss-rul-research-session7-verification-outputs-20261126`,
`~/cmapss-rul-research-session7-evidence-20261126` (predates the canonical run).
