# Session 8: integrity review note

Written 2026-11-26, after execution, in response to the second independent review (verdict
CONDITIONAL PASS: scientific findings resolved, integrity guards not). This note is publication
metadata. It is not execution evidence and changes nothing about how the results were produced.
Session 8 remains **not canonical**.

## What was fixed

| Re-review finding | Change |
|---|---|
| Deleting the seal and running the safe `seal` stage accepted forged predictions | The `seal` stage is removed. Registration is an authorized step of a new execution only and is refused for any location the reference covers or that holds a manifest. A manifest other than the reviewed one is rejected |
| A hand-edited `canonical: true` passed `verify` | Every manifest field is checked against the seal, the reference and the files; the manifest hash is pinned |
| Authorization only on the command line | The functions that train, infer, load models for inference, read test labels, write MLflow runs or register evidence require an `ExecutionAuthorization` |
| A rerun would silently break the seal | Execution into the completed execution's locations or MLflow experiment is refused |
| Three evidence-related files unsealed; stray files ignored | Hashes recorded below and checked; any other file in the working directory is reported |
| README and reproducibility guide inaccurate | Updated |

The evidence manifest was not rewritten. No model was trained or applied, no label file was parsed, no
MLflow run was created or changed, and no table, figure or report was regenerated.

## Trusted reference

`configs/session8_trusted_reference.json` holds the values below and is read, never written, by code.

| Item | SHA-256 |
|---|---|
| `artifacts/session8/evidence_manifest.json` (existing seal) | `6ee22d851e269d63c4bb278d95b5a5a85f4ff22d4000ddae87ae0f0d87738fd3` |
| `artifacts/session8/manifest.json` | `2821920e552b39f5b604b76bb99248f4a8ca594403a894858ae7607e64a62fbb` |

These were copied from the files as reviewed. The reference is not yet committed, so it has no
independent standing yet.

## Three files that were never part of the seal

All in `results/session8/` (Git-ignored). Hashes taken after execution, during review.

| File | Role | Limit | SHA-256 |
|---|---|---|---|
| `frozen_snapshot.json` | Pre-session hashes of 5,144 upstream files; `verify` checks them | Produced by a script that is not in the repository; unsealed until now, so a change before this note would not show | `670868f21fb1dad982415989225af2fb7bf84c89cf683bc2ac796c58534acb96` |
| `mlflow_before_session8.db` | Copy of the MLflow store before Session 8 (413 runs), used to show earlier runs unchanged | Local copy made by the same workflow; no automatic comparison in `verify` | `9fdb3e97baa8a57ac06da2112b53640ebfaf1d0af46fc7f5d6ceaa02ca6f8836` |
| `protocol_hashes_at_writing.txt` | Hashes of the first config and protocol drafts | The drafts themselves were not kept, and these differ from the final hashes | `441dc4509db353c1c710a442dffcbffb150c9eecf45e95ae39abfe94816cf70d` |

**Inventory:** 1,148 files in the seal + these 3 = 1,151 evidence-related files, the same set as the list
taken before the first remediation (`results/session8/remediation/local_evidence_before.sha`, SHA-256
`28817979ed188e92739f55bca3f43f37132a9e992c8bb120e6b65dd51f42c835`). Also present and deliberately
not evidence: `execution.lock` and the `remediation/` folder of pre-fix copies.

## Changes made after the results were derived

The artifacts manifest still names the sources that built the tables and figures. The guard changes
came later, so the reference lists each affected source with its old and new hash:
`session8_guard.py` (new), `session8_evidence.py`, `session8_data.py`, `session8_inference.py`,
`session8_sensitivity.py`, `run_session8.py`. `verify` re-derives every table with the new code.

`README.md` is one of the 5,144 files in the pre-session snapshot. It was changed on purpose to
document Session 8; the reference records its old and new hash. The other 5,143 are unchanged, and the
snapshot was not rewritten.

## Limits that remain

- All hashes and the seal were made after execution. They cannot show what happened before.
- Someone able to rewrite both the files and the reference is not detected.
- The authorization checks prevent accidents; they are not a security boundary.
- `results/session8/`, `mlruns/` and the Session 7 evidence are not in Git. Copy them off this disk.
- Redistribution terms of the NASA files were not checked.

## Review record

| Field | Value |
|---|---|
| Focused re-review of these fixes | pending |
| Reviewer decision | pending |
| Reviewed commit | none; nothing is committed |
