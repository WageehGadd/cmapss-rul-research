"""Read-only selection of the canonical Session 7 runs, for later sessions.

The canonical set is pinned here: the 174-condition frozen matrix, the clean commit that executed it
and the published manifest itself. A manifest that merely agrees with itself is not enough, and tags
alone are not either: `s7.verification=False` only says a run was not made from a dirty tree, so a
later clean execution would carry it too, and the verification runs are also tagged
`s7.status=completed`.

    from src.session7_canonical import canonical_runs
    runs = canonical_runs("artifacts/session7/manifest.json", "mlruns")   # {experiment_id: run_id}

The condition records and run folders under `working_dir` are git-ignored; they must be present and
unchanged. Nothing is written: the MLflow store is opened read-only.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .benchmark_config import resolve_path
from .evaluation import predictions_digest
from .session7_integrity import RunStore
from .session7_plan import sha256_file, sha256_json

# the equivalent MLflow search filter (use together with the manifest's commit)
MLFLOW_FILTER = ("tags.`s7.verification` = 'False' and tags.`s7.status` = 'completed' "
                 "and params.git_commit = '{git_commit}'")


@dataclass(frozen=True)
class Frozen:
    """What the canonical evidence must be. Only tests pass anything but CANONICAL."""
    n_conditions: int
    matrix_sha256: str          # matrix_digest() of the experiment plan
    git_commit: str             # clean commit that executed the matrix
    manifest_sha256: str        # the published manifest file


CANONICAL = Frozen(n_conditions=174,
                   matrix_sha256="782b3d1d725ad5bcd46c282c47e22b25d52ca0eac9a77c3284035addd4a08144",
                   git_commit="0b7d0b3e334d8a3453f56bc841159f373a50872c",
                   manifest_sha256="000a7ac84b5753f4cc946d708a550ea6af9689d47c940c4307c906e8d85b31b7")


def matrix_digest(conditions):
    """Digest of the (experiment ID, scientific identity) pairs of a plan, independent of order."""
    return sha256_json(sorted([c["experiment_id"], c["identity_sha256"]] for c in conditions))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _unique_keys(pairs):
    """json hook: a repeated key would otherwise silently replace the first entry."""
    keys = [k for k, _ in pairs]
    repeated = sorted({k for k in keys if keys.count(k) > 1})
    _require(not repeated, f"duplicated entries: {', '.join(repeated[:3])}")
    return dict(pairs)


def _read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_keys)


def _frozen_matrix(manifest_path, manifest, expected):
    """{experiment_id: identity} of the plan published beside the manifest, checked to be frozen."""
    plan_path = manifest_path.parent / "experiment_plan.json"
    _require(sha256_file(plan_path) == manifest["files"][plan_path.name],
             "the experiment plan differs from the one the manifest lists")
    conditions = _read_json(plan_path)["conditions"]
    matrix = {c["experiment_id"]: c["identity_sha256"] for c in conditions}
    _require(len(conditions) == len(matrix) == expected.n_conditions
             and matrix_digest(conditions) == expected.matrix_sha256,
             f"the experiment plan is not the frozen {expected.n_conditions}-condition matrix")
    return matrix


def _check_record(experiment_id, identity, entry, record_sha256, working_dir, expected):
    """The stored record and its files are those the manifest was published from."""
    path = working_dir / "conditions" / f"{experiment_id}.json"
    _require(sha256_file(path) == record_sha256, f"record of {experiment_id} differs from the manifest hash")
    record = _read_json(path)
    for key, value in (("experiment_id", experiment_id), ("identity_sha256", identity), ("status", "completed"),
                       ("mlflow_run_id", entry["mlflow_run_id"]), ("predictions_sha256", entry["predictions_sha256"]),
                       ("git_commit", expected.git_commit), ("git_dirty", False)):
        _require(record[key] == value, f"record of {experiment_id}: unexpected {key}")
    # located relative to working_dir, so a restored copy of the evidence can be checked too
    run_dir = working_dir / "runs" / entry["mlflow_run_id"]
    files = record["files"]
    _require("predictions.npz" in files and any(f.startswith("model/") for f in files),
             f"record of {experiment_id} does not reference predictions and model files")
    for name, digest in files.items():
        _require(sha256_file(run_dir / name) == digest, f"file of {experiment_id} changed: {name}")
    with np.load(run_dir / "predictions.npz") as z:
        _require(z.files == ["validation"] and predictions_digest(z["validation"]) == entry["predictions_sha256"],
                 f"predictions of {experiment_id} do not match the manifest")


def _check_run(experiment_id, identity, run_id, store, experiment_name, expected):
    run = store.get(run_id)
    wrong = (run is None or run["status"] != "FINISHED" or run["lifecycle_stage"] != "active"
             or run["experiment"] != experiment_name
             or run["tags"].get("s7.experiment_id") != experiment_id
             or run["tags"].get("s7.verification") != "False"             # tags are stored as strings
             or run["tags"].get("s7.status") != "completed"
             or run["params"].get("s7.identity_sha256") != identity
             or run["params"].get("git_commit") != expected.git_commit
             or run["params"].get("git_dirty") != "False")
    _require(not wrong, f"run {run_id} of {experiment_id} is not a canonical run")


def canonical_runs(manifest_path, tracking_dir, working_dir="results/session7", *, expected=CANONICAL):
    """{experiment_id: MLflow run ID} of the canonical execution. Relative paths are relative to the
    project root. Raises ValueError unless the manifest, the frozen matrix, every stored record with
    its files and every MLflow run qualify."""
    manifest_path = resolve_path(manifest_path)
    working_dir = resolve_path(working_dir)
    try:
        manifest = _read_json(manifest_path)
        _require(manifest["git_dirty"] is False and manifest["git_commit"] == expected.git_commit,
                 "the manifest does not describe the canonical clean-commit execution")
        _require(manifest["n_planned"] == manifest["n_completed"] == expected.n_conditions,
                 f"the manifest does not describe {expected.n_conditions} completed conditions")
        matrix = _frozen_matrix(manifest_path, manifest, expected)
        conditions, record_hashes = manifest["conditions"], manifest["record_files_sha256"]
        for name, found in (("conditions", conditions), ("record hashes", record_hashes)):
            missing, extra = sorted(set(matrix) - set(found)), sorted(set(found) - set(matrix))
            _require(not missing and not extra, f"manifest {name} differ from the frozen matrix: "
                     f"{len(missing)} missing {missing[:2]}, {len(extra)} unknown {extra[:2]}")
        out = {experiment_id: conditions[experiment_id]["mlflow_run_id"] for experiment_id in matrix}
        _require(len(set(out.values())) == len(out), "a run is listed for more than one condition")
        store = RunStore(resolve_path(tracking_dir))
        for experiment_id, identity in matrix.items():
            entry = conditions[experiment_id]
            _require(entry["status"] == "completed", f"manifest: {experiment_id} is not completed")
            _check_run(experiment_id, identity, out[experiment_id], store, manifest["mlflow_experiment"], expected)
            _check_record(experiment_id, identity, entry, record_hashes[experiment_id], working_dir, expected)
        # last, so that the checks above name what differs; this catches any other change
        _require(sha256_file(manifest_path) == expected.manifest_sha256, "this is not the published manifest")
    except (OSError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"canonical evidence is incomplete or malformed: {type(error).__name__}: {error}") from error
    return out
