"""Read-only selection of the canonical Session 7 runs, for later sessions.

The published manifest names the canonical MLflow run of every condition. Tags alone are not enough:
`s7.verification=False` only says a run was not made from a dirty tree, so a later clean execution
would carry it too, and the verification runs are also tagged `s7.status=completed`.

    from src.session7_canonical import canonical_runs
    runs = canonical_runs("artifacts/session7/manifest.json", "mlruns")   # {experiment_id: run_id}

Nothing is written: the MLflow store is opened read-only.
"""
import json
from pathlib import Path

from .session7_integrity import RunStore

# the equivalent MLflow search filter (use together with the manifest's commit)
MLFLOW_FILTER = ("tags.`s7.verification` = 'False' and tags.`s7.status` = 'completed' "
                 "and params.git_commit = '{git_commit}'")


def canonical_runs(manifest_path, tracking_dir):
    """{experiment_id: MLflow run ID} of the runs published in `manifest_path`, each confirmed in the
    store. Raises ValueError if the manifest or any run does not qualify."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("git_dirty") is not False:
        raise ValueError("the manifest does not describe a clean-commit execution")
    conditions = manifest["conditions"]
    if len(conditions) != manifest["n_planned"] or manifest["n_completed"] != manifest["n_planned"]:
        raise ValueError("the manifest does not describe a complete matrix")
    store, out = RunStore(tracking_dir), {}
    for experiment_id, entry in conditions.items():
        run = store.get(entry["mlflow_run_id"])
        wrong = (run is None or run["status"] != "FINISHED" or run["lifecycle_stage"] != "active"
                 or run["experiment"] != manifest["mlflow_experiment"]
                 or run["tags"].get("s7.experiment_id") != experiment_id
                 or run["tags"].get("s7.verification") != "False"         # tags are stored as strings
                 or run["tags"].get("s7.status") != "completed"
                 or run["params"].get("git_commit") != manifest["git_commit"])
        if wrong:
            raise ValueError(f"run {entry['mlflow_run_id']} of {experiment_id} is not a canonical run")
        out[experiment_id] = entry["mlflow_run_id"]
    if len(set(out.values())) != len(out):
        raise ValueError("a run is listed for more than one condition")
    return out
