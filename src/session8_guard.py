"""Session 8 guards for the operations that execute the experiment: explicit authorization, and the
rule that a completed execution is never executed into again.

These stop accidental execution through Session 8 code. They are not a security boundary: anyone who
can run this code can also edit it, or call the underlying libraries directly.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from .benchmark_config import resolve_path

EVIDENCE_NAME = "evidence_manifest.json"
REFERENCE_PATH = "configs/session8_trusted_reference.json"     # reviewed and versioned; never written by code
LOCATION_KEYS = ("working_dir", "artifacts_dir", "figures_dir", "report")


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Passed explicitly to every function that trains, infers, reads test labels, writes MLflow runs
    or registers evidence. There is no global switch and no default."""
    reason: str


def require_authorization(authorization, operation):
    if not isinstance(authorization, ExecutionAuthorization) or not authorization.reason:
        raise PermissionError(f"{operation} executes the experiment and needs an explicit ExecutionAuthorization "
                              "(command line: --authorize-execution). Saved results can be verified and analysed "
                              "without it.")


def reference_entries(cfg):
    """The reviewed executions: those of the project's reference and of the one `cfg` points to. A
    missing or unreadable reference is an error, never an empty list."""
    paths = {resolve_path(REFERENCE_PATH), resolve_path(cfg.get("_reference", REFERENCE_PATH))}
    out = []
    for path in sorted(paths):
        if not path.exists():
            raise RuntimeError(f"the trusted reference {path} is missing; nothing is executed or registered")
        out += json.loads(path.read_text())["executions"]
    return out


def _overlap(a, b):
    """Whether one path is the other or lies inside it (case-insensitive, links resolved)."""
    a, b = (Path(str(resolve_path(p).resolve()).casefold()) for p in (a, b))
    return a == b or a in b.parents or b in a.parents


def require_new_location(cfg):
    """A new execution needs outputs that hold no completed execution and, if it names an MLflow
    experiment, one that no reviewed execution used. Nothing is written by this check."""
    outputs = cfg["outputs"]
    mine = [outputs[k] for k in LOCATION_KEYS if k in outputs]
    refuse = ("a completed execution is never run into or registered again; a new execution needs its own "
              "working_dir, artifacts_dir, figures_dir, report and experiment_name")
    for entry in reference_entries(cfg):
        for new in mine:
            for key in LOCATION_KEYS:
                if _overlap(new, entry[key]):
                    raise RuntimeError(f"{new} overlaps the reviewed execution's {key} ({entry[key]}): {refuse}")
        if cfg.get("sensitivity", {}).get("experiment_name") == entry["experiment_name"]:
            raise RuntimeError(f"MLflow experiment {entry['experiment_name']} belongs to a reviewed execution: {refuse}")
    if "artifacts_dir" in outputs:
        for name in (EVIDENCE_NAME, "manifest.json"):
            if (resolve_path(outputs["artifacts_dir"]) / name).exists():
                raise RuntimeError(f"{outputs['artifacts_dir']}/{name} exists: {refuse}")
