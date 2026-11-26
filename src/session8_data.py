"""Session 8 inputs: configuration, FD003 preprocessing with the frozen FD001 definitions, the
FD001 validation reference, and the guarded one-time access to the FD003 test labels.

FD003 inputs are prepared without any label. Only `load_test_labels` reads RUL_FD003.txt, and it
records every access together with the protocol under which it happened.
"""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .benchmark_config import resolve_path
from .checks import (SanityCheckError, check_canonical_columns, check_counts, check_cycles, check_no_missing,
                     check_raw_column_count)
from .data_loading import SENSOR_COLUMNS, apply_schema, read_raw_table, read_rul_file
from .rul import add_rul
from .scaling import apply_scaler
from .session7_execution import atomic_write_json
from .session7_plan import sha256_file
from .session8_guard import REFERENCE_PATH, require_authorization, require_new_location
from .splitting import select_units
from .windows import make_test_windows, make_training_windows

PROTOCOL_PATH = "reports/session8_evaluation_protocol.md"
TRAJECTORY_SETS = ("fd001_validation", "fd003_train")
TEST_SET = "fd003_test"
DATASETS = (*TRAJECTORY_SETS, TEST_SET)


@dataclass(frozen=True)
class Windows:
    X: np.ndarray                    # float32 [n, 30, 14], C-contiguous
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    y_raw: np.ndarray = None         # uncapped true RUL; None for the FD003 test inputs


def load_config(path):
    path = resolve_path(path)
    cfg = yaml.safe_load(path.read_text())
    for key in ("upstream", "fd003", "evaluation", "candidates", "preferred_reduced", "statistics",
                "error_analysis", "sensitivity", "outputs"):
        if key not in cfg:
            raise ValueError(f"{path} is missing section '{key}'")
    keys = [c["key"] for c in cfg["candidates"]]
    if len(set(keys)) != len(keys):
        raise ValueError("candidate keys are not unique")
    named = [*cfg["statistics"]["families"]["A_reduced_vs_full"], *cfg["statistics"]["families"]["B_full_vs_reference"],
             *cfg["sensitivity"]["configurations"], cfg["preferred_reduced"]]
    if set(named) - set(keys):
        raise ValueError(f"unknown candidate keys: {sorted(set(named) - set(keys))}")
    cap, stages = cfg["evaluation"]["rul_cap"], cfg["evaluation"]["stages"]
    if stages[0]["low"] != 0 or stages[-1]["high"] is not None or any(
            b["low"] != a["high"] + 1 for a, b in zip(stages, stages[1:])):
        raise ValueError("RUL stages must be contiguous, start at 0 and end open")
    if stages[-1]["low"] > cap:
        raise ValueError("the last stage starts above the RUL cap")
    cfg["_path"] = path
    cfg["_reference"] = REFERENCE_PATH           # fixed here: a configuration cannot choose its own reference
    return cfg


def sensor_list(cfg):
    s1 = yaml.safe_load(resolve_path(cfg["upstream"]["session1_config"]).read_text())
    return list(s1["features"]["selected"]), int(s1["windows"]["sequence_length"]), int(s1["rul"]["cap"])


def load_scaler(cfg):
    return joblib.load(resolve_path(cfg["upstream"]["processed_dir"]) / "scaler.joblib")


def read_fd(path, name, expected_engines, expected_rows, features, seq_len):
    """A raw C-MAPSS file with the Session 1 schema checks; every engine must fill one window."""
    raw = read_raw_table(path)
    check_raw_column_count(raw, name)
    df = apply_schema(raw)
    check_canonical_columns(df, name)
    check_no_missing(df, name)
    check_cycles(df, name)
    check_counts(df, name, expected_engines, expected_rows)
    if not np.isfinite(df[features].to_numpy(dtype="float64")).all():
        raise SanityCheckError(f"{name} contains non-finite sensor values")
    shortest = int(df.groupby("unit_id").size().min())
    if shortest < seq_len:
        raise SanityCheckError(f"{name}: an engine has {shortest} cycles, fewer than the window length {seq_len}")
    return df


def fd003_frames(cfg):
    """The unscaled FD003 train and test frames, files checked against the declared hashes."""
    spec = cfg["fd003"]
    features, seq_len, _ = sensor_list(cfg)
    out = {}
    for split in ("train", "test"):
        path = resolve_path(spec["raw_dir"]) / spec["files"][split]
        if sha256_file(path) != spec["sha256"][split]:
            raise ValueError(f"{path} differs from the declared SHA-256")
        out[split] = read_fd(path, spec["files"][split], spec["expected"][f"{split}_engines"],
                             spec["expected"][f"{split}_rows"], features, seq_len)
    return out


def trajectory_windows(frame, scaler, features, seq_len, cap):
    """Every stride-1 window of run-to-failure engines, scaled with the given (already fitted) scaler."""
    ws = make_training_windows(apply_scaler(scaler, add_rul(frame, cap), features), features, seq_len)
    return Windows(np.ascontiguousarray(ws.X), ws.unit_ids, ws.end_cycles, ws.y_raw.astype("float64"))


def terminal_windows(frame, scaler, features, seq_len, cap):
    """The last window of each truncated test engine. No labels are involved."""
    n = frame["unit_id"].nunique()
    ws = make_test_windows(apply_scaler(scaler, frame, features), features, seq_len, np.zeros(n), cap)
    return Windows(np.ascontiguousarray(ws.X), ws.unit_ids, ws.end_cycles, None)


def fd001_validation(cfg):
    """FD001 validation windows rebuilt from the raw file and required to equal the frozen arrays,
    which proves that the preprocessing applied to FD003 is the Session 1 preprocessing."""
    features, seq_len, cap = sensor_list(cfg)
    processed = resolve_path(cfg["upstream"]["processed_dir"])
    s1 = yaml.safe_load(resolve_path(cfg["upstream"]["session1_config"]).read_text())
    split = json.loads((processed / "split.json").read_text())
    units = split.get("val_units") or split.get("validation_units")
    frame = apply_schema(read_raw_table(resolve_path(s1["dataset"]["raw_dir"]) / "train_FD001.txt"))
    built = trajectory_windows(select_units(frame, units), load_scaler(cfg), features, seq_len, cap)
    with np.load(processed / "val_windows.npz") as z:
        for name, mine in (("val_X", built.X), ("val_unit_ids", built.unit_ids), ("val_end_cycles", built.end_cycles),
                           ("val_y_raw", built.y_raw.astype("float32"))):
            if not np.array_equal(z[name], mine):
                raise ValueError(f"rebuilt FD001 validation windows differ from the frozen {name}")
        frozen_y = z["val_y"]
    if not np.array_equal(frozen_y, np.minimum(built.y_raw, cap).astype("float32")):
        raise ValueError("capped targets differ from the frozen targets")
    return built


def load_inputs(cfg):
    """{dataset: Windows} for the three evaluated sets. The FD003 test set carries no labels."""
    features, seq_len, cap = sensor_list(cfg)
    if cap != cfg["evaluation"]["rul_cap"]:
        raise ValueError("evaluation.rul_cap differs from the Session 1 cap")
    frames, scaler = fd003_frames(cfg), load_scaler(cfg)
    return {"fd001_validation": fd001_validation(cfg),
            "fd003_train": trajectory_windows(frames["train"], scaler, features, seq_len, cap),
            "fd003_test": terminal_windows(frames["test"], scaler, features, seq_len, cap)}


def windows_digest(windows):
    h = hashlib.sha256()
    for array in (windows.X, windows.unit_ids, windows.end_cycles):
        h.update(f"{array.dtype}{array.shape}".encode() + np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def input_shift(cfg):
    """FD003 training inputs expressed in FD001-scaled units (FD001 training rows are mean 0, sd 1),
    and the channels Session 1 dropped. Uses sensor values only."""
    features, _, _ = sensor_list(cfg)
    scaler = load_scaler(cfg)
    frame = fd003_frames(cfg)["train"]
    scaled = scaler.transform(frame[features].to_numpy(dtype="float64"))
    s1 = yaml.safe_load(resolve_path(cfg["upstream"]["session1_config"]).read_text())
    fd001 = apply_schema(read_raw_table(resolve_path(s1["dataset"]["raw_dir"]) / "train_FD001.txt"))
    test = fd003_frames(cfg)["test"]                       # inputs only; no label is involved
    last = scaler.transform(test.groupby("unit_id").tail(s1["windows"]["sequence_length"])[features]
                            .to_numpy(dtype="float64"))
    rows = [{"sensor": s, "used_by_models": True,
             "fd003_test_windows_mean_in_fd001_sd": float(last[:, i].mean()),
             "fd003_test_windows_sd_in_fd001_sd": float(last[:, i].std()),
             "fd003_test_windows_fraction_beyond_3_sd": float((np.abs(last[:, i]) > 3).mean()), "fd003_mean_in_fd001_sd": float(scaled[:, i].mean()),
             "fd003_sd_in_fd001_sd": float(scaled[:, i].std()),
             "fraction_beyond_3_sd": float((np.abs(scaled[:, i]) > 3).mean()),
             "fd001_raw_sd": float(fd001[s].std(ddof=0)), "fd003_raw_sd": float(frame[s].std(ddof=0))}
            for i, s in enumerate(features)]
    rows += [{"sensor": s, "used_by_models": False, "fd003_mean_in_fd001_sd": "", "fd003_sd_in_fd001_sd": "",
              "fraction_beyond_3_sd": "", "fd001_raw_sd": float(fd001[s].std(ddof=0)),
              "fd003_raw_sd": float(frame[s].std(ddof=0))} for s in SENSOR_COLUMNS if s not in features]
    return rows


def dataset_facts(cfg, inputs):
    """Engine, row and window counts and lifetimes, from inputs only."""
    frames = fd003_frames(cfg)
    s1 = yaml.safe_load(resolve_path(cfg["upstream"]["session1_config"]).read_text())
    fd001 = apply_schema(read_raw_table(resolve_path(s1["dataset"]["raw_dir"]) / "train_FD001.txt"))
    out = {}
    for name, frame in (("fd001_train_file", fd001), ("fd003_train_file", frames["train"]),
                        ("fd003_test_file", frames["test"])):
        lengths = frame.groupby("unit_id").size()
        out[name] = {"engines": int(len(lengths)), "rows": int(len(frame)), "min_cycles": int(lengths.min()),
                     "median_cycles": float(lengths.median()), "max_cycles": int(lengths.max()),
                     "operating_setting_sd": [float(frame[f"operational_setting_{i}"].std()) for i in (1, 2, 3)]}
    out["windows"] = {k: {"n_windows": int(len(w.X)), "n_engines": int(len(np.unique(w.unit_ids))),
                          "digest": windows_digest(w)} for k, w in inputs.items()}
    out["scaler_sha256"] = sha256_file(resolve_path(cfg["upstream"]["processed_dir"]) / "scaler.joblib")
    return out


# ---- guarded FD003 test labels -----------------------------------------------------------------

def protocol_identity(cfg):
    return {"config_sha256": sha256_file(cfg["_path"]), "protocol_sha256": sha256_file(resolve_path(PROTOCOL_PATH))}


def ledger_path(cfg):
    return resolve_path(cfg["outputs"]["working_dir"]) / "fd003_test_access.json"


def load_test_labels(cfg, purpose, *, authorization=None):
    """Raw official RUL of the 100 FD003 test engines. Every call is appended to the access ledger;
    a protocol or config different from the first access is refused. Refused without authorization
    and for a completed execution (its ledger is sealed evidence)."""
    require_authorization(authorization, "reading the FD003 test labels")
    require_new_location(cfg)
    path, identity = ledger_path(cfg), protocol_identity(cfg)
    ledger = json.loads(path.read_text()) if path.exists() else []
    if ledger and any(ledger[0][k] != v for k, v in identity.items()):
        raise RuntimeError("the FD003 test labels were first read under another protocol or config; "
                           "they cannot be evaluated again under a changed design")
    rul_path = resolve_path(cfg["fd003"]["raw_dir"]) / cfg["fd003"]["files"]["rul"]
    if sha256_file(rul_path) != cfg["fd003"]["sha256"]["rul"]:
        raise ValueError(f"{rul_path} differs from the declared SHA-256")
    ledger.append({**identity, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purpose": purpose})
    atomic_write_json(path, ledger)
    labels = read_rul_file(rul_path).astype("float64")
    if len(labels) != cfg["fd003"]["expected"]["test_engines"] or (labels < 0).any():
        raise ValueError("RUL file does not hold one non-negative value per test engine")
    return labels
