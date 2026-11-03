"""Session 1 entry point: raw FD001 files -> leakage-safe supervised RUL dataset.

Usage (from the project root):
    python -m src.run_session1 --config configs/session1_fd001.yaml
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

from . import checks, diagnostics
from .checks import CheckLog, SanityCheckError
from .config import load_config
from .data_loading import CANONICAL_COLUMNS, SENSOR_COLUMNS, apply_schema, read_raw_table, read_rul_file
from .report import build_report
from .rul import add_rul
from .scaling import apply_scaler, fit_scaler
from .splitting import select_units, split_units
from .windows import make_test_windows, make_training_windows


def _digest(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(f"{arr.dtype}{arr.shape}".encode() + arr.tobytes()).hexdigest()


def _load_split(raw_dir, kind, subset, log, expected_engines, expected_rows):
    name = f"{kind}_{subset}.txt"
    raw = read_raw_table(raw_dir / name)
    log.run(f"B. {name} has exactly 26 columns", checks.check_raw_column_count, raw, name)
    df = apply_schema(raw)
    log.run(f"A. {name} canonical columns", checks.check_canonical_columns, df, name)
    log.run(f"C. {name} no missing values", checks.check_no_missing, df, name)
    log.run(f"E/F. {name} cycles", checks.check_cycles, df, name)
    log.run(f"H. {name} expected engine/row counts", checks.check_counts, df, name,
            expected_engines, expected_rows)
    return df


def run(config_path):
    cfg, root = load_config(config_path)
    log = CheckLog()
    ds, rul_cfg, feats = cfg["dataset"], cfg["rul"], cfg["features"]["selected"]
    seq_len, subset, rul_cap = cfg["windows"]["sequence_length"], ds["subset"], rul_cfg["cap"]
    raw_dir, out = root / ds["raw_dir"], cfg["outputs"]

    unknown = [f for f in feats if f not in CANONICAL_COLUMNS[2:] or feats.count(f) > 1]
    if unknown:
        raise SanityCheckError(f"features.selected has unknown or duplicate columns: {unknown}")

    # 1. Load raw data with the canonical schema
    train = _load_split(raw_dir, "train", subset, log, ds["expected_train_engines"],
                        ds["expected_train_rows"])
    test = _load_split(raw_dir, "test", subset, log, ds["expected_test_engines"],
                       ds["expected_test_rows"])
    official_rul = read_rul_file(raw_dir / f"RUL_{subset}.txt")
    log.run("H. official RUL file matches test engines", checks.check_test_rul_file, test, official_rul)
    bad_examples = [u for u in cfg["diagnostics"]["example_units"] if u not in set(train["unit_id"])]
    if bad_examples:
        raise SanityCheckError(f"diagnostics.example_units not in training data: {bad_examples}")

    # 2. Training RUL (raw and capped)
    train = add_rul(train, rul_cap)
    log.run("RUL derivation and cap", checks.check_rul, train, rul_cap)

    # 3. Engine-level split
    all_units = sorted(train["unit_id"].unique().tolist())
    train_units, val_units = split_units(all_units, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    train_df, val_df = select_units(train, train_units), select_units(train, val_units)
    log.run("G. no train/validation unit_id overlap", checks.check_split, all_units, train_units,
            val_units, train_df, val_df)

    # 4. Scaling: fit on training engines only, apply to validation and test
    log.run("Selected features vary on training engines", checks.check_features_variable, train_df,
            feats, cfg["diagnostics"]["constant_std_threshold"])
    scaler, fit_units = fit_scaler(train_df, feats, cfg["scaler"]["type"])
    log.run("L. scaler fitted on training engines only", checks.check_scaler_fit_on_train_only,
            scaler, fit_units, train_units, val_units, train_df, feats)
    train_s, val_s, test_s = (apply_scaler(scaler, d, feats) for d in (train_df, val_df, test))
    for name, d in (("train", train_s), ("validation", val_s), ("test", test_s)):
        log.run(f"D. no NaN/inf after scaling ({name})", checks.check_finite,
                d[feats].to_numpy(), f"scaled {name} features")

    # 5. Sequence windows
    train_w = make_training_windows(train_s, feats, seq_len)
    val_w = make_training_windows(val_s, feats, seq_len)
    test_w = make_test_windows(test_s, feats, seq_len, official_rul, rul_cap)
    test_units = sorted(test["unit_id"].unique())
    test_targets = {(u, int(test.loc[test["unit_id"] == u, "cycle"].max())):
                    (int(r), min(int(r), rul_cap)) for u, r in zip(test_units, official_rul)}
    for name, ws, d, one in (("train", train_w, train_s, False), ("validation", val_w, val_s, False),
                             ("test", test_w, test_s, True)):
        log.run(f"I. window shapes ({name})", checks.check_window_shapes, ws, name, d, seq_len,
                len(feats), one)
        log.run(f"D. no NaN/inf in windows ({name})", checks.check_finite, ws.X, f"{name} X")
        targets = test_targets if one else {(u, c): (a, b) for u, c, a, b in zip(
            d["unit_id"], d["cycle"], d["rul_raw"], d["rul"])}
        log.run(f"J/K. window boundaries and target alignment ({name})",
                checks.check_windows_match_source, ws, d, feats, seq_len, targets, name)
    log.run("G. no engine contributes windows to both train and validation",
            checks.check_window_engines_disjoint, train_w, val_w)

    # 6. Diagnostics (training engines, raw units) and figures
    dpi, fig_dir = cfg["diagnostics"]["figure_dpi"], root / out["figures_dir"]
    table = diagnostics.channel_table(train_df, cfg["diagnostics"])
    root.joinpath(out["diagnostics_csv"]).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / out["diagnostics_csv"], index=False, float_format="%.10g")
    lifetimes = train.groupby("unit_id")["cycle"].max()
    figs = [
        diagnostics.plot_lifetime_distribution(lifetimes, train_units, val_units,
                                               fig_dir / "fig1_engine_lifetime_distribution.png", dpi),
        diagnostics.plot_sensor_trajectories(train, cfg["diagnostics"]["example_units"],
                                             cfg["diagnostics"]["example_sensors"],
                                             fig_dir / "fig2_sensor_trajectories.png", dpi),
        diagnostics.plot_sensor_variance(table, fig_dir / "fig3_channel_variance.png", dpi),
        diagnostics.plot_rul_correlation(table, rul_cap, fig_dir / "fig4_correlation_with_capped_rul.png", dpi),
        diagnostics.plot_rul_cap_example(train, cfg["diagnostics"]["example_units"][0], rul_cap,
                                         fig_dir / "fig5_rul_cap_example.png", dpi),
    ]

    # 7. Save processed arrays, split, scaler and resolved config
    proc = root / out["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, ws in (("train", train_w), ("val", val_w), ("test", test_w)):
        arrays = {f"{name}_X": ws.X, f"{name}_y": ws.y, f"{name}_y_raw": ws.y_raw,
                  f"{name}_unit_ids": ws.unit_ids, f"{name}_end_cycles": ws.end_cycles}
        hashes.update({k: _digest(v) for k, v in arrays.items()})
        np.savez(proc / f"{name}_windows.npz", **arrays)
    split = {"seed": cfg["split"]["seed"], "train_units": train_units, "val_units": val_units}
    (proc / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    joblib.dump(scaler, proc / "scaler.joblib")
    (proc / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (proc / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")

    # 8. Validation report (only reached if every check passed)
    shape = lambda a: "x".join(str(d) for d in a.shape)  # noqa: E731
    summary = {
        "config": cfg, "checks": log.results, "hashes": hashes, "channel_table": table,
        "figures": [str(Path(p).relative_to(root)) for p in figs],
        "raw_train_rows": len(train), "raw_test_rows": len(test),
        "n_train_engines_total": len(all_units), "n_test_engines": len(test_units),
        "train_units": train_units, "val_units": val_units, "overlap": len(set(train_units) & set(val_units)),
        "train_rows": len(train_df), "val_rows": len(val_df),
        "windows": {n: {"n": len(w.X), "X_shape": shape(w.X), "y_shape": shape(w.y)}
                    for n, w in (("train", train_w), ("validation", val_w), ("test", test_w))},
    }
    report_path = root / out["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Session 1 RUL dataset pipeline.")
    parser.add_argument("--config", default="configs/session1_fd001.yaml")
    args = parser.parse_args()
    try:
        s = run(args.config)
    except SanityCheckError as exc:
        print(f"SESSION 1 FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    w = s["windows"]
    print(f"Session 1 complete: {len(s['checks'])} sanity checks passed")
    print(f"  engines train/val/test: {len(s['train_units'])}/{len(s['val_units'])}/{s['n_test_engines']}")
    print(f"  windows train/val/test: {w['train']['n']}/{w['validation']['n']}/{w['test']['n']} "
          f"(X shape {w['train']['X_shape']})")
    print(f"  report: {s['config']['outputs']['report_path']}")


if __name__ == "__main__":
    main()
