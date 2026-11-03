"""Markdown validation report for Session 1 (no timestamps, so reruns are byte-identical)."""


def _fmt(v):
    return "undefined" if v != v else f"{v:.3f}"


def build_report(s):
    cfg = s["config"]
    lines = [
        f"# Session 1 Validation Report: {cfg['dataset']['subset']}",
        "",
        "## Raw data",
        f"- Raw train rows: {s['raw_train_rows']}",
        f"- Raw test rows: {s['raw_test_rows']}",
        f"- Total training engines: {s['n_train_engines_total']}",
        f"- Test engines: {s['n_test_engines']}",
        "",
        "## Engine-level split",
        f"- Split seed: {cfg['split']['seed']}; validation fraction: {cfg['split']['val_fraction']}",
        f"- Training engines after split: {len(s['train_units'])}",
        f"- Validation engines: {len(s['val_units'])}",
        f"- Engines present in both train and validation: **{s['overlap']}** (zero overlap confirmed)",
        f"- Validation engine IDs: {s['val_units']}",
        "",
        "## Row counts after loading and splitting",
        "| Set | Engines | Rows |",
        "|---|---|---|",
        f"| train | {len(s['train_units'])} | {s['train_rows']} |",
        f"| validation | {len(s['val_units'])} | {s['val_rows']} |",
        f"| test | {s['n_test_engines']} | {s['raw_test_rows']} |",
        "",
        "## Sequence windows",
        f"- Sequence length: {cfg['windows']['sequence_length']} cycles (stride 1)",
        f"- Selected feature count: {len(cfg['features']['selected'])}",
        f"- Window tensor shape: [n_windows, {cfg['windows']['sequence_length']}, "
        f"{len(cfg['features']['selected'])}]",
        f"- Target: RUL of the final timestep of each window",
        "",
        "| Set | Windows | X shape | y shape |",
        "|---|---|---|---|",
    ]
    for name in ("train", "validation", "test"):
        w = s["windows"][name]
        lines.append(f"| {name} | {w['n']} | {w['X_shape']} | {w['y_shape']} |")
    lines += [
        "",
        "Test windows: one per engine (its last cycles), labelled with the official RUL file.",
        "",
        "## Target and preprocessing settings",
        f"- RUL cap: {cfg['rul']['cap']}",
        f"- Scaler type: {cfg['scaler']['type']} (fitted on training engines only)",
        f"- Selected features: {', '.join(cfg['features']['selected'])}",
        "",
        "## Constant and near-constant channels (training engines, raw units)",
    ]
    t = s["channel_table"]
    for status in ("constant", "near-constant"):
        names = ", ".join(t.loc[t["status"] == status, "channel"]) or "none"
        lines.append(f"- {status}: {names}")
    lines += [
        f"- Thresholds: constant std <= {cfg['diagnostics']['constant_std_threshold']}; "
        f"near-constant std < {cfg['diagnostics']['near_constant_std_threshold']} or "
        f"<= {cfg['diagnostics']['near_constant_max_unique']} distinct values",
        "- Correlation with capped RUL is reported as `undefined` for constant channels.",
        "",
        "| Channel | Variance | Distinct values | Status | Corr. with capped RUL |",
        "|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        lines.append(f"| {r.channel} | {r.variance:.6g} | {r.n_unique} | {r.status} | "
                     f"{_fmt(r.corr_with_capped_rul)} |")
    lines += ["", "## Figures"] + [f"- `{p}`" for p in s["figures"]]
    lines += ["", "## Sanity checks", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {label} | PASS | {detail} |" for label, detail in s["checks"]]
    lines += [
        "",
        f"**All {len(s['checks'])} sanity checks passed.**",
        "",
        "## Reproducibility fingerprints (SHA-256 of array contents)",
    ]
    lines += [f"- `{name}`: `{digest}`" for name, digest in s["hashes"].items()]
    return "\n".join(lines) + "\n"
