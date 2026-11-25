"""Figures and report for Session 6. Interpretation text and caveats are generated from run data,
so they survive reruns."""
from pathlib import Path

from .session4_outputs import (file_sha256, freeze_checkpoint, markdown_table,  # noqa: F401  (re-exported)
                               plot_prediction_vs_target, plot_training_curves, write_json)

MARKERS = {"random_forest": "s", "gradient_boosting": "D", "lstm": "^", "gru": "v", "tcn": "o", "transformer": "*"}
COLORS = {"classical": "tab:green", "recurrent": "tab:orange", "convolutional": "tab:blue", "attention": "tab:red"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_architecture_study(path, records, selected_name):
    """Validation RMSE of every core candidate, annotated with its parameter count."""
    plt = _pyplot()
    labels = [r["run_name"].replace("tf_", "") for r in records]
    values = [r["validation"]["rmse"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(records) + 1.4))
    colors = ["tab:red" if r["run_name"] == selected_name else "tab:gray" for r in records]
    ax.barh(range(len(records)), values, color=colors)
    ax.set_yticks(range(len(records)), labels, fontsize=8)
    ax.invert_yaxis()
    for i, r in enumerate(records):
        ax.text(values[i] + 0.1, i, f"{values[i]:.3f}  ({r['n_parameters']:,} par.)", va="center", fontsize=7)
    ax.set_xlim(0, max(values) * 1.35)
    ax.set_xlabel("validation RMSE (cycles)")
    ax.set_title("Transformer candidates (red = selected)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def latency_ticks(low, high):
    """Sparse, plain-number ticks for the log latency axis (no crowded 3x10^0 style labels)."""
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if low <= t <= high]
    return ticks or [round(low, 1), round(high, 1)]


def format_latency_axis(ax, latencies):
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
    ax.set_xlim(min(latencies) / 1.3, max(latencies) * 1.3)
    ticks = latency_ticks(*ax.get_xlim())
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_family_rmse_vs_latency(path, points):
    """Validation RMSE vs latency. Models are identified in the legend (no point labels, so nothing
    can overlap). points: [{model, family, latency, rmse, n_parameters}]."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for p in sorted(points, key=lambda q: list(MARKERS).index(q["model"])):
        count = f"{p['n_parameters']:,} " + ("tree nodes" if p["family"] == "classical" else "par.")
        ax.scatter([p["latency"]], [p["rmse"]], s=150 if p["model"] == "transformer" else 70,
                   c=COLORS[p["family"]], marker=MARKERS[p["model"]], edgecolors="k", linewidths=0.5,
                   label=f"{p['model']}: {p['rmse']:.3f}, {p['latency']:.2f} ms ({count})")
    ax.set_xscale("log")
    ax.margins(x=0.1, y=0.1)
    format_latency_axis(ax, [p["latency"] for p in points])
    ax.set_xlabel("inference latency per batch of 256 windows (ms, log scale)")
    ax.set_ylabel("validation RMSE (cycles)")
    ax.set_title("Full-sensor models: validation RMSE vs inference latency")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def latency_notes(comparison):
    lat = {r["model"]: float(r["latency_ms"]) for r in comparison}
    mine = lat.get("transformer")
    notes = []
    for other, value in lat.items():
        if mine is None or other == "transformer" or value <= 0:
            continue
        ratio = mine / value
        notes.append(f"{ratio:.1f}x the latency of {other}" if ratio >= 1 else f"{1 / ratio:.1f}x faster than {other}")
    return notes


def comparison_note(comparison, tolerance):
    """Plain statement of where the Transformer ranks on validation RMSE, from this run's numbers."""
    ordered = sorted(comparison, key=lambda r: r["val_rmse"])
    names = [r["model"] for r in ordered]
    mine = next(r for r in ordered if r["model"] == "transformer")
    best = ordered[0]
    rank = names.index("transformer") + 1
    text = f"On validation RMSE the Transformer ranks {rank} of {len(ordered)} ({mine['val_rmse']:.3f})"
    if rank == 1:
        second = ordered[1]
        margin = second["val_rmse"] / mine["val_rmse"] - 1
        text += f"; the next model is {second['model']} at {second['val_rmse']:.3f} ({margin:.1%} higher)"
        within = margin <= tolerance
    else:
        margin = mine["val_rmse"] / best["val_rmse"] - 1
        text += f"; the lowest is {best['model']} at {best['val_rmse']:.3f} (the Transformer is {margin:.1%} higher)"
        within = margin <= tolerance
    text += (". This gap is within the 1% equivalence band and, from one seed, should not be read as a difference."
             if within else ". With one seed per model, this ranking is descriptive, not a statistical result.")
    return text


FAMILY_NOTE = {"random_forest": "classical", "gradient_boosting": "classical"}


def head_count_spread(stage_a_records):
    """Largest validation-RMSE difference between Stage A candidates that differ only in head count."""
    groups = {}
    for r in stage_a_records:
        groups.setdefault((r["d_model"], r["num_layers"]), []).append(r["validation"]["rmse"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) > 1]
    return max(spreads) if spreads else None


def interpretation_notes(comparison, historical_test_rmse, transformer_test_rmse, stage_a_records):
    """Cross-family caveats derived from the frozen historical records and this run. Test values are
    previously recorded observations and are not used for any choice."""
    mine = next(r for r in comparison if r["model"] == "transformer")
    others = [r for r in comparison if r["model"] != "transformer"]
    if not others:
        return ["No historical records were available for comparison."]
    ref = min(others, key=lambda r: r["val_rmse"])
    name, margin = ref["model"], ref["val_rmse"] - mine["val_rmse"]
    rel = abs(margin) / ref["val_rmse"]
    notes = [f"Validation RMSE: Transformer {mine['val_rmse']:.3f} vs {name} {ref['val_rmse']:.3f} (the best other "
             f"model), a relative difference of {rel:.2%}" + (", i.e. less than 1%." if rel < 0.01 else "."),
             "Only one seed was evaluated, so no statistical superiority is claimed."]
    for key, label, unit in (("val_nasa", "validation NASA score", ""), ("n_parameters", "trainable parameter count", ""),
                             ("latency_ms", "recorded CPU latency per batch", " ms")):
        a, b = mine[key], ref[key]
        word = "lower" if b < a else "higher" if b > a else "equal"
        fmt = (lambda v: f"{v:,}") if key == "n_parameters" else (lambda v: f"{v:,.0f}") if key == "val_nasa" \
            else (lambda v: f"{v:.2f}{unit}")
        notes.append(f"{name} has a {word} {label} ({fmt(b)} vs {fmt(a)} for the Transformer).")
    better = {m: v for m, v in historical_test_rmse.items() if v < transformer_test_rmse}
    if better:
        listed = ", ".join(f"{m} {v:.3f}" + (" (classical)" if m in FAMILY_NOTE else "")
                           for m, v in sorted(better.items(), key=lambda kv: kv[1]))
        notes.append(f"Previously recorded official-test RMSE is lower than the Transformer's {transformer_test_rmse:.3f} "
                     f"for: {listed}. These are observations only; no Session 6 choice used test results.")
    else:
        notes.append("No historical model has a lower previously recorded official-test RMSE.")
    spread = head_count_spread(stage_a_records)
    if spread is not None:
        relation = "larger than" if spread > abs(margin) else "not larger than"
        notes.append(f"Changing only the number of attention heads moved validation RMSE by up to {spread:.3f} in "
                     f"Stage A, {relation} the Transformer-versus-{name} difference of {abs(margin):.3f}.")
    return notes


def early_stopping_text(summary):
    """Explain the min_delta rule using this run's histories."""
    delta = summary["min_delta"]
    lines = [f"The retained checkpoint is the best epoch under the configured minimum-improvement rule: an epoch "
             f"replaces the best one only if its validation loss is lower by more than `min_delta` = {delta:g}. A "
             "numerically lower loss that improves by less than that does not count (the protocol of Sessions 4 and 5)."]
    sel = summary["selected"]
    if sel:
        lines.append(
            f"Selected model: retained epoch {sel['retained_epoch']} is also its absolute validation-loss minimum."
            if sel["gap"] <= 0 else
            f"Selected model: retained epoch {sel['retained_epoch']} (loss {sel['retained_val_loss']:.6f}) differs from "
            f"its numerical minimum at epoch {sel['lowest_epoch']} ({sel['lowest_val_loss']:.6f}).")
    for d in summary["differing"]:
        lines.append(f"`{d['run_name']}`: retained epoch {d['retained_epoch']} (loss {d['retained_val_loss']:.6f}); "
                     f"epoch {d['lowest_epoch']} was lower ({d['lowest_val_loss']:.6f}) by {d['gap']:.1e}.")
    return lines


CAVEATS = [
    "**Single seed.** Every candidate was trained once with seed 42. Small differences between candidates, "
    "poolings or model families are not statistical evidence of superiority.",
    "**Head count does not change the parameter count.** With a fixed d_model, 2 and 4 heads have the same number of "
    "parameters, and so do the two poolings. When such candidates are within 1% of each other the tie is broken by "
    "measured latency, which is noisy; `decided_by` records when a selection reached that level.",
    "**Threshold sensitivity.** The 1% equivalence rule is applied to unrounded validation RMSE; a representative "
    "can depend on the exact threshold.",
    "**Attention over the window.** Attention is bidirectional over the 30 observed cycles (no causal mask). This uses "
    "nothing after the prediction time, but the intermediate representations are not causal.",
    "**Not studied.** Positional encoding (sinusoidal only), normalization (pre-norm only), dropout, feed-forward "
    "ratio, learning rate and sequence length were fixed, so no conclusion about them follows.",
    "**Small dataset.** FD001 training data come from 80 engines; windows overlap heavily and are not independent.",
    "**Timing is environment dependent.** Latency, throughput and training time were measured on one machine (CPU) "
    "and vary between runs and hardware. Training time also depends on the early-stopping epoch.",
    "**Test set.** Test data did not participate in any architecture or pooling choice; only the final selected "
    "Transformer was evaluated on it. Test windows differ in difficulty from validation windows, targets are capped "
    "at 125, and the NASA score is a sum (not comparable between sets of different size).",
    "**Unequal comparison.** Families differ in input representation (classical models use 84 tabular features), "
    "search budget and compute; parameter counts of trees (nodes) and networks are not equivalent.",
    "**Unconstrained output.** Predictions are not clipped to [0, 125].",
    "**CPU repeatability only.** MPS and CUDA kernels are not guaranteed to be bit-reproducible.",
]


def write_report(path, *, cfg, config_ref, seed, device, gate, stage_a_rows, pooling_rows, rule_a, rule_pool,
                 near_misses, selected, final_metrics, efficiency, comparison, comparison_text, timing_notes,
                 reproducibility, session7_rows, figures, unstable, git, early_stopping, interpretation):
    tr, m = cfg["training"], cfg["model"]
    L, F = int(cfg["input"]["sequence_length"]), int(cfg["input"]["n_sensors"])
    arch = selected["architecture"]
    cols = ["candidate", "n_parameters", "val_rmse", "val_mae", "val_r2", "val_nasa", "best_epoch", "epochs_run",
            "train_loss", "val_loss", "train_time_s", "latency_ms", "throughput", "status", "run_id"]
    fmt = {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f", "train_loss": ".5f",
           "val_loss": ".5f", "train_time_s": ".1f", "latency_ms": ".2f", "throughput": ".0f"}
    verification = ("" if not git["git_dirty"] else
                    " **This is a verification run made from an uncommitted working tree (`git_dirty=True`); it is "
                    "not a canonical result.**")
    lines = [
        "# Session 6: compact Transformer for RUL regression", "",
        f"Config `{config_ref}`, seed {seed}, target `{cfg['benchmark']['target_type']}`, device `{device}`, commit "
        f"`{str(git['git_commit'])[:12]}`.{verification}", "",
        "## 1. Objective", "",
        "Build a compact Transformer suited to a small multivariate prognostics dataset and decide which attention "
        "architecture to carry into Session 7.", "",
        "## 2. Architecture and data flow", "",
        "Input projection -> positional encoding -> dropout -> pre-norm Transformer encoder layers "
        "(`torch.nn.TransformerEncoderLayer`, `batch_first=True`; multi-head scaled dot-product self-attention and a "
        f"{m['ff_multiplier']}x feed-forward block with {m['activation']}, each with a residual connection and dropout "
        f"{m['dropout']}) -> final LayerNorm -> pooling -> dropout -> `Linear(d_model, 1)`. Targets are divided by "
        f"{tr['target_scale']} in training and predictions multiplied back, as in Sessions 4 and 5. Integer-valued "
        "settings (width, heads, layers, feed-forward multiplier, sequence length, sensor count) must be true "
        "integers; fractional or boolean values are rejected, never truncated.", "",
        "## 3. Tensor dimensions", "",
        markdown_table([
            {"stage": "input windows", "shape": f"[B, {L}, {F}]"},
            {"stage": "input projection + positions", "shape": f"[B, {L}, {arch['d_model']}]"},
            {"stage": "attention weights per layer", "shape": f"[B, {arch['num_heads']}, {L}, {L}]"},
            {"stage": "encoder output", "shape": f"[B, {L}, {arch['d_model']}]"},
            {"stage": "pooled", "shape": f"[B, {arch['d_model']}]"},
            {"stage": "prediction (RUL, cycles)", "shape": "[B]"}], ["stage", "shape"]), "",
        "## 4. Positional encoding", "",
        "Deterministic sinusoidal encoding over the cycle index 0..29 is added to the projected inputs. It has no "
        "parameters and is what lets self-attention distinguish cycle order (without it, mean pooling would be "
        "permutation invariant).", "",
        "## 5. Attention-mask policy", "",
        "No mask. The prediction is sequence-to-one for the last cycle of a fully observed window: all 30 cycles are "
        "available at inference, so attending across them uses nothing from the future of the prediction time, and "
        "nothing outside the window can be reached. Windows have equal length, so there is no padding to mask. A "
        "mask argument is supported and tested.", "",
        "Pooling: `last` takes the final timestep's contextualised representation (emphasises the current state); "
        "`mean` averages all timesteps (aggregates the window). Through self-attention the last position can already "
        "carry information from the others. Both feed the same regression head.", "",
        "## 6. Architecture grid and protocol", "",
        f"Stage A: d_model {cfg['study']['stage_a']['d_model']} x heads {cfg['study']['stage_a']['num_heads']} x layers "
        f"{cfg['study']['stage_a']['num_layers']}, {cfg['study']['stage_a']['pooling']} pooling (8 candidates). Stage B: "
        f"the leader with the other pooling. Adam lr {tr['learning_rate']}, MSE, batch {tr['batch_size']}, max "
        f"{tr['max_epochs']} epochs, clip {tr['grad_clip']}, ReduceLROnPlateau ({tr['scheduler']['factor']}, patience "
        f"{tr['scheduler']['patience']}), early stopping on validation loss (patience {tr['early_stopping_patience']}, "
        f"min delta {tr['min_delta']}), best weights restored (identical to Session 4, checked at start). Selection: "
        "lowest validation RMSE; within 1% equivalent; then fewer parameters, lower latency, declared order.", "",
        "## 7. Pooling comparison (validation)", "", markdown_table(pooling_rows, cols, fmt), "",
        f"Rule: best {rule_pool['best_validation_value']:.4f}; {rule_pool['n_equivalent']} of "
        f"{rule_pool['n_candidates']} within 1%; decided by {rule_pool['decided_by']}.", "",
        "## 8. Stability gate", "",
        markdown_table([{"check": k.replace("_", " "), **{g["pooling"]: g["checks"][k] for g in gate}}
                        for k in gate[0]["checks"]], ["check", *[g["pooling"] for g in gate]]), "",
        "Tiny-batch training loss: " + "; ".join(
            f"{g['pooling']}: {g['first_loss']:.3f} -> {g['last_loss']:.4f}" for g in gate) + ".", "",
        "## 9. Stage A candidate results (validation)", "", markdown_table(stage_a_rows, cols, fmt), "",
        f"Rule: best {rule_a['best_validation_value']:.4f}; {rule_a['n_equivalent']} of {rule_a['n_candidates']} within "
        f"1%; decided by {rule_a['decided_by']}.", "",
        *[f"Near-miss note ({name}): {text}" for name, text in near_misses.items()], "",
        "## 10. Selected Transformer", "",
        markdown_table([{"setting": k, "value": v} for k, v in {
            **arch, "n_parameters": selected["n_parameters"]}.items()], ["setting", "value"]), "",
        "## 11-12. Validation and final test metrics", "",
        markdown_table(final_metrics, ["eval_set", "n_samples", "rmse", "mae", "r2", "nasa_score"],
                       {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa_score": ".1f"}), "",
        "Test metrics were computed only after the architecture and pooling were frozen.", "",
        "## 13-15. Parameters, training time and latency", "",
        markdown_table([efficiency], list(efficiency), {"train_time_s": ".1f", "latency_ms_per_batch": ".2f",
                                                        "latency_ms_per_sample": ".4f", "throughput_per_s": ".0f"}), "",
        "## 16. Comparison with the historical families (full sensors)", "",
        markdown_table(comparison, ["family", "model", "n_parameters", "val_rmse", "val_mae", "val_r2", "val_nasa",
                                    "latency_ms", "train_time_s"],
                       {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f", "latency_ms": ".2f",
                        "train_time_s": ".1f"}), "",
        comparison_text, "",
        *[f"- {note}" for note in interpretation], "",
        "Timing interpretation (this machine only): the Transformer has " + ("; ".join(timing_notes) or "n/a") + ".", "",
        "Historical rows are the canonical Session 3-5 validation records, read from the results table without "
        "retraining. Parameter counts of the classical models are tree nodes.", "",
        "## 17. Checkpoint and reproducibility", "", reproducibility, "",
        "**Early-stopping semantics.**", "", *[f"- {line}" for line in early_stopping_text(early_stopping)], "",
        "## 18. Scientific caveats", ""] + [f"- {c}" for c in CAVEATS] + [
        "", "## 19. Session 7 compatibility", "",
        "`artifacts/session6/selected_transformer.json` is sufficient to rebuild the model with "
        "`build_from_definition(definition, n_sensors=k)`. Only the input projection changes with k; every other "
        "hyperparameter and the training protocol stay frozen (models are retrained from scratch in Session 7).", "",
        markdown_table(session7_rows, ["n_sensors", "n_parameters", "projection_parameters", "output_shape"]), "",
        "`output_shape` is the prediction shape for a synthetic batch of three samples: `(3,)` means one scalar "
        "prediction for each of the three samples.", "",
        "## 20. Optional sensor-aware variant", "",
        "Not attempted. It is optional in the roadmap; a sensor-token design would be a second architecture family "
        "with its own search, would make the Session 7 comparison less direct (the token count would change with the "
        "sensor budget), and is not needed to answer this session's question.", "",
        "## Unstable or failed runs", "",
        ("None." if not unstable else markdown_table(
            [{"run": r["run_name"], "status": r["status"], "stop_reason": r["stop_reason"]} for r in unstable],
            ["run", "status", "stop_reason"])), "",
        "## Figures", ""] + [f"![{n}](figures/{n})" for n in figures] + [""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
