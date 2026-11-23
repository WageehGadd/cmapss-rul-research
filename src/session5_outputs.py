"""Figures and the report for Session 5. The caveats section is generated, so it survives reruns."""
from pathlib import Path

from .session4_outputs import (file_sha256, freeze_checkpoint, markdown_table,  # noqa: F401  (re-exported)
                               plot_prediction_vs_target, plot_training_curves, write_json)

LABEL_OFFSETS = {"gru": (8, -14, "left"), "gradient_boosting": (-10, -3, "right"), "random_forest": (-8, 12, "right"),
                 "lstm": (12, -16, "left"), "cnn": (10, -14, "left"), "tcn": (-8, 10, "right")}
FAMILY_STYLE = {"classical": ("tab:green", "s"), "recurrent": ("tab:orange", "^"), "convolutional": ("tab:blue", "o")}


def plot_rmse_vs_latency(path, points):
    """Validation RMSE vs inference latency per batch. points: [{label, family, latency, rmse, n_parameters}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for fam, (color, marker) in FAMILY_STYLE.items():
        pts = [p for p in points if p["family"] == fam]
        if pts:
            ax.scatter([p["latency"] for p in pts], [p["rmse"] for p in pts], s=70, c=color, marker=marker, label=fam)
    # Deterministic label placement (display-point offsets per model, independent of the input order), chosen
    # so that the labels of the crowded lower-left cluster do not overlap. Plotted values are not affected.
    for p in points:
        dx, dy, ha = LABEL_OFFSETS.get(p["label"], (10, -14, "left"))
        ax.annotate(f"{p['label']} ({p['n_parameters']:,} par.)", (p["latency"], p["rmse"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7,
                    arrowprops={"arrowstyle": "-", "color": "grey", "lw": 0.5})
    ax.margins(x=0.08, y=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("inference latency per batch of 256 windows (ms, log scale)")
    ax.set_ylabel("validation RMSE (cycles)")
    ax.set_title("Full-sensor models: validation RMSE vs inference latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


CAVEATS = [
    "**Single seed.** Every configuration was trained once, with one seed. Differences between architectures "
    "of a few percent should not be presented as statistical superiority; the study supports selection within "
    "this controlled experiment and does not measure run-to-run variance.",
    "**Receptive field vs the window.** A theoretical receptive field larger than the 30-cycle window does not "
    "create additional observations: the extra positions lie on the zero padding to the left of the window, and "
    "the model never sees more than 30 real cycles.",
    "**Plain CNN receptive field.** The unpadded CNN reads out the last time step, so it sees only the last "
    "1 + depth x (kernel_size - 1) cycles (5 to 13 of the 30). It is a deliberately minimal baseline, not a "
    "model of the whole window.",
    "**Timing is environment dependent.** Latency, throughput and training time were measured on the local "
    "machine (CPU) with the Session 2 protocol and vary between runs and hardware. Training time also depends on "
    "how many epochs early stopping allowed.",
    "**Latency in the tie-break.** Latency is the third tie-break criterion; because it is noisy, a decision "
    "that reaches it is less reproducible than one decided by RMSE or parameter count. The `decided_by` field "
    "records which level decided each selection.",
    "**Test set.** The test set did not participate in any architecture choice. Only the final selected "
    "convolutional architecture was evaluated on it, after the choice was frozen. Test windows differ in difficulty "
    "from validation windows, and targets are capped at 125 on every split.",
    "**Comparison with earlier families.** The classical and recurrent results are the frozen Session 3 and 4 rows, "
    "read unchanged from the results table. The families differ in input representation (84 tabular features vs raw "
    "windows) and in how their hyperparameters were searched, so the comparison is descriptive.",
    "**Unconstrained output.** The regression head is not constrained to [0, 125] and no clipping is applied; all "
    "metrics use the raw predictions.",
    "**Fixed dropout.** A single dropout value is used for all candidates, so dropout is not part of this ablation.",
    "**Sensitivity to the 1% threshold.** The equivalence rule is applied to the unrounded validation RMSE. A "
    "candidate just outside the threshold is not equivalent even if the margin is tiny (see the near-miss notes "
    "in section 6), so a representative can depend on the exact threshold.",
]


def latency_ratio_notes(comparison, final_kind):
    """Sentences comparing the latency of the final convolutional model with the recurrent models,
    computed from the comparison rows of this run (nothing is hard-coded)."""
    lat = {r["model"]: float(r["latency_ms"]) for r in comparison}
    mine = next((v for k, v in lat.items() if k.startswith(f"{final_kind} (")), None)
    notes = []
    for other in ("gru", "lstm"):
        if mine is not None and other in lat and lat[other] > 0:
            ratio = mine / lat[other]
            notes.append(f"{final_kind.upper()} latency is {ratio:.1f}x that of {other.upper()} "
                         f"({mine:.2f} ms vs {lat[other]:.2f} ms per batch of 256)" if ratio >= 1.0 else
                         f"{final_kind.upper()} latency is {1 / ratio:.1f}x lower than that of {other.upper()} "
                         f"({mine:.2f} ms vs {lat[other]:.2f} ms per batch of 256)")
    return notes


def near_miss_sentence(kind, miss, rule):
    """One sentence about the closest candidate just outside the equivalence threshold."""
    if miss is None:
        return (f"{kind.upper()}: no candidate with fewer parameters than the representative lies outside the "
                "threshold, so the 1% rule did not exclude a smaller model.")
    rel = miss["margin"] / miss["threshold"]
    flag = " The selection is therefore sensitive to the exact threshold." if rel < 0.005 else ""
    return (f"{kind.upper()}: threshold {miss['threshold']:.4f} ({rule['tie_tolerance_rel']:.0%} above the best "
            f"{rule['best_validation_value']:.4f}). The closest smaller candidate outside it is {miss['run_name']} "
            f"({miss['n_parameters']:,} parameters, validation RMSE {miss['value']:.4f}), missing it by "
            f"{miss['margin']:.4f} ({rel:.2%}); it was not considered equivalent, and had it fallen inside, the "
            f"smaller model would have won the tie-break.{flag}")


def write_report(path, *, cfg, device, config_ref, seed, gate, tables, family_rule, reps, final, comparison,
                 unstable, reproducibility, figures, receptive_fields, near_miss, latency_notes):
    tr, arch = cfg["training"], cfg["architecture"]
    L = int(cfg["input"]["sequence_length"])
    lines = ["# Session 5: temporal CNN and TCN", "",
             f"Config `{config_ref}`, seed {seed}, target `{cfg['benchmark']['target_type']}`, device `{device}`. "
             "Architectures were selected on validation data only; the test set was evaluated once, for the final "
             "selected convolutional architecture.", "",
             "## 1. Purpose", "",
             "Do convolutional temporal models capture engine degradation as well as recurrent ones, without "
             "recurrent computation? A plain 1D-CNN (minimal baseline) and a residual TCN are compared with each other "
             "and with the frozen classical (Session 3) and recurrent (Session 4) models on the same full-sensor data.", "",
             "## 2. Architectures and tensor shapes", "",
             f"Input `[B, {L}, {cfg['input']['n_sensors']}]` (Session 1 windows, no tabular features) is transposed "
             f"inside the model to `[B, {cfg['input']['n_sensors']}, {L}]` for `Conv1d`; stored arrays are unchanged. "
             "Output is one RUL per window, `[B]`, in cycles (targets are divided by 125 internally, as in Session 4).", "",
             "- **Plain 1D-CNN:** `depth` x (`Conv1d`, stride 1, no padding -> ReLU -> dropout); the time axis shrinks by "
             "`kernel_size - 1` per layer; readout = last time step -> dropout -> `Linear(channels, 1)` (the head "
             "dropout is applied in addition to the dropout after the last convolution).",
             "- **TCN:** one residual block per dilation: causal `Conv1d` -> ReLU -> dropout -> causal `Conv1d` -> ReLU -> "
             "dropout, plus the block input (1x1 projection when the channel count changes), then ReLU. Causal = left "
             "zero padding of `(kernel_size - 1) x dilation`, so the time axis keeps length 30 and no output depends "
             "on later cycles; readout = last time step -> dropout -> `Linear(channels, 1)` (the head dropout is "
             "applied in addition to the dropout inside the last block).", "",
             "## 3. Receptive field", "",
             "TCN: `RF = 1 + 2 x (kernel_size - 1) x sum(dilations)` (two convolutions per block). Plain CNN: "
             "`RF = 1 + depth x (kernel_size - 1)`. Example: kernel 3, dilations [1, 2, 4] -> 1 + 2 x 2 x 7 = 29.", "",
             markdown_table(receptive_fields, ["family", "kernel_size", "depth_or_dilations", "receptive_field",
                                               "vs_window"]), "",
             "An RF above 30 means the extra positions fall on zero padding; the model still sees at most 30 real cycles.", "",
             "## 4. Search space and protocol", "",
             f"CNN: channels {arch['cnn']['channels']} x kernel {arch['cnn']['kernel_size']} x depth {arch['cnn']['depth']} "
             f"(8 candidates). TCN: channels {arch['tcn']['channels']} x kernel {arch['tcn']['kernel_size']} x dilations "
             f"{arch['tcn']['dilations']} (8 candidates). Dropout {arch['dropout']} for all. L={L}, 14 sensors.", "",
             f"Training protocol identical to Session 4 (checked at start): Adam lr {tr['learning_rate']}, MSE on RUL / "
             f"{tr['target_scale']}, batch {tr['batch_size']}, max {tr['max_epochs']} epochs, gradient clipping "
             f"{tr['grad_clip']}, ReduceLROnPlateau (factor {tr['scheduler']['factor']}, patience "
             f"{tr['scheduler']['patience']}), early stopping on validation loss (patience {tr['early_stopping_patience']}, "
             f"min delta {tr['min_delta']}), best validation weights restored. Selection: lowest validation "
             f"{cfg['selection']['metric']}; within {cfg['selection']['tie_tolerance_rel']:.0%} of the best is equivalent; "
             "then fewest parameters, then lower inference latency, then grid order.", "",
             "## 5. Stability gate", "",
             markdown_table(gate, ["family", "first_loss", "last_loss", "relative_drop", "passed"],
                            {"first_loss": ".4f", "last_loss": ".5f", "relative_drop": ".3f"}), "",
             "Each family had to overfit a 64-window batch with finite predictions before the study started.", ""]
    cols = ["channels", "kernel_size", "depth_or_dilations", "receptive_field", "n_parameters", "val_rmse", "val_mae",
            "val_r2", "val_nasa", "best_epoch", "epochs_run", "train_time_s", "latency_ms", "status", "chosen"]
    fmt = {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f", "train_time_s": ".1f",
           "latency_ms": ".2f"}
    for kind in ("cnn", "tcn"):
        rule = family_rule[kind]
        lines += [f"## 6. {kind.upper()} candidates (validation)", "", markdown_table(tables[kind], cols, fmt), "",
                  f"Rule: best validation {rule['metric']} {rule['best_validation_value']:.3f}; {rule['n_equivalent']} "
                  f"candidate(s) within {rule['tie_tolerance_rel']:.0%}; decided by {rule['decided_by']}.", "",
                  "Near-miss note: " + near_miss_sentence(kind, near_miss[kind], rule), ""]
    lines += ["## 7. Representatives and final convolutional architecture", "",
              markdown_table(reps, ["model", "architecture", "receptive_field", "n_parameters", "val_rmse", "val_mae", "val_r2",
                                    "val_nasa", "best_epoch", "epochs_run", "train_time_s", "latency_ms", "throughput"],
                             {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f",
                              "train_time_s": ".1f", "latency_ms": ".2f", "throughput": ".0f"}), "",
              f"Final convolutional architecture (for Session 7): **{final['family'].upper()}** "
              f"({final['architecture']}); decided by {final['rule']['decided_by']} "
              f"({final['rule']['n_equivalent']} equivalent of 2).", "",
              "## 8. Final test result (selected architecture only)", "",
              markdown_table(final["metrics"], ["eval_set", "n_samples", "rmse", "mae", "r2", "nasa_score"],
                             {"rmse": ".3f", "mae": ".3f", "r2": ".4f", "nasa_score": ".1f"}), "",
              "NASA score is a sum, so validation and test totals are not comparable.", "",
              "## 9. Comparison with the frozen classical and recurrent families (full sensors)", "",
              markdown_table(comparison, ["family", "model", "n_parameters", "val_rmse", "val_mae", "val_r2", "val_nasa",
                                          "latency_ms", "train_time_s", "test_rmse"],
                             {"val_rmse": ".3f", "val_mae": ".3f", "val_r2": ".4f", "val_nasa": ".0f",
                              "latency_ms": ".2f", "train_time_s": ".1f"}), "",
              "Validation columns are the development comparison. `test_rmse` is shown only where the earlier session "
              "already evaluated that model (and for the final convolutional model); it was not used for any choice here. "
              "Parameter count for the classical models is the number of tree nodes.", "",
              "Timing interpretation (this run, this machine; not an architecture-independent conclusion): "
              + "; ".join(latency_notes) + ". Latency depends on the hardware, threading and implementation.", ""]
    if figures:
        lines += ["## 10. Figures", ""] + [f"![{n}](figures/{n})" for n in figures] + [""]
    lines += ["## 11. Unstable or failed runs", "",
              ("None: every candidate trained without non-finite loss or gradient values." if not unstable else
               markdown_table([{"run": r["run_name"], "stop_reason": r["stop_reason"]} for r in unstable],
                              ["run", "stop_reason"])), "",
              "## 12. Reproducibility", "", reproducibility, "",
              "## 13. Scientific caveats and interpretation", ""] + [f"- {c}" for c in CAVEATS] + [""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
