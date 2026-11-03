"""Initial data diagnostics: channel statistics table and report figures.

Channel statistics and correlations use the TRAINING-split engines in raw (unscaled) units.
Pearson correlation is NaN (undefined) for zero-variance channels instead of 0 or an error.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_loading import OP_SETTING_COLUMNS, SENSOR_COLUMNS  # noqa: E402

CHANNELS = OP_SETTING_COLUMNS + SENSOR_COLUMNS
STATUS_COLORS = {"ok": "#1f77b4", "near-constant": "#ff7f0e", "constant": "#7f7f7f"}


def _short(name):
    return name.replace("operational_setting_", "op").replace("sensor_", "s")


def pearson(x, y, tiny=1e-12):
    """Pearson correlation, NaN when either input has (numerically) zero variance."""
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float("nan") if denom <= tiny else float((xc * yc).sum() / denom)


def channel_table(train_df, diag_cfg):
    """One row per channel: mean, std, variance, distinct values, status, corr with capped RUL."""
    rows = []
    rul = train_df["rul"].to_numpy(dtype="float64")
    for ch in CHANNELS:
        values = train_df[ch].to_numpy(dtype="float64")
        std, n_unique = float(values.std()), int(np.unique(values).size)
        if std <= diag_cfg["constant_std_threshold"]:
            status = "constant"
        elif std < diag_cfg["near_constant_std_threshold"] or \
                n_unique <= diag_cfg["near_constant_max_unique"]:
            status = "near-constant"
        else:
            status = "ok"
        if status == "constant":  # drop floating-point noise (e.g. 1e-29) so variance is exactly 0
            std = 0.0
        corr = float("nan") if status == "constant" else pearson(values, rul)
        rows.append({"channel": ch, "mean": float(values.mean()), "std": std,
                     "variance": std ** 2, "n_unique": n_unique, "status": status,
                     "corr_with_capped_rul": corr})
    return pd.DataFrame(rows)


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"Software": None})
    plt.close(fig)
    return path


def plot_lifetime_distribution(lifetimes, train_units, val_units, path, dpi):
    """Histogram of engine lifetimes (cycles to failure), coloured by split."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(lifetimes.min() // 10 * 10, lifetimes.max() + 11, 10)
    ax.hist([lifetimes[lifetimes.index.isin(train_units)], lifetimes[lifetimes.index.isin(val_units)]],
            bins=bins, stacked=True, color=["#1f77b4", "#ff7f0e"], edgecolor="white",
            label=[f"train engines (n={len(train_units)})", f"validation engines (n={len(val_units)})"])
    ax.axvline(lifetimes.mean(), color="k", linestyle="--", label=f"mean = {lifetimes.mean():.1f}")
    ax.axvline(lifetimes.median(), color="k", linestyle=":", label=f"median = {lifetimes.median():.0f}")
    ax.set(title="FD001 engine lifetime distribution (all training engines)",
           xlabel="Engine lifetime (cycles until failure)", ylabel="Number of engines")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_sensor_trajectories(df, units, sensors, path, dpi):
    """Raw sensor value against cycle for a few example engines."""
    ncols = 3
    nrows = int(np.ceil(len(sensors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    for ax, sensor in zip(axes.ravel(), sensors):
        for unit in units:
            g = df[df["unit_id"] == unit]
            ax.plot(g["cycle"], g[sensor], linewidth=1, label=f"engine {unit}")
        ax.set(title=sensor, xlabel="Cycle", ylabel="Sensor value (raw units)")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(sensors):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(units), frameon=False)
    fig.suptitle("FD001 example sensor trajectories over engine life")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, path, dpi)


def plot_sensor_variance(table, path, dpi):
    """Raw variance of every channel (log scale); constant channels are marked with x."""
    positive = table.loc[table["variance"] > 0, "variance"]
    floor = positive.min() / 10
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for status, color in STATUS_COLORS.items():
        sel = (table["status"] == status).to_numpy()
        if status == "constant":
            ax.scatter(x[sel], np.full(sel.sum(), floor), marker="x", color=color,
                       label="constant (variance = 0, drawn at axis floor)")
        elif sel.any():
            ax.bar(x[sel], table.loc[sel, "variance"], color=color, label=status)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor / 2)
    ax.set_xticks(x, [_short(c) for c in table["channel"]], rotation=90)
    ax.set(title="FD001 channel variance on training engines (raw units)",
           xlabel="Channel (opN = operational_setting_N, sN = sensor_N)",
           ylabel="Variance (log scale)")
    ax.set_title(ax.get_title(), pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_rul_correlation(table, rul_cap, path, dpi):
    """Pearson correlation of every channel with capped RUL; undefined for constant channels."""
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for status in ("ok", "near-constant"):
        sel = (table["status"] == status).to_numpy()
        if sel.any():
            ax.bar(x[sel], table.loc[sel, "corr_with_capped_rul"], color=STATUS_COLORS[status],
                   label=status)
    undefined = (table["status"] == "constant").to_numpy()
    ax.scatter(x[undefined], np.zeros(undefined.sum()), marker="x", color=STATUS_COLORS["constant"],
               label="constant: correlation undefined")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_xticks(x, [_short(c) for c in table["channel"]], rotation=90)
    ax.set(title=f"FD001 correlation of channels with capped RUL (cap = {rul_cap})",
           xlabel="Channel (opN = operational_setting_N, sN = sensor_N)",
           ylabel="Pearson correlation with capped RUL")
    ax.set_title(ax.get_title(), pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path, dpi)


def plot_rul_cap_example(df, unit, rul_cap, path, dpi):
    """Raw versus capped RUL for one training engine."""
    g = df[df["unit_id"] == unit]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g["cycle"], g["rul_raw"], label="raw RUL = max_cycle - cycle")
    ax.plot(g["cycle"], g["rul"], linestyle="--", label=f"capped RUL (cap = {rul_cap})")
    ax.set(title=f"Piecewise-linear RUL target, FD001 engine {unit}",
           xlabel="Cycle", ylabel="RUL (cycles)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path, dpi)
