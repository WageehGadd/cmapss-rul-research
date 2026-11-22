"""Figures, report and frozen-artifact helpers for Session 4."""
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np

from .session3_outputs import file_sha256, markdown_table, write_json  # noqa: F401  (re-exported)


def _rmse_cycles(loss, scale):
    return math.sqrt(loss) * scale if np.isfinite(loss) else float("nan")


def plot_training_curves(path, histories, scale):
    """Train/validation loss per epoch for each selected model. Loss is shown as RMSE in cycles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(5.8 * len(histories), 4), squeeze=False)
    for ax, (label, hist) in zip(axes[0], histories.items()):
        epochs = [e["epoch"] for e in hist]
        ax.plot(epochs, [_rmse_cycles(e["train_loss"], scale) for e in hist], label="training (dropout on)")
        ax.plot(epochs, [_rmse_cycles(e["val_loss"], scale) for e in hist], label="validation")
        for prev, cur in zip(hist, hist[1:]):
            if cur["lr"] < prev["lr"]:
                ax.axvline(cur["epoch"], color="grey", ls=":", lw=1)
        best = min(hist, key=lambda e: e["val_loss"])
        ax.axvline(best["epoch"], color="green", ls="--", lw=1, label=f"best epoch {best['epoch']}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE (cycles) = sqrt(MSE loss) x RUL cap")
        ax.set_title(f"{label}: dotted lines = LR reductions")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction_vs_target(path, items, cap):
    """Validation predictions vs capped RUL targets, one panel per selected model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(items), figsize=(5.4 * len(items), 5), squeeze=False)
    for ax, (label, y, pred, rmse) in zip(axes[0], items):
        ax.scatter(y, pred, s=4, alpha=0.3)
        ax.plot([0, cap], [0, cap], color="k", lw=1, ls="--")
        ax.set_xlabel("capped RUL target (validation)")
        ax.set_ylabel("predicted RUL")
        ax.set_title(f"{label}  (validation RMSE {rmse:.2f})")
        ax.set_xlim(-2, cap + 5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def freeze_checkpoint(source_dir, family, checkpoint_dir):
    """Copy the selected checkpoint to the git-ignored checkpoint directory; return its identity."""
    src = Path(source_dir) / f"{family}.pt"
    dst_dir = Path(checkpoint_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"selected_{family}.pt"
    shutil.copy2(src, dst)
    return dst, file_sha256(dst)
