"""Engine-level (unit_id) train / validation split."""
import numpy as np


def split_units(unit_ids, val_fraction, seed):
    """Deterministically split engine IDs into (train_units, val_units), both sorted lists.

    Whole engines are assigned to one side, so no engine contributes rows or windows to both.
    """
    units = np.array(sorted(set(int(u) for u in unit_ids)))
    n_val = int(round(val_fraction * len(units)))
    if not 1 <= n_val < len(units):
        raise ValueError(f"val_fraction={val_fraction} leaves an empty train or validation set "
                         f"for {len(units)} engines")
    order = np.random.default_rng(seed).permutation(len(units))
    val_units = np.sort(units[order[:n_val]])
    train_units = np.sort(units[order[n_val:]])
    return train_units.tolist(), val_units.tolist()


def select_units(df, units):
    """Rows of df that belong to the given engines (original row order is kept)."""
    return df[df["unit_id"].isin(units)].copy()
