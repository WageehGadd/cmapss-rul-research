"""Automated sanity checks. Every check returns a short detail string or raises SanityCheckError."""
import numpy as np

from .data_loading import CANONICAL_COLUMNS


class SanityCheckError(Exception):
    """Raised when a Session 1 invariant is violated."""


class CheckLog:
    """Runs checks, records the passing ones, and lets the first failure propagate."""

    def __init__(self):
        self.results = []  # (label, detail)

    def run(self, label, func, *args, **kwargs):
        try:
            detail = func(*args, **kwargs)
        except SanityCheckError as exc:
            raise SanityCheckError(f"[{label}] FAILED: {exc}") from exc
        self.results.append((label, detail))


# ---------------------------------------------------------------- raw data / schema

def check_raw_column_count(raw, name, expected=len(CANONICAL_COLUMNS)):
    if raw.shape[1] != expected:
        raise SanityCheckError(f"{name} has {raw.shape[1]} columns, expected {expected}")
    return f"{name}: {raw.shape[1]} columns"


def check_canonical_columns(df, name):
    if list(df.columns[:len(CANONICAL_COLUMNS)]) != CANONICAL_COLUMNS:
        missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        raise SanityCheckError(f"{name} columns differ from the canonical schema; missing {missing}")
    return f"{name}: all {len(CANONICAL_COLUMNS)} canonical columns present, in order"


def check_no_missing(df, name):
    n_missing = int(df[CANONICAL_COLUMNS].isna().sum().sum())
    if n_missing:
        raise SanityCheckError(f"{name} has {n_missing} missing values in required columns")
    return f"{name}: no missing values"


def check_cycles(df, name):
    """Engines are contiguous blocks; cycles start at 1 and increase by exactly 1."""
    if (df["unit_id"].diff().dropna() < 0).any():
        raise SanityCheckError(f"{name}: rows are not grouped by ascending unit_id")
    for unit, g in df.groupby("unit_id"):
        cycles = g["cycle"].to_numpy()
        if cycles[0] != 1:
            raise SanityCheckError(f"{name}: engine {unit} starts at cycle {cycles[0]}, not 1")
        if not (np.diff(cycles) == 1).all():
            raise SanityCheckError(f"{name}: engine {unit} cycles are not strictly increasing by 1")
    return f"{name}: every engine starts at cycle 1 and cycles increase by 1"


def check_counts(df, name, expected_engines, expected_rows):
    n_engines, n_rows = df["unit_id"].nunique(), len(df)
    if n_engines != expected_engines or n_rows != expected_rows:
        raise SanityCheckError(f"{name}: found {n_engines} engines / {n_rows} rows, expected "
                               f"{expected_engines} / {expected_rows}")
    return f"{name}: {n_engines} engines, {n_rows} rows"


def check_test_rul_file(df_test, official_rul):
    if len(official_rul) != df_test["unit_id"].nunique():
        raise SanityCheckError(f"RUL file has {len(official_rul)} values for "
                               f"{df_test['unit_id'].nunique()} test engines")
    if (official_rul < 0).any():
        raise SanityCheckError("RUL file contains negative values")
    return f"RUL file: {len(official_rul)} values, one per test engine"


# ---------------------------------------------------------------- RUL, split, scaling

def check_rul(df, rul_cap):
    for unit, g in df.groupby("unit_id"):
        raw = g["rul_raw"].to_numpy()
        if raw[-1] != 0 or not (np.diff(raw) == -1).all():
            raise SanityCheckError(f"engine {unit}: raw RUL is not max_cycle - cycle")
    if not (df["rul"] == df["rul_raw"].clip(upper=rul_cap)).all() or df["rul"].max() > rul_cap:
        raise SanityCheckError(f"capped RUL is not min(raw RUL, {rul_cap})")
    return f"raw RUL = max_cycle - cycle (0 at failure); capped RUL = min(raw, {rul_cap})"


def check_split(all_units, train_units, val_units, train_df, val_df):
    overlap = set(train_units) & set(val_units)
    if overlap:
        raise SanityCheckError(f"engines in both train and validation: {sorted(overlap)}")
    if set(train_units) | set(val_units) != set(all_units) or not train_units or not val_units:
        raise SanityCheckError("train + validation engines do not partition the training engines")
    if set(train_df["unit_id"]) != set(train_units) or set(val_df["unit_id"]) != set(val_units):
        raise SanityCheckError("train/validation rows do not match the assigned engines")
    return (f"{len(train_units)} train / {len(val_units)} validation engines, "
            f"0 engines in both, together they cover all {len(all_units)}")


def check_features_variable(train_df, features, std_threshold):
    flat = [f for f in features if train_df[f].std(ddof=0) <= std_threshold]
    if flat:
        raise SanityCheckError(f"selected features are constant on the training engines: {flat}")
    return f"all {len(features)} selected features vary on the training engines"


def check_finite(arr, name):
    if not np.isfinite(arr).all():
        raise SanityCheckError(f"{name} contains NaN or inf")
    return f"{name}: all values finite"


def check_scaler_fit_on_train_only(scaler, fit_unit_ids, train_units, val_units, train_df, features):
    if fit_unit_ids != sorted(train_units) or set(fit_unit_ids) & set(val_units):
        raise SanityCheckError("scaler was fitted on engines other than the training engines")
    n_seen = int(scaler.n_samples_seen_)
    if n_seen != len(train_df):
        raise SanityCheckError(f"scaler saw {n_seen} rows, expected the {len(train_df)} training rows")
    expected = train_df[features].to_numpy(dtype="float64")
    reference = getattr(scaler, "mean_", None)
    reference_expected = expected.mean(axis=0)
    if reference is None:  # minmax scaler
        reference, reference_expected = scaler.data_min_, expected.min(axis=0)
    if not np.allclose(reference, reference_expected, rtol=1e-10, atol=0):
        raise SanityCheckError("scaler statistics differ from the training-engine statistics")
    return (f"scaler fitted on {n_seen} rows of the {len(fit_unit_ids)} training "
            f"engines; statistics match the training rows; no validation/test engine used")


# ---------------------------------------------------------------- windows

def check_window_shapes(ws, name, df, seq_len, n_features, one_per_engine=False):
    lengths = df.groupby("unit_id").size().to_numpy()
    expected_n = len(lengths) if one_per_engine else int((lengths - seq_len + 1).sum())
    expected = (expected_n, seq_len, n_features)
    if ws.X.shape != expected:
        raise SanityCheckError(f"{name}: X shape {ws.X.shape}, expected {expected}")
    for attr in ("y", "y_raw", "unit_ids", "end_cycles"):
        if getattr(ws, attr).shape != (expected_n,):
            raise SanityCheckError(f"{name}: {attr} shape {getattr(ws, attr).shape}, "
                                   f"expected {(expected_n,)}")
    return f"{name}: X {ws.X.shape}, y {ws.y.shape}"


def check_window_engines_disjoint(train_ws, val_ws):
    shared = set(train_ws.unit_ids.tolist()) & set(val_ws.unit_ids.tolist())
    if shared:
        raise SanityCheckError(f"engines contribute windows to both sets: {sorted(shared)}")
    return "train and validation windows come from disjoint engines"


def check_windows_match_source(ws, df, features, seq_len, targets, name):
    """Rebuild every window from the source frame with an independent lookup.

    Confirms that each window is `seq_len` consecutive cycles of one engine (so it cannot
    cross an engine boundary) and that its labels equal targets[(unit, final_cycle)].
    targets maps (unit_id, cycle) -> (rul_raw, rul_capped).
    """
    source = {u: (g["cycle"].to_numpy(), g[features].to_numpy(dtype="float32"))
              for u, g in df.groupby("unit_id")}
    for i in range(len(ws.X)):
        unit, end = int(ws.unit_ids[i]), int(ws.end_cycles[i])
        cycles, feats = source[unit]
        pos = int(np.searchsorted(cycles, end))
        if pos >= len(cycles) or cycles[pos] != end:
            raise SanityCheckError(f"{name}: window {i} ends at cycle {end}, not in engine {unit}")
        start = pos - seq_len + 1
        if start < 0 or not (cycles[start:pos + 1] == np.arange(end - seq_len + 1, end + 1)).all():
            raise SanityCheckError(f"{name}: window {i} crosses the boundary of engine {unit}")
        if not np.array_equal(ws.X[i], feats[start:pos + 1]):
            raise SanityCheckError(f"{name}: window {i} features differ from engine {unit} rows")
        raw, capped = targets[(unit, end)]
        if ws.y_raw[i] != raw or ws.y[i] != capped:
            raise SanityCheckError(f"{name}: window {i} target is not the RUL of its final "
                                   f"timestep (engine {unit}, cycle {end})")
    return (f"{name}: all {len(ws.X)} windows are {seq_len} consecutive cycles of one engine "
            f"and their targets equal the RUL of the final timestep")
