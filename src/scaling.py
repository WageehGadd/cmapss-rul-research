"""Feature scaling. The scaler is fitted on training engines only."""
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALER_CLASSES = {"standard": StandardScaler, "minmax": MinMaxScaler}


def fit_scaler(train_df, features, scaler_type):
    """Fit the scaler on the rows of the training engines. Returns (scaler, fit_unit_ids)."""
    scaler = SCALER_CLASSES[scaler_type]()
    scaler.fit(train_df[features].to_numpy(dtype="float64"))
    return scaler, sorted(int(u) for u in train_df["unit_id"].unique())


def apply_scaler(scaler, df, features):
    """Return a copy of df whose selected feature columns are replaced by scaled values."""
    out = df.copy()
    out[features] = scaler.transform(df[features].to_numpy(dtype="float64"))
    return out
