"""Load raw C-MAPSS files and give them the canonical schema."""
import numpy as np
import pandas as pd

OP_SETTING_COLUMNS = [f"operational_setting_{i}" for i in (1, 2, 3)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
CANONICAL_COLUMNS = ["unit_id", "cycle"] + OP_SETTING_COLUMNS + SENSOR_COLUMNS


def read_raw_table(path):
    """Read a raw train/test file. The files are space separated with a trailing space on
    every line, so a whitespace regex is used (sep=" " would add an empty column)."""
    return pd.read_csv(path, sep=r"\s+", header=None)


def apply_schema(raw):
    """Name the 26 raw columns canonically and make unit_id / cycle integers."""
    df = raw.copy()
    df.columns = CANONICAL_COLUMNS
    df["unit_id"] = df["unit_id"].astype("int64")
    df["cycle"] = df["cycle"].astype("int64")
    return df


def read_rul_file(path):
    """Read an official RUL_FDxxx.txt file: one value per test engine, in unit_id order."""
    table = pd.read_csv(path, sep=r"\s+", header=None)
    if table.shape[1] != 1:
        raise ValueError(f"{path} should have exactly 1 column, found {table.shape[1]}")
    return table.iloc[:, 0].to_numpy(dtype="int64")
