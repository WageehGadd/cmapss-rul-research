"""Training RUL derivation with a piecewise-linear cap."""


def add_rul(df, rul_cap):
    """Add RUL targets to a TRAINING frame (every engine runs to failure).

    rul_raw = max_cycle(unit) - cycle      (0 at the last cycle)
    rul     = min(rul_raw, rul_cap)        (the capped target used for learning)
    """
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["rul_raw"] = max_cycle - out["cycle"]
    out["rul"] = out["rul_raw"].clip(upper=rul_cap)
    return out
