from __future__ import annotations

import pandas as pd


def horizon_columns(max_holding_days: int) -> list[str]:
    """Column names for forward returns at each hold horizon 1..N."""
    return [f"forward_return_{k}d" for k in range(1, max_holding_days + 1)]


def make_labels(
    close: pd.Series,
    max_holding_days: int,
    min_forward_return: float = 0.0,
) -> pd.DataFrame:
    """
    Forward close-to-close returns for every horizon from 1..max_holding_days.

    For each day t and hold k:
        forward_return_kd = close[t+k] / close[t] - 1

    Also sets:
        forward_return = max over k of those returns (best exit within the window)
        label = 1 if forward_return > min_forward_return
    """
    if max_holding_days < 1:
        raise ValueError("max_holding_days must be >= 1")

    data: dict[str, pd.Series] = {}
    for k in range(1, max_holding_days + 1):
        data[f"forward_return_{k}d"] = close.shift(-k) / close - 1.0

    horizons = pd.DataFrame(data, index=close.index)
    # Best achievable close-to-close return if exiting on any day 1..N
    forward_return = horizons.max(axis=1, skipna=False)
    label = (forward_return > min_forward_return).astype("float")
    label = label.where(forward_return.notna())

    out = horizons.copy()
    out["forward_return"] = forward_return
    out["label"] = label
    return out


def attach_labels(
    panel: pd.DataFrame,
    max_holding_days: int,
    min_forward_return: float = 0.0,
) -> pd.DataFrame:
    """Attach per-horizon and aggregate forward-return labels to a panel."""
    cols = [*horizon_columns(max_holding_days), "forward_return", "label"]
    if panel.empty:
        out = panel.copy()
        for col in cols:
            out[col] = pd.Series(dtype=float)
        return out

    parts: list[pd.DataFrame] = []
    for ticker, group in panel.groupby(level="ticker", sort=False):
        closes = group["close"].droplevel("ticker")
        labeled = make_labels(closes, max_holding_days, min_forward_return)
        labeled["ticker"] = str(ticker)
        labeled.index.name = "date"
        parts.append(labeled.reset_index().set_index(["date", "ticker"]))

    labels = pd.concat(parts).sort_index()
    out = panel.copy()
    for col in cols:
        out[col] = labels[col]
    return out
