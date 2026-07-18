from __future__ import annotations

import pandas as pd


def make_labels(
    close: pd.Series,
    max_holding_days: int,
    min_forward_return: float = 0.0,
) -> pd.DataFrame:
    """Forward close-to-close return over max_holding_days and binary label."""
    if max_holding_days < 1:
        raise ValueError("max_holding_days must be >= 1")

    forward_return = close.shift(-max_holding_days) / close - 1.0
    label = (forward_return > min_forward_return).astype("float")
    label = label.where(forward_return.notna())
    return pd.DataFrame(
        {
            "forward_return": forward_return,
            "label": label,
        },
        index=close.index,
    )


def attach_labels(
    panel: pd.DataFrame,
    max_holding_days: int,
    min_forward_return: float = 0.0,
) -> pd.DataFrame:
    """Attach forward-return labels to a (date, ticker) feature panel."""
    if panel.empty:
        out = panel.copy()
        out["forward_return"] = pd.Series(dtype=float)
        out["label"] = pd.Series(dtype=float)
        return out

    parts: list[pd.DataFrame] = []
    for ticker, group in panel.groupby(level="ticker", sort=False):
        closes = group["close"].droplevel("ticker")
        labeled = make_labels(closes, max_holding_days, min_forward_return)
        labeled["ticker"] = ticker
        labeled.index.name = "date"
        parts.append(labeled.reset_index().set_index(["date", "ticker"]))

    labels = pd.concat(parts).sort_index()
    out = panel.copy()
    out["forward_return"] = labels["forward_return"]
    out["label"] = labels["label"]
    return out
