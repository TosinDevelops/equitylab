from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equitylab.signals.labels import attach_labels, make_labels


def test_make_labels_forward_return_and_binary() -> None:
    close = pd.Series(
        [100.0, 110.0, 121.0, 100.0],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    out = make_labels(close, max_holding_days=2, min_forward_return=0.0)
    assert pd.isna(out["forward_return"].iloc[-1])
    assert pd.isna(out["forward_return"].iloc[-2])
    assert float(out["forward_return"].iloc[0]) == pytest.approx(0.21)
    assert float(out["label"].iloc[0]) == 1.0
    # 121 -> 100 is negative
    assert float(out["forward_return"].iloc[1]) == pytest.approx(100 / 110 - 1.0)
    assert float(out["label"].iloc[1]) == 0.0


def test_attach_labels_on_panel() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    panel = pd.DataFrame(
        {"close": np.linspace(100, 120, 5)},
        index=idx,
    )
    labeled = attach_labels(panel, max_holding_days=2, min_forward_return=0.0)
    assert "label" in labeled.columns
    assert "forward_return" in labeled.columns
    assert pd.isna(labeled["label"].iloc[-1])
    assert float(labeled["label"].iloc[0]) == 1.0
