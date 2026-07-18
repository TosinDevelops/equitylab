from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equitylab.signals.labels import attach_labels, horizon_columns, make_labels


def test_make_labels_all_horizons_and_best() -> None:
    # closes: 100, 110, 90, 120
    close = pd.Series(
        [100.0, 110.0, 90.0, 120.0],
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    out = make_labels(close, max_holding_days=2, min_forward_return=0.0)

    assert list(horizon_columns(2)) == ["forward_return_1d", "forward_return_2d"]
    assert float(out.loc[close.index[0], "forward_return_1d"]) == pytest.approx(0.10)
    assert float(out.loc[close.index[0], "forward_return_2d"]) == pytest.approx(-0.10)
    # Best of +10% (day 1) and -10% (day 2)
    assert float(out.loc[close.index[0], "forward_return"]) == pytest.approx(0.10)
    assert float(out.loc[close.index[0], "label"]) == 1.0

    # Last two rows lack full 2-day horizon → NaN aggregate
    assert pd.isna(out["forward_return"].iloc[-1])
    assert pd.isna(out["forward_return"].iloc[-2])


def test_attach_labels_on_panel() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    panel = pd.DataFrame(
        {"close": np.linspace(100, 120, 5)},
        index=idx,
    )
    labeled = attach_labels(panel, max_holding_days=2, min_forward_return=0.0)
    assert "forward_return_1d" in labeled.columns
    assert "forward_return_2d" in labeled.columns
    assert "forward_return" in labeled.columns
    assert pd.isna(labeled["label"].iloc[-1])
    assert float(labeled["label"].iloc[0]) == 1.0
