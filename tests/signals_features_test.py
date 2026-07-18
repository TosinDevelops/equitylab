from __future__ import annotations

import numpy as np
import pandas as pd

from equitylab.signals.features import FEATURE_COLUMNS, build_feature_frame, build_feature_panel


def _prices(bars: int = 260, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=bars, freq="B")
    close = start + np.linspace(0, 10, bars) + np.sin(np.arange(bars) / 5.0)
    volume = np.full(bars, 1_000_000.0)
    volume[-1] = 2_000_000.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


def test_build_feature_frame_has_expected_columns_and_warmup_nans() -> None:
    frame = build_feature_frame(_prices(), "AAA")
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert frame["ticker"].eq("AAA").all()
    # EWMA200 (min_periods=200) / 52w need 200+ / 252 bars
    assert frame["distance_from_ewma_200"].iloc[:199].isna().all()
    assert frame["drawdown_52w"].iloc[:251].isna().all()
    assert frame["rsi_14"].iloc[:13].isna().all()
    assert frame["atr_14_pct"].iloc[:13].isna().all()
    # MACD hist needs EMA26 + signal9 warm-up
    assert frame["macd_hist"].iloc[:33].isna().all()
    assert frame[FEATURE_COLUMNS].iloc[-1].notna().all()


def test_build_feature_panel_multiindex() -> None:
    panel = build_feature_panel({"AAA": _prices(), "BBB": _prices(start=50.0)})
    assert panel.index.names == ["date", "ticker"]
    assert set(panel.index.get_level_values("ticker")) == {"AAA", "BBB"}
    assert panel.loc[(panel.index.get_level_values("date").max(), "AAA"), "close"] > 0
