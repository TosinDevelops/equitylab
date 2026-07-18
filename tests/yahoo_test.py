from __future__ import annotations

import pandas as pd
import pytest

from equitylab.data.loaders.yahoo import _normalize_price_frame, normalize_ticker


def test_normalize_ticker_strips_uppercases_and_replaces_dot() -> None:
    assert normalize_ticker(" brk.b ") == "BRK-B"
    assert normalize_ticker("aapl") == "AAPL"


def test_normalize_price_frame_flattens_and_renames() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )
    raw = pd.DataFrame(
        [
            [10, 11, 9, 10.5, 1000],
            [10.5, 11.5, 10, 11, 1100],
            [11, 12, 10.5, 11.5, 1200],
        ],
        index=index,
        columns=columns,
    )

    out = _normalize_price_frame(raw)

    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert out.index.name == "date"
    assert out["adj_close"].tolist() == out["close"].tolist()


def test_normalize_price_frame_requires_ohlcv() -> None:
    frame = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
    with pytest.raises(ValueError, match="Missing required price columns"):
        _normalize_price_frame(frame)
