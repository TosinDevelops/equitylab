from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from equitylab.data.cache import evict_stale, load_cached_prices


def _fake_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Inclusive daily bars from start..end (end here is exclusive ISO from fetch)."""
    start_d = pd.Timestamp(start).date() if not isinstance(start, date) else start
    # fetch passes end exclusive ISO string
    end_excl = pd.Timestamp(end).date()
    if start_d >= end_excl:
        raise ValueError(f"No data found for {ticker}")
    idx = pd.date_range(start_d, end_excl - timedelta(days=1), freq="D")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": range(n),
            "high": range(1, n + 1),
            "low": range(n),
            "close": [float(i) for i in range(n)],
            "adj_close": [float(i) for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        },
        index=idx,
    )


def test_load_cached_prices_fetches_once_then_hits_cache(tmp_path) -> None:
    db = tmp_path / "prices.duckdb"
    calls: list[tuple[str, str]] = []

    def fetch(ticker, interval="1d", start=None, end=None, use_cache=False):
        calls.append((start, end))
        return _fake_bars(ticker, start, end)

    out1 = load_cached_prices(
        "aapl",
        date(2024, 1, 1),
        date(2024, 1, 10),
        fetch=fetch,
        db_path=db,
    )
    assert len(out1) == 10
    assert len(calls) == 1

    out2 = load_cached_prices(
        "AAPL",
        date(2024, 1, 1),
        date(2024, 1, 10),
        fetch=fetch,
        db_path=db,
    )
    assert len(out2) == 10
    assert len(calls) == 1  # no second network fetch
    assert out2["close"].tolist() == out1["close"].tolist()


def test_load_cached_prices_extends_forward_only(tmp_path) -> None:
    db = tmp_path / "prices.duckdb"
    calls: list[tuple[str, str]] = []

    def fetch(ticker, interval="1d", start=None, end=None, use_cache=False):
        calls.append((start, end))
        return _fake_bars(ticker, start, end)

    load_cached_prices("MSFT", date(2024, 1, 1), date(2024, 1, 5), fetch=fetch, db_path=db)
    assert len(calls) == 1

    out = load_cached_prices(
        "MSFT", date(2024, 1, 1), date(2024, 1, 10), fetch=fetch, db_path=db
    )
    assert len(calls) == 2
    # Second fetch should start after previously cached max
    assert calls[1][0] == "2024-01-06"
    assert len(out) == 10


def test_load_cached_prices_extends_backward(tmp_path) -> None:
    db = tmp_path / "prices.duckdb"
    calls: list[tuple[str, str]] = []

    def fetch(ticker, interval="1d", start=None, end=None, use_cache=False):
        calls.append((start, end))
        return _fake_bars(ticker, start, end)

    load_cached_prices("XOM", date(2024, 1, 5), date(2024, 1, 10), fetch=fetch, db_path=db)
    out = load_cached_prices(
        "XOM", date(2024, 1, 1), date(2024, 1, 10), fetch=fetch, db_path=db
    )
    assert len(calls) == 2
    assert calls[1][0] == "2024-01-01"
    assert calls[1][1] == "2024-01-05"  # exclusive end for days before cached min
    assert len(out) == 10


def test_evict_stale(tmp_path) -> None:
    import duckdb

    db = tmp_path / "prices.duckdb"

    def fetch(ticker, interval="1d", start=None, end=None, use_cache=False):
        return _fake_bars(ticker, start, end)

    load_cached_prices("AAA", date(2024, 1, 1), date(2024, 1, 3), fetch=fetch, db_path=db)

    con = duckdb.connect(str(db))
    con.execute(
        "UPDATE cache_meta SET last_accessed = now() - INTERVAL 120 DAY WHERE ticker = 'AAA'"
    )
    con.close()

    assert evict_stale(days=90, db_path=db) == 1

    con = duckdb.connect(str(db))
    assert con.execute("SELECT COUNT(*) FROM prices WHERE ticker = 'AAA'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM cache_meta WHERE ticker = 'AAA'").fetchone()[0] == 0
    con.close()
