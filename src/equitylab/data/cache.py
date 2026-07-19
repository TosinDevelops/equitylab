from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from equitylab.data.loaders.yahoo import normalize_ticker

FetchFn = Callable[..., pd.DataFrame]

_PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def default_db_path() -> Path:
    env = os.environ.get("EQUITYLAB_CACHE_PATH")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "equitylab" / "prices.duckdb"


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            adj_close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cache_meta (
            ticker VARCHAR PRIMARY KEY,
            last_accessed TIMESTAMP,
            min_date DATE,
            max_date DATE
        )
        """
    )
    return con


def _as_date(value: str | date | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _coverage(
    con: duckdb.DuckDBPyConnection, ticker: str
) -> tuple[date, date] | None:
    row = con.execute(
        "SELECT min_date, max_date FROM cache_meta WHERE ticker = ?",
        [ticker],
    ).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None
    return row[0], row[1]


def _upsert_frame(con: duckdb.DuckDBPyConnection, ticker: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    fresh = frame.copy()
    if "date" not in fresh.columns:
        fresh.index.name = fresh.index.name or "date"
        fresh = fresh.reset_index()
    if "date" not in fresh.columns:
        raise ValueError("Price frame must have a date index or column")
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.date
    fresh["ticker"] = ticker
    fresh["volume"] = fresh["volume"].astype("int64")
    fresh = fresh[["ticker", "date", *_PRICE_COLUMNS]]
    con.register("_incoming_prices", fresh)
    try:
        con.execute("INSERT OR REPLACE INTO prices SELECT * FROM _incoming_prices")
    finally:
        con.unregister("_incoming_prices")


def _refresh_meta(con: duckdb.DuckDBPyConnection, ticker: str) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute(
        """
        INSERT INTO cache_meta AS m (ticker, last_accessed, min_date, max_date)
        SELECT
            ? AS ticker,
            ? AS last_accessed,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM prices
        WHERE ticker = ?
        ON CONFLICT (ticker) DO UPDATE SET
            last_accessed = excluded.last_accessed,
            min_date = excluded.min_date,
            max_date = excluded.max_date
        """,
        [ticker, now, ticker],
    )


def _fetch_and_upsert(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    start: date,
    end: date,
    fetch: FetchFn,
) -> None:
    # yfinance `end` is exclusive; request one day past the last wanted date.
    if start > end:
        return
    try:
        frame = fetch(
            ticker,
            interval="1d",
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            use_cache=False,
        )
    except ValueError:
        return
    _upsert_frame(con, ticker, frame)


def load_cached_prices(
    ticker: str,
    start: str | date,
    end: str | date,
    *,
    fetch: FetchFn | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """
    Return daily OHLCV for ticker in [start, end], filling DuckDB gaps via fetch.

    Only missing ranges (before cached min, after cached max, or new tickers) hit
    the network. Overlapping days are upserted.
    """
    from equitylab.data.loaders.yahoo import load_data as yahoo_load

    ticker = normalize_ticker(ticker)
    start_d = _as_date(start)
    end_d = _as_date(end)
    if start_d > end_d:
        raise ValueError("start must be on or before end")

    fetch_fn: FetchFn = fetch or yahoo_load
    con = connect(db_path)
    try:
        coverage = _coverage(con, ticker)
        if coverage is None:
            _fetch_and_upsert(con, ticker, start_d, end_d, fetch_fn)
        else:
            cached_min, cached_max = coverage
            if start_d < cached_min:
                _fetch_and_upsert(con, ticker, start_d, cached_min - timedelta(days=1), fetch_fn)
            if end_d > cached_max:
                _fetch_and_upsert(con, ticker, cached_max + timedelta(days=1), end_d, fetch_fn)

        # Recompute meta even if fetch returned nothing (touch last_accessed).
        count_row = con.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
        has_rows = count_row[0] if count_row is not None else 0
        if has_rows:
            _refresh_meta(con, ticker)
        else:
            raise ValueError(f"No data found for {ticker}")

        result = con.execute(
            """
            SELECT date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date
            """,
            [ticker, start_d, end_d],
        ).df()
    finally:
        con.close()

    if result.empty:
        raise ValueError(f"No data found for {ticker}")

    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")


def evict_stale(days: int = 90, *, db_path: Path | None = None) -> int:
    """Delete tickers not accessed within `days`. Returns number of tickers removed."""
    if days < 1:
        raise ValueError("days must be >= 1")
    con = connect(db_path)
    try:
        stale = con.execute(
            f"""
            SELECT ticker FROM cache_meta
            WHERE last_accessed < now() - INTERVAL {int(days)} DAY
            """
        ).fetchall()
        tickers = [row[0] for row in stale]
        if not tickers:
            return 0
        for ticker in tickers:
            con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
            con.execute("DELETE FROM cache_meta WHERE ticker = ?", [ticker])
        return len(tickers)
    finally:
        con.close()
