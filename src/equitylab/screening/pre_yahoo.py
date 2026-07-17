from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf
from yfinance import EquityQuery

YAHOO_FETCH_SIZE = 150
_YAHOO_PAGE_SIZE = 250
_EXCHANGES = ("NMS", "NYQ")


@dataclass(frozen=True)
class UniverseConfig:
    region: str = "us"
    exchanges: tuple[str, ...] = _EXCHANGES
    min_market_cap: float = 500_000_000
    max_market_cap: float | None = 100_000_000_000
    min_price: float = 5.0
    min_avg_daily_volume: float = 500_000
    drop_symbols: tuple[str, ...] = ("SPY",)


def fetch_universe(config: UniverseConfig) -> list[dict]:
    """Yahoo EquityQuery pre-screen: always fetch YAHOO_FETCH_SIZE candidates."""
    target = YAHOO_FETCH_SIZE
    operands: list[EquityQuery] = [
        EquityQuery("eq", ["region", config.region]),
        EquityQuery("is-in", ["exchange", *config.exchanges]),
        EquityQuery("gte", ["intradaymarketcap", config.min_market_cap]),
        EquityQuery("gte", ["intradayprice", config.min_price]),
        EquityQuery("gte", ["avgdailyvol3m", config.min_avg_daily_volume]),
    ]
    if config.max_market_cap is not None:
        operands.append(EquityQuery("lte", ["intradaymarketcap", config.max_market_cap]))

    query = EquityQuery("and", operands)
    quotes: list[dict] = []
    seen: set[str] = set()
    offset = 0
    drop = {symbol.upper() for symbol in config.drop_symbols}

    while len(quotes) < target:
        page_size = min(_YAHOO_PAGE_SIZE, target - len(quotes))
        response = yf.screen(
            query,
            offset=offset,
            size=page_size,
            sortField="avgdailyvol3m",
            sortAsc=False,
        )
        page = response.get("quotes", []) if isinstance(response, dict) else []
        if not page:
            break

        for quote in page:
            symbol = str(quote.get("symbol") or "").upper()
            if not symbol or symbol in seen or symbol in drop:
                continue
            quote_type = quote.get("quoteType")
            if quote_type is not None and quote_type != "EQUITY":
                continue
            seen.add(symbol)
            quotes.append(quote)
            if len(quotes) >= target:
                break

        offset += len(page)
        if len(page) < page_size:
            break

    return quotes[:target]
