from __future__ import annotations

from equitylab.screening.pre_yahoo import YAHOO_FETCH_SIZE, UniverseConfig


def test_yahoo_fetch_size_is_150() -> None:
    assert YAHOO_FETCH_SIZE == 150


def test_universe_config_defaults_match_spec() -> None:
    config = UniverseConfig()
    assert config.region == "us"
    assert config.exchanges == ("NMS", "NYQ")
    assert config.min_market_cap == 500_000_000
    assert config.max_market_cap == 100_000_000_000
    assert config.min_price == 5.0
    assert config.min_avg_daily_volume == 500_000
    assert config.drop_symbols == ("SPY",)
