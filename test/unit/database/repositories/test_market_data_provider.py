# filename: test_market_data_provider.py
from unittest.mock import MagicMock

import pandas

from app.database.repositories.market_data_provider import MarketDataProvider


def test_market_data_provider_init() -> None:
    """Verifies that MarketDataProvider initializes its cache attributes correctly."""
    mock_session = MagicMock()
    provider = MarketDataProvider(mock_session)

    assert provider.session == mock_session
    assert provider._in_memory_cache is None
    assert provider._cache_lookback == 0
    assert provider._all_daily_data_cache == {}


def test_market_data_provider_clear_cache() -> None:
    """Verifies that clear_cache correctly resets all caching attributes."""
    mock_session = MagicMock()
    provider = MarketDataProvider(mock_session)

    # Populate caches with dummy data
    dummy_df = pandas.DataFrame()
    provider._all_daily_data_cache = {100: {"open": dummy_df}}
    provider._in_memory_cache = {"open": dummy_df}
    provider._cache_lookback = 100

    # Execute cache clearing
    provider.clear_cache()

    # Assert cache is completely cleared
    assert provider._all_daily_data_cache == {}
    assert provider._in_memory_cache is None
    assert provider._cache_lookback == 0
