from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.session import DatabaseSession


@pytest.fixture
def market_data_session(tmp_path: Path) -> DatabaseSession:
    db_file = tmp_path / "test_market_provider.db"
    session = DatabaseSession(str(db_file))

    with session.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_prices (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                timeframe TEXT DEFAULT '1D',
                provider TEXT DEFAULT 'yahoo',
                PRIMARY KEY (symbol, date, timeframe, provider)
            )
            """
        )
        records = [
            ("AAPL", "2026-08-01", 150.0, 155.0, 149.0, 154.0, 1000, "1D", "yahoo"),
            ("AAPL", "2026-08-02", 154.0, 158.0, 153.0, 157.0, 1200, "1D", "yahoo"),
            ("MSFT", "2026-08-01", 300.0, 305.0, 299.0, 304.0, 2000, "1D", "yahoo"),
            ("MSFT", "2026-08-02", 304.0, 308.0, 303.0, 307.0, 2200, "1D", "yahoo"),
        ]
        conn.executemany(
            "INSERT INTO market_prices (symbol, date, open, high, low, close, volume, timeframe, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            records,
        )

    return session


def test_market_data_provider_init_and_clear_cache(
    market_data_session: DatabaseSession,
) -> None:
    provider = MarketDataProvider(market_data_session)
    assert provider.session == market_data_session
    assert provider._in_memory_cache is None
    assert provider._cache_lookback == 0
    assert provider._all_daily_data_cache == {}

    # Populate cache & clear
    provider._cache_lookback = 100
    provider._all_daily_data_cache[100] = {}
    provider.clear_cache()

    assert provider._in_memory_cache is None
    assert provider._cache_lookback == 0
    assert provider._all_daily_data_cache == {}


def test_preload_all_data_and_get_all_daily_data(
    market_data_session: DatabaseSession,
) -> None:
    provider = MarketDataProvider(market_data_session)
    provider.preload_all_data(days=365)

    assert provider._in_memory_cache is not None
    assert "close" in provider._in_memory_cache
    assert "AAPL" in provider._in_memory_cache["close"].columns

    # Test cache hit on get_all_daily_data
    cached = provider.get_all_daily_data(days=30)
    assert cached == provider._in_memory_cache


def test_get_all_daily_data_db_fetch_and_lru_cache(
    market_data_session: DatabaseSession,
) -> None:
    provider = MarketDataProvider(market_data_session)
    res = provider.get_all_daily_data(days=365)
    assert res is not None
    assert 365 in provider._all_daily_data_cache

    # Verify hit from _all_daily_data_cache
    res2 = provider.get_all_daily_data(days=365)
    assert res2 == res


def test_get_universe_daily_data(
    market_data_session: DatabaseSession,
) -> None:
    provider = MarketDataProvider(market_data_session)
    assert provider.get_universe_daily_data([], days=10) is None

    # Fetch without in-memory cache
    data = provider.get_universe_daily_data(["AAPL"], days=365)
    assert data is not None
    assert "AAPL" in data["close"].columns
    assert "MSFT" not in data["close"].columns

    # Preload and test filtered return from in-memory cache
    provider.preload_all_data(days=365)
    data_cached = provider.get_universe_daily_data(["MSFT"], days=10)
    assert data_cached is not None
    assert "MSFT" in data_cached["close"].columns


def test_get_symbol_history(market_data_session: DatabaseSession) -> None:
    provider = MarketDataProvider(market_data_session)

    # From DB
    df_db = provider.get_symbol_history("AAPL", days=365)
    assert not df_db.empty
    assert len(df_db) == 2

    # Preload and get from memory cache
    provider.preload_all_data(days=365)
    df_cache = provider.get_symbol_history("AAPL", days=10)
    assert not df_cache.empty
    assert "close" in df_cache.columns


def test_get_batch_history(market_data_session: DatabaseSession) -> None:
    provider = MarketDataProvider(market_data_session)

    assert provider.get_batch_history([]) == {}

    batch = provider.get_batch_history(
        ["AAPL", "MSFT"], days=365, end_date="2026-08-05"
    )
    assert "AAPL" in batch
    assert "MSFT" in batch
    assert len(batch["AAPL"]) == 2


def test_get_available_dates_and_latest_date(
    market_data_session: DatabaseSession,
) -> None:
    provider = MarketDataProvider(market_data_session)

    dates = provider.get_available_dates("2026-08-01", "2026-08-02")
    assert len(dates) == 2
    assert pd.Timestamp("2026-08-01") in dates

    latest = provider.get_latest_date()
    assert latest == "2026-08-02"
