import pandas as pd
import pytest

from app.database.repositories.market import MarketRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.session import DatabaseSession
from app.models import MarketPrice


@pytest.fixture
def multi_provider_session(tmp_path):
    db_file = tmp_path / "test_multi_provider.db"
    session = DatabaseSession(str(db_file))

    repo = MarketRepository(session)
    repo.init_schema()

    # Dataset with overlapping and missing dates:
    # AAPL 2026-07-20: Present in both Yahoo ($150 close) and TradingView ($152 close) -> TradingView should win
    # AAPL 2026-07-21: Present ONLY in TradingView ($155 close) -> TradingView fallback
    # MSFT 2026-07-20: Present ONLY in Yahoo ($300 close)
    records = [
        MarketPrice(
            symbol="AAPL",
            date="2026-07-20",
            open=148.0,
            high=151.0,
            low=147.0,
            close=150.0,
            volume=1000,
            provider="yahoo",
        ),
        MarketPrice(
            symbol="AAPL",
            date="2026-07-20",
            open=149.0,
            high=153.0,
            low=148.0,
            close=152.0,
            volume=1100,
            provider="tradingview",
        ),
        MarketPrice(
            symbol="AAPL",
            date="2026-07-21",
            open=153.0,
            high=156.0,
            low=152.0,
            close=155.0,
            volume=1200,
            provider="tradingview",
        ),
        MarketPrice(
            symbol="MSFT",
            date="2026-07-20",
            open=298.0,
            high=302.0,
            low=297.0,
            close=300.0,
            volume=2000,
            provider="yahoo",
        ),
    ]
    repo.save_bulk_prices(records)
    return session


def test_query_dual_provider_priority(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # 2026-07-20 has both Yahoo (150.0) and TradingView (152.0) -> TradingView wins
    ohlcv = repo.get_ohlcv("AAPL", "2026-07-20")
    assert ohlcv is not None
    assert ohlcv["close"] == 152.0
    assert ohlcv["provider"] == "tradingview"


def test_query_tradingview_only_fallback(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # 2026-07-21 has ONLY TradingView (155.0)
    ohlcv = repo.get_ohlcv("AAPL", "2026-07-21")
    assert ohlcv is not None
    assert ohlcv["close"] == 155.0
    assert ohlcv["provider"] == "tradingview"


def test_query_interleaved_date_patching(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # get_symbol_history_raw should prioritize TradingView (152.0) on 2026-07-20 and TradingView (155.0) on 2026-07-21
    df = repo.get_symbol_history_raw("AAPL", "2026-07-20")
    assert len(df) == 2
    assert df["close"].tolist() == [152.0, 155.0]


def test_get_symbol_history_raw_defaults_start_date_when_omitted(
    multi_provider_session,
) -> None:
    """Verifies that get_symbol_history_raw defaults start_date to '2020-01-01' when omitted."""
    repo = MarketRepository(multi_provider_session)

    # Act - call without start_date positional parameter
    df = repo.get_symbol_history_raw("AAPL")

    # Assert
    assert not df.empty
    assert len(df) == 2
    assert df["close"].tolist() == [152.0, 155.0]


def test_market_repository_methods_handle_default_parameters_when_omitted(
    multi_provider_session,
) -> None:
    """Verifies that all MarketRepository queries with optional date parameters execute cleanly without positional arg errors."""
    repo = MarketRepository(multi_provider_session)

    # Act & Assert
    missing_symbols = repo.get_symbols_with_missing_history()
    assert isinstance(missing_symbols, list)

    trading_days = repo.get_trading_days_count("AAPL")
    assert trading_days >= 0

    lookback_df = repo.get_data_for_lookback()
    assert isinstance(lookback_df, pd.DataFrame)

    batch_df = repo.get_batch_history_raw(["AAPL", "MSFT"])
    assert isinstance(batch_df, pd.DataFrame)


def test_market_data_provider_pivoting_with_fallback(multi_provider_session):
    provider = MarketDataProvider(multi_provider_session)
    data = provider.get_universe_daily_data(["AAPL", "MSFT"], days=30000)

    assert data is not None
    close_df = data["close"]

    # AAPL 2026-07-20 should be 152.0 (TradingView), 2026-07-21 should be 155.0 (TradingView)
    assert "AAPL" in close_df.columns
    aapl_values = close_df["AAPL"].dropna().tolist()
    assert aapl_values == [152.0, 155.0]


def test_delete_prices_purges_specific_provider_and_dates(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # Initial state: 2026-07-20 has both Yahoo and TradingView records
    deleted_count = repo.delete_prices("AAPL", ["2026-07-20"], provider="yahoo")
    assert deleted_count == 1

    # Empty date list returns 0 safely
    assert repo.delete_prices("AAPL", [], provider="yahoo") == 0

    # Query directly to ensure Yahoo record is gone but TradingView record remains
    with multi_provider_session.connect() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT provider FROM market_prices WHERE symbol = 'AAPL' AND date = '2026-07-20'"
        )
        remaining_providers = [row[0] for row in cursor.fetchall()]
        assert remaining_providers == ["tradingview"]


def test_delete_corrupt_candles_purges_invalid_geometry(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # Insert corrupt candle (high < low, negative prices, etc.)
    corrupt_records = [
        MarketPrice(
            symbol="CORRUPT",
            date="2026-07-22",
            open=100.0,
            high=90.0,
            low=105.0,
            close=95.0,
            volume=500,
            provider="yahoo",
        ),
        MarketPrice(
            symbol="VALID",
            date="2026-07-22",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=500,
            provider="yahoo",
        ),
    ]
    repo.save_bulk_prices(corrupt_records)

    deleted_count = repo.delete_corrupt_candles(
        "2026-07-22", symbols=["CORRUPT"], provider="yahoo"
    )
    assert deleted_count == 1

    # Verify corrupt record was removed and valid remains
    ohlcv_corrupt = repo.get_ohlcv("CORRUPT", "2026-07-22")
    assert ohlcv_corrupt is None

    ohlcv_valid = repo.get_ohlcv("VALID", "2026-07-22")
    assert ohlcv_valid is not None
