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
    # AAPL 2026-07-20: Present in both Yahoo ($150 close) and TradingView ($152 close) -> Yahoo should win
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

    # 2026-07-20 has both Yahoo (150.0) and TradingView (152.0)
    ohlcv = repo.get_ohlcv("AAPL", "2026-07-20")
    assert ohlcv is not None
    assert ohlcv["close"] == 150.0
    assert ohlcv["provider"] == "yahoo"


def test_query_tradingview_only_fallback(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # 2026-07-21 has ONLY TradingView (155.0)
    ohlcv = repo.get_ohlcv("AAPL", "2026-07-21")
    assert ohlcv is not None
    assert ohlcv["close"] == 155.0
    assert ohlcv["provider"] == "tradingview"


def test_query_interleaved_date_patching(multi_provider_session):
    repo = MarketRepository(multi_provider_session)

    # get_symbol_history_raw should seamlessly patch 2026-07-20 (Yahoo) and 2026-07-21 (TradingView)
    df = repo.get_symbol_history_raw("AAPL", "2026-07-20")
    assert len(df) == 2
    assert df["close"].tolist() == [150.0, 155.0]


def test_market_data_provider_pivoting_with_fallback(multi_provider_session):
    provider = MarketDataProvider(multi_provider_session)
    data = provider.get_universe_daily_data(["AAPL", "MSFT"], days=10)

    assert data is not None
    close_df = data["close"]

    # AAPL 2026-07-20 should be 150.0 (Yahoo), 2026-07-21 should be 155.0 (TradingView)
    assert "AAPL" in close_df.columns
    aapl_values = close_df["AAPL"].dropna().tolist()
    assert aapl_values == [150.0, 155.0]
