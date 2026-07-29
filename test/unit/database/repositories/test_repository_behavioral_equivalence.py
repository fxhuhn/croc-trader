import pandas
import pytest

from app.database.repositories.market import MarketRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.session import DatabaseSession
from app.models import MarketPrice


@pytest.fixture
def sample_db_session(tmp_path):
    db_file = tmp_path / "test_stocks.db"
    session = DatabaseSession(str(db_file))

    # Populate initial Yahoo data
    repo = MarketRepository(session)
    repo.init_schema()

    records = [
        MarketPrice(
            symbol="AAPL",
            date="2026-07-20",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000,
            provider="yahoo",
        ),
        MarketPrice(
            symbol="AAPL",
            date="2026-07-21",
            open=154.0,
            high=158.0,
            low=153.0,
            close=157.0,
            volume=1200,
            provider="yahoo",
        ),
        MarketPrice(
            symbol="MSFT",
            date="2026-07-20",
            open=300.0,
            high=305.0,
            low=298.0,
            close=304.0,
            volume=2000,
            provider="yahoo",
        ),
    ]
    repo.save_bulk_prices(records)
    return session


def test_market_repository_equivalence_get_symbol_history_raw(sample_db_session):
    repo = MarketRepository(sample_db_session)
    df = repo.get_symbol_history_raw("AAPL", "2026-07-20")

    assert len(df) == 2
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [154.0, 157.0]
    assert pandas.api.types.is_datetime64_any_dtype(df["date"])


def test_market_repository_equivalence_get_batch_history_raw(sample_db_session):
    repo = MarketRepository(sample_db_session)
    df = repo.get_batch_history_raw(["AAPL", "MSFT"], "2026-07-20", "2026-07-21")

    assert len(df) == 3
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}


def test_market_repository_equivalence_get_data_for_lookback(sample_db_session):
    repo = MarketRepository(sample_db_session)
    df = repo.get_data_for_lookback("2026-07-20")

    assert len(df) == 3
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}


def test_market_repository_equivalence_get_latest_price(sample_db_session):
    repo = MarketRepository(sample_db_session)
    price = repo.get_latest_price("AAPL")

    assert price == 157.0


def test_market_data_provider_equivalence_get_symbol_history(sample_db_session):
    provider = MarketDataProvider(sample_db_session)
    df = provider.get_symbol_history("AAPL", days=10)

    assert len(df) == 2
    assert "close" in df.columns
    assert df["close"].tolist() == [154.0, 157.0]


def test_market_data_provider_equivalence_get_universe_daily_data(sample_db_session):
    provider = MarketDataProvider(sample_db_session)
    data = provider.get_universe_daily_data(["AAPL", "MSFT"], days=10)

    assert data is not None
    assert "close" in data
    assert "AAPL" in data["close"].columns
    assert "MSFT" in data["close"].columns
