from pathlib import Path

import pandas as pd  # type: ignore[import-untyped] # pandas has no inline stubs
import pytest

from app.database.repositories.market import MarketRepository
from app.database.session import DatabaseSession
from app.models import MarketPrice


@pytest.fixture
def repository_session(
    tmp_path: Path,
) -> tuple[DatabaseSession, MarketRepository]:
    """Fixture providing an isolated SQLite DatabaseSession with initialized schema."""
    database_file = tmp_path / "test_market_coverage.db"
    session = DatabaseSession(str(database_file))
    repository = MarketRepository(session)
    repository.init_schema()
    return session, repository


def test_blacklist_operations(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies adding, fetching, updating, removing, and clearing blacklisted symbols."""
    _, repository = repository_session

    # Initial state
    assert repository.get_ignored_symbols() == set()

    # Add symbols to blacklist
    repository.ignore_symbol("BAD1", "Delisted")
    repository.ignore_symbol("BAD2", "Low Volume")
    assert repository.get_ignored_symbols() == {"BAD1", "BAD2"}

    # Update existing reason
    repository.ignore_symbol("BAD1", "Bankruptcy")
    assert repository.get_ignored_symbols() == {"BAD1", "BAD2"}

    # Remove single symbol
    repository.remove_ignored_symbol("BAD1")
    assert repository.get_ignored_symbols() == {"BAD2"}

    # Clear all ignored symbols
    repository.clear_ignored_symbols()
    assert repository.get_ignored_symbols() == set()


def test_get_all_known_symbols(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies fetching distinct symbols present in market_prices."""
    _, repository = repository_session

    # Empty table
    assert repository.get_all_known_symbols() == []

    # Insert price records
    records = [
        MarketPrice("AAPL", "2026-07-20", 150.0, 155.0, 149.0, 154.0, 1000, "yahoo"),
        MarketPrice("AAPL", "2026-07-21", 154.0, 158.0, 153.0, 157.0, 1200, "yahoo"),
        MarketPrice("MSFT", "2026-07-20", 300.0, 305.0, 298.0, 304.0, 2000, "yahoo"),
    ]
    repository.save_bulk_prices(records)

    known_symbols = repository.get_all_known_symbols()
    assert sorted(known_symbols) == ["AAPL", "MSFT"]


def test_get_outdated_symbols(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies finding symbols whose latest date is older than reference date."""
    _, repository = repository_session

    records = [
        MarketPrice("AAPL", "2026-07-20", 150.0, 155.0, 149.0, 154.0, 1000, "yahoo"),
        MarketPrice("MSFT", "2026-07-10", 300.0, 305.0, 298.0, 304.0, 2000, "yahoo"),
        MarketPrice(
            "NVDA", "2026-07-10", 100.0, 105.0, 98.0, 104.0, 2000, "tradingview"
        ),
        MarketPrice("IGNORED", "2026-07-01", 10.0, 11.0, 9.0, 10.0, 500, "yahoo"),
    ]
    repository.save_bulk_prices(records)
    repository.ignore_symbol("IGNORED", "Test Ignored")

    # Without provider filter (reference date: 2026-07-15)
    outdated_all = repository.get_outdated_symbols("2026-07-15")
    assert sorted(outdated_all) == ["MSFT", "NVDA"]

    # With provider filter
    outdated_yahoo = repository.get_outdated_symbols("2026-07-15", provider="yahoo")
    assert outdated_yahoo == ["MSFT"]

    outdated_tradingview = repository.get_outdated_symbols(
        "2026-07-15", provider="tradingview"
    )
    assert outdated_tradingview == ["NVDA"]


def test_get_symbols_with_missing_history(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies finding symbols whose history starts after cutoff date."""
    _, repository = repository_session

    records = [
        MarketPrice("OLD", "2019-01-01", 10.0, 12.0, 9.0, 11.0, 100, "yahoo"),
        MarketPrice("NEW", "2021-06-01", 20.0, 22.0, 19.0, 21.0, 200, "yahoo"),
        MarketPrice("SKIP", "2022-01-01", 5.0, 6.0, 4.0, 5.0, 50, "yahoo"),
    ]
    repository.save_bulk_prices(records)
    repository.ignore_symbol("SKIP", "Ignore skip")

    # Cutoff 2020-01-01 -> NEW starts after cutoff, SKIP is ignored, OLD starts before
    missing_symbols = repository.get_symbols_with_missing_history(
        cutoff_date="2020-01-01"
    )
    assert missing_symbols == ["NEW"]


def test_get_latest_updated_at_variations(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies latest updated_at parsing for various timestamp formats and empty state."""
    session, repository = repository_session

    # Empty table
    assert repository.get_latest_updated_at() is None

    # Insert raw row with datetime space separated
    with session.connect() as connection:
        connection.execute(
            "INSERT INTO market_prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("AAPL", "2026-07-20", 150.0, "2026-07-20 14:35:12.987654"),
        )
    assert repository.get_latest_updated_at() == "2026-07-20 14:35"

    # Update with ISO 'T' formatted timestamp
    with session.connect() as connection:
        connection.execute(
            "INSERT INTO market_prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("MSFT", "2026-07-21", 300.0, "2026-07-21T18:45:00"),
        )
    assert repository.get_latest_updated_at() == "2026-07-21 18:45"

    # Update with Date-only string (no time part)
    with session.connect() as connection:
        connection.execute("DELETE FROM market_prices")
        connection.execute(
            "INSERT INTO market_prices (symbol, date, close, updated_at) VALUES (?, ?, ?, ?)",
            ("NVDA", "2026-07-22", 120.0, "2026-07-22"),
        )
    assert repository.get_latest_updated_at() == "2026-07-22"


def test_get_latest_price_not_found(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies get_latest_price returns None for unknown symbols."""
    _, repository = repository_session
    assert repository.get_latest_price("UNKNOWN") is None


def test_get_trading_days_count_options(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies trading days count calculation with explicit end_date and datetime strings."""
    _, repository = repository_session

    records = [
        MarketPrice("AAPL", "2026-07-20", 150.0, 155.0, 149.0, 154.0, 1000, "yahoo"),
        MarketPrice("AAPL", "2026-07-21", 154.0, 158.0, 153.0, 157.0, 1200, "yahoo"),
        MarketPrice("AAPL", "2026-07-22", 157.0, 160.0, 156.0, 159.0, 1100, "yahoo"),
    ]
    repository.save_bulk_prices(records)

    # Explicit date range with datetime string inputs
    count = repository.get_trading_days_count(
        "AAPL", start_date="2026-07-20 00:00:00", end_date="2026-07-21 23:59:59"
    )
    assert count == 2

    # Non-existent symbol returns 0
    assert repository.get_trading_days_count("UNKNOWN") == 0


def test_get_ohlcv_not_found(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies get_ohlcv returns None when no matching bar exists."""
    _, repository = repository_session
    assert repository.get_ohlcv("AAPL", "2026-07-20") is None


def test_get_data_for_lookback_empty(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies get_data_for_lookback returns empty DataFrame when no records match."""
    _, repository = repository_session
    dataframe = repository.get_data_for_lookback(start_date="2099-01-01")
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.empty


def test_get_symbol_history_raw_empty(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies get_symbol_history_raw returns empty DataFrame when no records match."""
    _, repository = repository_session
    dataframe = repository.get_symbol_history_raw("UNKNOWN", start_date="2020-01-01")
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.empty


def test_get_batch_history_raw_edge_cases(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies get_batch_history_raw with empty symbols list and with empty result match."""
    _, repository = repository_session

    # Empty symbols list
    empty_df = repository.get_batch_history_raw([])
    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty

    # Symbols provided but no matching rows found (empty result)
    no_match_df = repository.get_batch_history_raw(
        ["NON_EXISTENT"], start_date="2020-01-01"
    )
    assert isinstance(no_match_df, pd.DataFrame)
    assert no_match_df.empty


def test_save_bulk_prices_edge_cases(
    repository_session: tuple[DatabaseSession, MarketRepository],
) -> None:
    """Verifies saving empty list and saving raw tuples without to_db_row."""
    _, repository = repository_session

    # Empty list should return cleanly
    repository.save_bulk_prices([])
    assert repository.get_all_known_symbols() == []

    # List of raw tuples
    raw_tuples = [
        ("TSLA", "2026-07-20", 250.0, 255.0, 248.0, 252.0, 5000, "yahoo", "1D"),
        ("TSLA", "2026-07-21", 252.0, 260.0, 251.0, 258.0, 6000, "yahoo", "1D"),
    ]
    repository.save_bulk_prices(raw_tuples)

    assert repository.get_all_known_symbols() == ["TSLA"]
    latest_price = repository.get_latest_price("TSLA")
    assert latest_price == 258.0
