# filename: test_engine_robustness.py
import pytest
import sqlite3
import logging
from unittest.mock import MagicMock
import pandas as pd
from app.services.backtester.engine import BacktestEngine

# --- FIXTURES ---


@pytest.fixture
def mock_dependencies() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Provides mock objects for MarketService, TradeRepository, and Console."""
    market = MagicMock()
    repository = MagicMock()
    console = MagicMock()
    return market, repository, console


# --- TESTS ---


def test_engine_runs_with_empty_market_data(
    mock_dependencies: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Verifies that the engine handles empty market data without crashing."""
    # Arrange
    market, repository, console = mock_dependencies
    market.get_symbol_history.return_value = pd.DataFrame()
    market.get_available_dates.return_value = []

    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repository, console)

    # Act
    engine.run()

    # Assert
    # We expect a CRITICAL message to be printed to the console
    print_calls = [str(call) for call in console.print.call_args_list]
    assert any("CRITICAL" in call or "No market data" in call for call in print_calls)


def test_engine_handles_repository_database_lock(
    mock_dependencies: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Verifies that the engine handles database lock errors during trade retrieval."""
    # Arrange
    market, repository, console = mock_dependencies
    # Setup some market dates so it actually tries to run
    market.get_symbol_history.return_value = pd.DataFrame(
        {"close": [100.0]}, index=pd.to_datetime(["2025-01-01"])
    )
    # Raising error on first call inside run()
    repository.get_by_status.side_effect = sqlite3.OperationalError(
        "database is locked"
    )

    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repository, console)

    # Act & Assert
    with pytest.raises(RuntimeError, match="PortfolioManager: Database unavailable."):
        engine.run()


def test_engine_resilient_to_screener_crash(
    mock_dependencies: tuple[MagicMock, MagicMock, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies that a crash in one screener doesn't stop the entire backtest."""
    # Arrange
    market, repository, console = mock_dependencies

    # Setup 1 day of valid history
    dates = pd.to_datetime(["2025-01-01"])
    history_df = pd.DataFrame({"close": [100.0]}, index=dates)
    history_df.index.name = "date"
    market.get_symbol_history.return_value = history_df
    market.get_available_dates.return_value = ["2025-01-01"]

    engine = BacktestEngine("2025-01-01", "2025-01-01", market, repository, console)

    # Force one screener to crash. Note: engine.screeners is a list of strategy objects.
    engine.screeners[0].run = MagicMock(side_effect=Exception("Screener Exploded"))

    # Act
    with caplog.at_level(logging.ERROR):
        engine.run()

    # Assert
    assert "Screener" in caplog.text
    assert "Screener Exploded" in caplog.text


def test_engine_handles_poisoned_price_data(
    mock_dependencies: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Verifies that the engine handles negative or NaN prices in market data."""
    # Arrange
    market, repository, console = mock_dependencies

    # Poisoned History
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    history_df = pd.DataFrame(
        {
            "open": [100.0, -50.0, float("nan")],
            "high": [110.0, -40.0, 200.0],
            "low": [90.0, -60.0, 50.0],
            "close": [105.0, -55.0, 150.0],
        },
        index=dates,
    )
    history_df.index.name = "date"

    market.get_symbol_history.return_value = history_df
    market.get_available_dates.return_value = [d.strftime("%Y-%m-%d") for d in dates]

    engine = BacktestEngine("2025-01-01", "2025-01-03", market, repository, console)

    # Act
    engine.run()

    # Assert
    assert engine.current_date is not None


def test_engine_setup_handles_missing_spy_history(
    mock_dependencies: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    """Verifies fallback logic when primary benchmark (SPY) is missing."""
    # Arrange
    market, repository, console = mock_dependencies
    market.get_symbol_history.return_value = pd.DataFrame()  # SPY missing
    market.get_available_dates.return_value = ["2025-01-05", "2025-01-06"]

    engine = BacktestEngine("2025-01-01", "2025-01-31", market, repository, console)

    # Act
    engine.setup()

    # Assert
    assert engine.market_dates == ["2025-01-05", "2025-01-06"]
    print_calls = [str(call) for call in console.print.call_args_list]
    assert any("WARNING" in call for call in print_calls)
