# filename: test_turnover_timing.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import date

from app.services.screener.strategies.turnover_timing import (
    TurnoverTimingStrategy,
    TurnoverConfiguration,
)
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.tools.market_holidays import MarketHolidayChecker

@pytest.fixture
def mock_trade_repository() -> MagicMock:
    """Fixture for mocked TradeRepository."""
    repository = MagicMock(spec=TradeRepository)
    repository.exists.return_value = False
    return repository

@pytest.fixture
def mock_market_data_provider() -> MagicMock:
    """Fixture for mocked MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)

@pytest.fixture
def strategy(
    mock_trade_repository: MagicMock,
    mock_market_data_provider: MagicMock
) -> TurnoverTimingStrategy:
    """Fixture for TurnoverTimingStrategy with mocked dependencies."""
    return TurnoverTimingStrategy(
        trade_repository=mock_trade_repository,
        data_provider=mock_market_data_provider,
        config=TurnoverConfiguration()
    )

@pytest.fixture
def sample_market_data() -> dict[str, pd.DataFrame]:
    """Generates standard market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    data = {}
    for column in ["open", "high", "low", "close"]:
        df = pd.DataFrame(
            np.random.uniform(100, 200, size=(250, 5)),
            index=dates,
            columns=symbols
        )
        data[column] = df
        
    data["volume"] = pd.DataFrame(
        np.random.uniform(1000000, 2000000, size=(250, 5)),
        index=dates,
        columns=symbols
    )
    return data

@pytest.mark.parametrize("test_date, is_friday_holiday, expected_run", [
    ("2026-02-09", False, False), # Monday
    ("2026-02-10", False, False), # Tuesday
    ("2026-02-11", False, False), # Wednesday
    ("2026-02-12", False, False), # Thursday (Normal)
    ("2026-02-13", False, True),  # Friday (Normal)
    ("2026-02-12", True, True),   # Thursday (Friday is Holiday)
])
def test_screener_execution_timing_logic(
    strategy: TurnoverTimingStrategy,
    test_date: str,
    is_friday_holiday: bool,
    expected_run: bool
) -> None:
    """Verifies that the screener only executes on the designated 'End of Week' days."""
    # Arrange
    with patch("app.tools.market_holidays.MarketHolidayChecker.is_holiday") as mock_is_holiday:
        def side_effect(date_obj):
            # 2026-02-13 is a Friday
            if is_friday_holiday and str(date_obj) == "2026-02-13":
                return True
            return False
        mock_is_holiday.side_effect = side_effect
        strategy.data_provider.get_universe_daily_data.return_value = {}
        
        # Act
        result = strategy.run(analysis_date=test_date)
        
        # Assert
        if expected_run:
            strategy.data_provider.get_universe_daily_data.assert_called()
        else:
            assert result == 0
            strategy.data_provider.get_universe_daily_data.assert_not_called()

@patch("app.services.screener.strategies.turnover_timing.ExchangeSymbol")
def test_turnover_strategy_generates_signals_on_valid_setup(
    mock_exchange_symbol: MagicMock,
    strategy: TurnoverTimingStrategy,
    mock_market_data_provider: MagicMock,
    mock_trade_repository: MagicMock,
    sample_market_data: dict[str, pd.DataFrame]
) -> None:
    """Verifies full execution flow and signal generation for Turnover Timing."""
    # Arrange
    setup_date = "2024-05-17"  # Friday
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    
    mock_loader = MagicMock()
    mock_loader.nasdaq_100 = symbols[:2]
    mock_loader.sp_500 = symbols[2:4]
    mock_loader.russell_1000 = [symbols[4]]
    mock_exchange_symbol.return_value = mock_loader
    
    mock_market_data_provider.get_universe_daily_data.return_value = sample_market_data
    
    # Mock Indicators
    mock_sma_df = pd.DataFrame(50.0, index=sample_market_data["close"].index, columns=symbols)
    mock_atr_df = pd.DataFrame(5.0, index=sample_market_data["close"].index, columns=symbols)

    with patch("app.tools.indicators.calculate_sma", return_value=mock_sma_df):
        with patch("app.tools.indicators.calculate_atr", return_value=mock_atr_df):
            # Act
            signal_count = strategy.run(analysis_date=setup_date)
            
            # Assert
            assert signal_count > 0
            assert mock_trade_repository.create_trade.called

def test_extract_safe_float_value_handles_anomalies(strategy: TurnoverTimingStrategy) -> None:
    """Verifies safety helper handles NaN correctly."""
    # Act & Assert
    assert strategy._extract_safe_float_value(np.nan, default=1.0) == 1.0
    assert strategy._extract_safe_float_value(123.45) == 123.45

def test_compile_target_universe_handles_duplicates(strategy: TurnoverTimingStrategy) -> None:
    """Verifies universe compilation removes duplicates across multiple indices."""
    # Arrange
    index_constituents = {
        "IDX1": ["AAPL", "MSFT"],
        "IDX2": ["MSFT", "GOOG"]
    }
    
    # Act
    universe = strategy._compile_target_universe(index_constituents, specific_symbols=None)
    
    # Assert
    assert len(universe) == 3
    assert set(universe) == {"AAPL", "MSFT", "GOOG"}
