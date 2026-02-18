# filename: test_croc_setup.py
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from app.services.screener.strategies.croc_setup import CrocSetupStrategy, PriceData
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.services.telegram import TelegramBot
from app.const import Strategies

@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)

@pytest.fixture
def mock_data_provider() -> MagicMock:
    return MagicMock(spec=MarketDataProvider)

@pytest.fixture
def mock_signal_repo() -> MagicMock:
    return MagicMock(spec=SignalRepository)

@pytest.fixture
def mock_telegram_bot() -> MagicMock:
    return MagicMock(spec=TelegramBot)

@pytest.fixture
def strategy(
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
    mock_signal_repo: MagicMock,
    mock_telegram_bot: MagicMock
) -> CrocSetupStrategy:
    with patch("app.services.screener.strategies.croc_setup.settings.get_path") as mock_get_path:
        mock_get_path.return_value = Path("mock_ranking.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="ranking_2026: []")):
                with patch("app.services.screener.strategies.croc_setup.ExchangeSymbol") as mock_ex:
                    instance = mock_ex.return_value
                    instance.sp_500 = []
                    instance.nasdaq_100 = []
                    instance.dow_30 = []
                    instance.russell_1000 = []
                    return CrocSetupStrategy(
                        trade_repository=mock_trade_repo,
                        data_provider=mock_data_provider,
                        signal_repository=mock_signal_repo,
                        telegram_bot=mock_telegram_bot
                    )

def test_price_data_from_row_valid() -> None:
    """Tests PriceData conversion from a valid dictionary."""
    # Arrange
    row = {
        "high": 110.0,
        "low": 90.0,
        "close": 100.0,
        "sma_20": 95.0,
        "sma_200": 80.0
    }
    
    # Act
    price_data = PriceData.from_row(row)
    
    # Assert
    assert price_data is not None
    assert price_data.high == 110.0
    assert price_data.risk_range == 20.0

def test_price_data_from_row_invalid() -> None:
    """Tests PriceData conversion from an invalid dictionary."""
    # Arrange
    row = {"high": "invalid"}
    
    # Act
    price_data = PriceData.from_row(row)
    
    # Assert
    assert price_data is None

def test_load_config_success(strategy: CrocSetupStrategy) -> None:
    """Tests successful loading of ranking configuration."""
    # Arrange
    mock_yaml = "ranking_2026: [{'Signal': 'Test', 'Score': 10}]"
    
    # Act
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_yaml)):
            rules = strategy._load_config()
    
    # Assert
    assert len(rules) == 1
    assert rules[0]["Signal"] == "Test"

def test_load_config_missing_file(strategy: CrocSetupStrategy) -> None:
    """Tests behavior when config file is missing."""
    # Arrange
    strategy.config_path = Path("non_existent.yaml")
    
    # Act
    with patch("pathlib.Path.exists", return_value=False):
        rules = strategy._load_config()
    
    # Assert
    assert rules == []

def test_run_no_signals(strategy: CrocSetupStrategy, mock_signal_repo: MagicMock) -> None:
    """Tests strategy run when no signals are found."""
    # Arrange
    mock_signal_repo.get_signals_by_date.return_value = []
    
    # Act
    count = strategy.run(analysis_date="2026-02-18")
    
    # Assert
    assert count == 0

def test_process_single_signal_success(strategy: CrocSetupStrategy) -> None:
    """Tests full processing of a single signal."""
    # Arrange
    strategy.ranking_rules = [{"Signal": "Croc1", "Score": 5, "Exit": "split"}]
    row = {
        "symbol": "AAPL",
        "signal": "Croc1",
        "data": json.dumps({"high": 105, "low": 95, "close": 100}),
        "date_str": "2026-02-18"
    }
    
    with patch.object(strategy, "_get_indices_string", return_value="SPX"):
        # Act
        result = strategy._process_single_signal(row)
        
        # Assert
        assert result is not None
        assert result["Symbol"] == "AAPL"
        assert result["Score"] == 5.0
        assert result["Entry"] == 105.0

def test_rule_matching_logic(strategy: CrocSetupStrategy) -> None:
    """Tests the detailed rule matching logic including SMA/EMA mapping."""
    # Arrange
    strategy.ranking_rules = [
        {"Signal": "CROC", "Score": 10, "EMA 200": "> 10%", "RSI": "strong"}
    ]
    row = {
        "signal": "CROC",
        "dist_sma_200": 15.0,
        "rsi": 60.0
    }
    
    # Act
    match = strategy._find_best_match(row)
    
    # Assert
    assert match is not None
    assert match["Score"] == 10

def test_check_value_numeric_and_string(strategy: CrocSetupStrategy) -> None:
    """Tests _check_value with numeric conditions and string fallbacks."""
    # Arrange & Act & Assert
    assert strategy._check_value(25.0, "oversold") is True
    assert strategy._check_value(75.0, "overbought") is True
    assert strategy._check_value("Long", "Long") is True
    assert strategy._check_value(None, "anything") is False

def test_create_trade_invalid_exit(strategy: CrocSetupStrategy) -> None:
    """Tests that trade creation fails for unknown exit strategies."""
    # Arrange
    prices = PriceData(high=100, low=90, close=95)
    match = {"Exit": "unknown", "Score": 1}
    row = {"symbol": "AAPL"}
    
    # Act
    with patch.object(strategy, "_get_indices_string", return_value="SPX"):
        result = strategy._create_trade(row, prices, match)
    
    # Assert
    assert result is None

def test_create_trade_split_logic(strategy: CrocSetupStrategy, mock_trade_repo: MagicMock) -> None:
    """Tests correct calculation of targets for split exit strategy."""
    # Arrange
    prices = PriceData(high=110, low=100, close=105) # Risk = 10
    match = {"Exit": "split", "Score": 8}
    row = {"symbol": "TSLA", "date_str": "2026-02-18"}
    
    with patch.object(strategy, "_get_indices_string", return_value="NDX"):
        # Act
        result = strategy._create_trade(row, prices, match)
        
        # Assert
        assert result is not None
        assert result["TP"] == 140.0 # Entry 110 + 3 * Risk 10 = 140
        mock_trade_repo.create_trade.assert_called_once()
        args, kwargs = mock_trade_repo.create_trade.call_args
        assert kwargs["strategy"] == Strategies.SplitTarget

def test_create_trade_hold_logic(strategy: CrocSetupStrategy, mock_trade_repo: MagicMock) -> None:
    """Tests correct mapping for hold exit strategy."""
    # Arrange
    prices = PriceData(high=110, low=100, close=105)
    match = {"Exit": "hold", "Score": 7}
    row = {"symbol": "MSFT"}
    
    with patch.object(strategy, "_get_indices_string", return_value="SPX"):
        # Act
        result = strategy._create_trade(row, prices, match)
        
        # Assert
        assert result is not None
        mock_trade_repo.create_trade.assert_called_once()
        assert mock_trade_repo.create_trade.call_args[1]["strategy"] == Strategies.HoldTarget

def test_get_indices_string(strategy: CrocSetupStrategy) -> None:
    """Tests the index lookup logic."""
    # Arrange
    strategy.exchange_symbols.sp_500 = ["AAPL"]
    strategy.exchange_symbols.nasdaq_100 = ["AAPL", "MSFT"]
    strategy.exchange_symbols.dow_30 = []
    strategy.exchange_symbols.russell_1000 = []
    
    # Act & Assert
    assert "SPX" in strategy._get_indices_string("AAPL")
    assert "NDX" in strategy._get_indices_string("AAPL")
    assert strategy._get_indices_string("INVALID") == "-"
