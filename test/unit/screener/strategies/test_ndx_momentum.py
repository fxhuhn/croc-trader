from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.ndx_momentum import NDXMomentumScreener


@pytest.fixture
def mock_trade_repository():
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_market_data_provider():
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def strategy(mock_trade_repository, mock_market_data_provider):
    return NDXMomentumScreener(
        trade_repository=mock_trade_repository,
        market_data_provider=mock_market_data_provider,
    )


def test_calculate_analysis_last_trading_day_success(
    strategy, mock_market_data_provider
):
    """Verifies that calculate_analysis performs technical analysis correctly on the last trading day."""
    # Arrange
    analysis_date = "2026-01-30"  # Friday, last trading day of Jan 2026
    symbols = ["AAPL", "MSFT", "QQQ"]

    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_exchange:
        mock_inst = MagicMock()
        mock_inst.nasdaq_100 = ["AAPL", "MSFT"]
        mock_exchange.return_value = mock_inst

        # Create mock history for each symbol
        dates = pd.date_range(end=analysis_date, periods=300, freq="B")
        history_map = {}
        for s in symbols:
            df = pd.DataFrame(
                {
                    "date": dates,
                    "close": np.linspace(100, 150, 300)
                    if s != "QQQ"
                    else np.linspace(200, 250, 300),
                    "high": np.linspace(105, 155, 300),
                    "low": np.linspace(95, 145, 300),
                    "volume": 1000000,
                }
            )
            history_map[s] = df

        mock_market_data_provider.get_batch_history.return_value = history_map

        # Act
        result = strategy.calculate_analysis(analysis_date=analysis_date)

        # Assert
        assert result["triggered"] is True
        assert result["date"] == analysis_date
        assert "regime_indicators" in result
        assert "top_symbols" in result
        assert len(result["top_symbols"]) <= 5


def test_calculate_analysis_not_last_trading_day(strategy):
    """Verifies that calculate_analysis returns triggered=False for non-last-trading days."""
    analysis_date = "2026-01-15"
    result = strategy.calculate_analysis(analysis_date=analysis_date)
    assert result["triggered"] is False
    assert result["is_rebalance_day"] is False


def test_calculate_analysis_empty_universe(strategy):
    """Tests handling of an empty universe."""
    analysis_date = "2026-01-30"
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = []
        result = strategy.calculate_analysis(analysis_date=analysis_date)
        assert result["triggered"] is False
        assert "error" in result


def test_calculate_analysis_missing_qqq(strategy, mock_market_data_provider):
    """Tests handling of missing QQQ data."""
    analysis_date = "2026-01-30"
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL"]
        # Must provide high/low as well for pivot
        mock_market_data_provider.get_batch_history.return_value = {
            "AAPL": pd.DataFrame(
                {
                    "date": [pd.Timestamp(analysis_date)],
                    "close": [100.0],
                    "high": [105.0],
                    "low": [95.0],
                }
            )
        }
        result = strategy.calculate_analysis(analysis_date=analysis_date)
        assert result["triggered"] is False
        assert "QQQ" in result["error"]


def test_calculate_analysis_force_run(strategy, mock_market_data_provider):
    """Tests force_run logic on a non-rebalance day."""
    analysis_date = "2026-01-15"
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL"]
        dates = pd.date_range(end=analysis_date, periods=300, freq="B")
        data = {
            "date": dates,
            "close": [100.0] * 300,
            "high": [110.0] * 300,
            "low": [90.0] * 300,
        }
        mock_market_data_provider.get_batch_history.return_value = {
            "AAPL": pd.DataFrame(data),
            "QQQ": pd.DataFrame(data),
        }

        result = strategy.calculate_analysis(
            analysis_date=analysis_date, force_run=True
        )
        assert result["triggered"] is True
        assert result["is_rebalance_day"] is False


def test_run_triggers_trade_creation(strategy, mock_trade_repository):
    """Verifies that run() leads to trade creation if triggered."""
    analysis_date = pd.Timestamp("2026-01-30")
    roc_df = pd.DataFrame({"AAPL": [5.0]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    with patch.object(strategy, "calculate_analysis") as mock_calc:
        mock_calc.return_value = {
            "triggered": True,
            "date": "2026-01-30",
            "top_symbols": ["AAPL"],
            "momentum_scores": pd.Series([10.0], index=["AAPL"]),
            "roc_matrices": roc_map,
            "price_data": {
                "close": pd.DataFrame({"AAPL": [100.0]}, index=[analysis_date])
            },
            "regime_indicators": {
                "bull": True,
                "qqq": 100,
                "qqq_sma": 90,
                "breadth_fast": 60,
                "breadth_slow": 50,
            },
        }

        count = strategy.run(analysis_date="2026-01-30")
        assert count == 1
        mock_trade_repository.create_trade.assert_called_once()


def test_create_trades_direct_error_handling(strategy, mock_trade_repository):
    """Tests that data-level errors in individual trade creation are logged and don't stop the process."""
    analysis_date = pd.Timestamp("2026-01-30")
    symbols = ["AAPL", "MSFT"]
    momentum_scores = pd.Series([10.0, 5.0], index=symbols)
    price_data = {
        "close": pd.DataFrame({"AAPL": [100.0], "MSFT": [50.0]}, index=[analysis_date])
    }
    regime = {
        "bull": True,
        "qqq": 100,
        "qqq_sma": 90,
        "breadth_fast": 60,
        "breadth_slow": 50,
    }

    roc_df = pd.DataFrame({"AAPL": [5.0], "MSFT": [2.0]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    # Mock create_trade to fail with a data-level ValueError for AAPL only
    def side_effect(symbol, **kwargs):
        if symbol == "AAPL":
            raise ValueError("Missing price data for symbol")
        return MagicMock()

    mock_trade_repository.create_trade.side_effect = side_effect

    count = strategy._create_trades_direct(
        symbols, momentum_scores, roc_map, analysis_date, price_data, regime
    )
    assert count == 1
    assert mock_trade_repository.create_trade.call_count == 2
