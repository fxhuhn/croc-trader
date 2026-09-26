from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.ndx_momentum import (
    MomentumTradeContext,
    NDXMomentumScreener,
)


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

    context = MomentumTradeContext(
        symbols=symbols,
        momentum_scores=momentum_scores,
        roc_matrices=roc_map,
        analysis_date=analysis_date,
        price_data=price_data,
        regime_indicators=regime,
    )
    count = strategy._create_trades_direct(context)
    assert count == 1
    assert mock_trade_repository.create_trade.call_count == 2


def test_run_not_triggered_returns_zero(strategy):
    """Verifies that run() returns 0 when analysis is not triggered."""
    with patch.object(strategy, "calculate_analysis") as mock_calc:
        mock_calc.return_value = {
            "triggered": False,
            "date": "2026-01-15",
            "is_rebalance_day": False,
        }
        assert strategy.run(analysis_date="2026-01-15") == 0


def test_calculate_analysis_without_date_defaults_today(strategy):
    """Verifies that calculate_analysis handles missing analysis_date."""
    with patch.object(strategy, "_is_last_trading_day", return_value=False):
        result = strategy.calculate_analysis(analysis_date=None)
        assert result["triggered"] is False
        assert "date" in result


def test_calculate_analysis_no_market_data(strategy, mock_market_data_provider):
    """Tests handling when market data batch history returns empty dictionary."""
    analysis_date = "2026-01-30"
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL"]
        mock_market_data_provider.get_batch_history.return_value = {}

        result = strategy.calculate_analysis(analysis_date=analysis_date)
        assert result["triggered"] is False
        assert result["error"] == "No market data"


def test_calculate_analysis_effective_date_not_in_data_unforced(
    strategy, mock_market_data_provider
):
    """Tests handling when effective date does not match target date and force_run is False."""
    analysis_date = "2026-01-30"
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL"]
        earlier_date = "2026-01-20"
        dates = pd.date_range(end=earlier_date, periods=20, freq="B")
        data = {
            "date": dates,
            "close": [100.0] * 20,
            "high": [110.0] * 20,
            "low": [90.0] * 20,
        }
        mock_market_data_provider.get_batch_history.return_value = {
            "AAPL": pd.DataFrame(data),
            "QQQ": pd.DataFrame(data),
        }

        with patch.object(strategy, "_is_last_trading_day", return_value=True):
            result = strategy.calculate_analysis(
                analysis_date=analysis_date, force_run=False
            )
            assert result["triggered"] is False
            assert "not in data" in result["error"]


def test_create_trades_direct_sends_telegram(strategy, mock_trade_repository):
    """Verifies that telegram notification is dispatched if telegram_bot is configured."""
    mock_bot = MagicMock()
    strategy.telegram_bot = mock_bot

    analysis_date = pd.Timestamp("2026-01-30")
    symbols = ["AAPL"]
    momentum_scores = pd.Series([10.0], index=symbols)
    price_data = {"close": pd.DataFrame({"AAPL": [150.0]}, index=[analysis_date])}
    regime = {
        "bull": True,
        "qqq": 200,
        "qqq_sma": 190,
        "breadth_fast": 60,
        "breadth_slow": 50,
    }
    roc_df = pd.DataFrame({"AAPL": [5.0]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    context = MomentumTradeContext(
        symbols=symbols,
        momentum_scores=momentum_scores,
        roc_matrices=roc_map,
        analysis_date=analysis_date,
        price_data=price_data,
        regime_indicators=regime,
    )

    with patch.object(strategy, "_send_telegram_report") as mock_telegram:
        count = strategy._create_trades_direct(context)
        assert count == 1
        mock_telegram.assert_called_once()


def test_create_trades_direct_database_error_raises_runtime_error(
    strategy, mock_trade_repository
):
    """Verifies that a fatal sqlite3 error during trade creation raises RuntimeError."""
    import sqlite3

    analysis_date = pd.Timestamp("2026-01-30")
    symbols = ["AAPL"]
    momentum_scores = pd.Series([10.0], index=symbols)
    price_data = {"close": pd.DataFrame({"AAPL": [150.0]}, index=[analysis_date])}
    regime = {
        "bull": True,
        "qqq": 200,
        "qqq_sma": 190,
        "breadth_fast": 60,
        "breadth_slow": 50,
    }
    roc_df = pd.DataFrame({"AAPL": [5.0]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    mock_trade_repository.create_trade.side_effect = sqlite3.OperationalError(
        "database is locked"
    )

    context = MomentumTradeContext(
        symbols=symbols,
        momentum_scores=momentum_scores,
        roc_matrices=roc_map,
        analysis_date=analysis_date,
        price_data=price_data,
        regime_indicators=regime,
    )

    with pytest.raises(RuntimeError, match="Database unavailable saving trade"):
        strategy._create_trades_direct(context)


def test_create_single_momentum_trade_fallback_price(strategy, mock_trade_repository):
    """Verifies that _create_single_momentum_trade fetches closing price from context when omitted."""
    analysis_date = pd.Timestamp("2026-01-30")
    symbols = ["MSFT"]
    momentum_scores = pd.Series([12.5], index=symbols)
    price_data = {"close": pd.DataFrame({"MSFT": [250.75]}, index=[analysis_date])}
    regime = {
        "bull": True,
        "qqq": 200,
        "qqq_sma": 190,
        "breadth_fast": 60,
        "breadth_slow": 50,
    }
    roc_df = pd.DataFrame({"MSFT": [8.2]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    context = MomentumTradeContext(
        symbols=symbols,
        momentum_scores=momentum_scores,
        roc_matrices=roc_map,
        analysis_date=analysis_date,
        price_data=price_data,
        regime_indicators=regime,
    )

    entry_price = strategy._create_single_momentum_trade(
        "MSFT", context, "2026-01-30", closing_price=None
    )
    assert entry_price == 250.75
    mock_trade_repository.create_trade.assert_called_once()
    call_kwargs = mock_trade_repository.create_trade.call_args.kwargs
    assert call_kwargs["symbol"] == "MSFT"
    assert call_kwargs["entry"] == 250.75
    assert call_kwargs["context"]["roc_1"] == 8.2
    assert call_kwargs["context"]["momentum_score"] == 12.5
    assert call_kwargs["context"]["regime"] == "BULL"


def test_build_trade_context_bear_regime():
    """Verifies that _build_trade_context sets BEAR flags when regime indicators fall below thresholds."""
    from app.services.screener.strategies.ndx_momentum import _build_trade_context

    analysis_date = pd.Timestamp("2026-01-30")
    symbols = ["GOOGL"]
    momentum_scores = pd.Series([-5.5], index=symbols)
    price_data = {"close": pd.DataFrame({"GOOGL": [100.0]}, index=[analysis_date])}
    regime = {
        "bull": False,
        "qqq": 180,
        "qqq_sma": 190,
        "breadth_fast": 40,
        "breadth_slow": 50,
    }
    roc_df = pd.DataFrame({"GOOGL": [-2.0]}, index=[analysis_date])
    roc_map = {21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df}

    context = MomentumTradeContext(
        symbols=symbols,
        momentum_scores=momentum_scores,
        roc_matrices=roc_map,
        analysis_date=analysis_date,
        price_data=price_data,
        regime_indicators=regime,
    )

    trade_ctx = _build_trade_context("GOOGL", context, "2026-01-30")
    assert trade_ctx["qqq_regime"] == "BEAR"
    assert trade_ctx["breadth_regime"] == "BEAR"
    assert trade_ctx["regime"] == "BEAR"
    assert trade_ctx["momentum_score"] == -5.5
    assert trade_ctx["roc_1"] == -2.0
