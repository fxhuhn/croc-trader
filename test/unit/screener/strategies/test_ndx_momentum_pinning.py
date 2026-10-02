"""Characterization and Pinning Tests for NDX Momentum Screener and Trade Strategy.

These tests pin the exact baseline behavior of NDX Momentum to ensure that
refactoring NDXMomentumConfiguration and introducing regime_mode has ZERO
regressive impact on current production behavior.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.ndx_momentum import (
    MomentumTradeContext,
    NDXMomentumConfiguration,
    NDXMomentumScreener,
    _build_trade_context,
)
from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
)
from app.types import TradeData, TradeStatus

# ==============================================================================
# 1. Trade Manager Baseline Behavior: QQQ as Sole Gate, Breadth Ignored
# ==============================================================================


@pytest.mark.tier1
def test_pinning_trade_manager_accepts_bull_qqq_even_with_bear_breadth() -> None:
    """Pins existing behavior: TradeManager enters when qqq_regime is BULL, ignoring bear breadth."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.get_by_status.return_value = []
    strategy = NDXMomentumTradeStrategy()

    signal_date = pd.Timestamp("2026-01-30")
    trade: TradeData = {
        "id": "T1",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": (
            '{"qqq_regime": "BULL", "breadth_regime": "BEAR", "regime": "BEAR", "date": "2026-01-30"}'
        ),
    }
    candle = pd.Series(
        {"open": 150.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 149.0, "close": 150.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 150.0, "close": 152.0},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    # Must be accepted into ACTIVE status (ignoring the bear breadth)
    assert result is not None
    assert "FILLED" in result.reason
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.ACTIVE


@pytest.mark.tier1
def test_pinning_trade_manager_rejects_bear_qqq_even_with_bull_breadth() -> None:
    """Pins existing behavior: TradeManager rejects when qqq_regime is BEAR, even with bull breadth."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    strategy = NDXMomentumTradeStrategy()

    signal_date = pd.Timestamp("2026-01-30")
    trade: TradeData = {
        "id": "T2",
        "symbol": "MSFT",
        "strategy": "ndx_momentum",
        "signal_context": (
            '{"qqq_regime": "BEAR", "breadth_regime": "BULL", "regime": "BEAR", "date": "2026-01-30"}'
        ),
    }
    candle = pd.Series(
        {"open": 400.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 395.0, "close": 400.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 400.0, "close": 405.0},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    # Must be rejected because QQQ is BEAR
    assert result is not None
    assert "QQQ Regime: BEAR" in result.reason
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.INVALID


# ==============================================================================
# 2. Screener Signal Context Schema Parity
# ==============================================================================


@pytest.mark.tier1
def test_pinning_screener_build_trade_context_exact_keys() -> None:
    """Pins exact schema of signal_context emitted by _build_trade_context."""
    analysis_date = pd.Timestamp("2026-01-30")
    symbol = "NVDA"
    context = MomentumTradeContext(
        symbols=[symbol],
        momentum_scores=pd.Series([45.67], index=[symbol]),
        roc_matrices={
            21: pd.DataFrame({symbol: [10.2]}, index=[analysis_date]),
            63: pd.DataFrame({symbol: [15.4]}, index=[analysis_date]),
            126: pd.DataFrame({symbol: [20.6]}, index=[analysis_date]),
            252: pd.DataFrame({symbol: [25.8]}, index=[analysis_date]),
        },
        analysis_date=analysis_date,
        price_data={"close": pd.DataFrame({symbol: [120.0]}, index=[analysis_date])},
        regime_indicators={
            "bull": True,
            "qqq": 450.5,
            "qqq_sma": 430.2,
            "breadth_fast": 65.4,
            "breadth_slow": 55.1,
        },
    )

    trade_context = _build_trade_context(symbol, context, "2026-01-30")

    expected_keys = {
        "source",
        "date",
        "roc_1",
        "roc_3",
        "roc_6",
        "roc_12",
        "momentum_score",
        "qqq_regime",
        "breadth_regime",
        "regime",
        "qqq_abs",
        "qqq_sma",
        "breadth_fast",
        "breadth_slow",
    }
    assert set(trade_context.keys()) == expected_keys
    assert trade_context["qqq_regime"] == "BULL"
    assert trade_context["breadth_regime"] == "BULL"
    assert trade_context["regime"] == "BULL"
    assert trade_context["momentum_score"] == 45.67
    assert trade_context["source"] == "screener"


# ==============================================================================
# 3. Screener Default Configuration Baseline
# ==============================================================================


@pytest.mark.tier1
def test_pinning_screener_default_initialization() -> None:
    """Pins that NDXMomentumScreener initializes with correct defaults without arguments."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = NDXMomentumScreener(
        trade_repository=mock_trade_repo,
        market_data_provider=mock_market_provider,
    )

    assert screener.configuration is not None
    assert screener.configuration.maximum_ticker_count == 5
    assert screener.configuration.regime_mode == "qqq_only"
    assert screener.configuration.qqq_trend_sma_window == 200
    assert screener.configuration.breadth_sma_window == 100
    assert screener.configuration.breadth_fast_sma_window == 10
    assert screener.configuration.breadth_slow_sma_window == 50
    assert screener.configuration.roc_windows == (21, 63, 126, 252)
    assert screener.configuration.history_fetch_days == 450
    assert screener.name == "ndx_momentum"


# ==============================================================================
# 4. Configurable Regime Mode Tests (Combined, Breadth Only, None)
# ==============================================================================


@pytest.mark.tier1
def test_trade_manager_combined_regime_rejects_when_breadth_is_bear() -> None:
    """Verifies that combined regime mode rejects setup when breadth is BEAR even if QQQ is BULL."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    config = NDXMomentumConfiguration(regime_mode="combined")
    strategy = NDXMomentumTradeStrategy(configuration=config)

    signal_date = pd.Timestamp("2026-01-30")
    trade: TradeData = {
        "id": "T1",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": (
            '{"qqq_regime": "BULL", "breadth_regime": "BEAR", "regime": "BEAR", "date": "2026-01-30"}'
        ),
    }
    candle = pd.Series(
        {"open": 150.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 149.0, "close": 150.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 150.0, "close": 152.0},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is not None
    assert "Combined Regime: QQQ=BULL, Breadth=BEAR" in result.reason
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.INVALID


@pytest.mark.tier1
def test_trade_manager_combined_regime_accepts_when_both_bull() -> None:
    """Verifies that combined regime mode accepts setup when both QQQ and breadth are BULL."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.get_by_status.return_value = []
    config = NDXMomentumConfiguration(regime_mode="combined")
    strategy = NDXMomentumTradeStrategy(configuration=config)

    signal_date = pd.Timestamp("2026-01-30")
    trade: TradeData = {
        "id": "T2",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": (
            '{"qqq_regime": "BULL", "breadth_regime": "BULL", "regime": "BULL", "date": "2026-01-30"}'
        ),
    }
    candle = pd.Series(
        {"open": 150.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 149.0, "close": 150.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 150.0, "close": 152.0},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is not None
    assert "FILLED" in result.reason
    mock_trade_repo.update_trade.assert_called_once()
    assert mock_trade_repo.update_trade.call_args[0][1]["status"] == TradeStatus.ACTIVE


@pytest.mark.tier1
def test_trade_manager_breadth_only_mode() -> None:
    """Verifies breadth_only mode ignores QQQ status and relies solely on breadth."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.get_by_status.return_value = []
    config = NDXMomentumConfiguration(regime_mode="breadth_only")
    strategy = NDXMomentumTradeStrategy(configuration=config)

    signal_date = pd.Timestamp("2026-01-30")
    # QQQ is BEAR, but Breadth is BULL -> must accept
    trade: TradeData = {
        "id": "T3",
        "symbol": "AAPL",
        "strategy": "ndx_momentum",
        "signal_context": (
            '{"qqq_regime": "BEAR", "breadth_regime": "BULL", "regime": "BEAR", "date": "2026-01-30"}'
        ),
    }
    candle = pd.Series(
        {"open": 150.0, "date": pd.Timestamp("2026-02-02")},
        name=pd.Timestamp("2026-02-02"),
    )
    history = pd.DataFrame(
        [
            {"date": signal_date, "open": 149.0, "close": 150.0},
            {"date": pd.Timestamp("2026-02-02"), "open": 150.0, "close": 152.0},
        ]
    )

    result = strategy.check_entry(trade, candle, history, mock_trade_repo)

    assert result is not None
    assert "FILLED" in result.reason


@pytest.mark.tier1
def test_screener_configuration_parameters_customizable() -> None:
    """Verifies that NDXMomentumConfiguration accepts custom parameter values."""
    config = NDXMomentumConfiguration(
        maximum_ticker_count=3,
        regime_mode="combined",
        qqq_trend_sma_window=150,
        breadth_sma_window=80,
        breadth_fast_sma_window=5,
        breadth_slow_sma_window=30,
        roc_windows=(10, 30),
        history_fetch_days=300,
    )

    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)
    screener = NDXMomentumScreener(
        trade_repository=mock_trade_repo,
        market_data_provider=mock_market_provider,
        configuration=config,
    )

    assert screener.configuration.maximum_ticker_count == 3
    assert screener.configuration.regime_mode == "combined"
    assert screener.configuration.qqq_trend_sma_window == 150
    assert screener.configuration.breadth_sma_window == 80
    assert screener.configuration.breadth_fast_sma_window == 5
    assert screener.configuration.breadth_slow_sma_window == 30
    assert screener.configuration.roc_windows == (10, 30)
    assert screener.configuration.history_fetch_days == 300
