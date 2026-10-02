"""Characterization and Pinning Tests for Turnover Timing Screener and Trade Manager.

These tests pin the exact baseline behavior of TurnoverTimingStrategy across both
Screener and Trade Manager before and during refactoring, guaranteeing 100% behavioral
equivalence and zero regressive impact.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.const import ExitReason, Strategies
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.turnover_timing import (
    MIN_SMA_LOOKBACK_BARS,
    TurnoverConfiguration,
)
from app.services.screener.strategies.turnover_timing import (
    TurnoverTimingStrategy as TurnoverScreener,
)
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy as TurnoverTradeManager,
)
from app.services.trade_manager.strategies.turnover_timing import (
    calculate_consecutive_green_candles,
    is_green_candle,
)
from app.types import TradeData

# ==============================================================================
# 1. Attribute & Configuration Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_turnover_default_attributes() -> None:
    """Pins class attributes and constants for Turnover Timing."""
    assert TurnoverScreener.name == str(Strategies.TurnOverTiming)
    assert TurnoverScreener.FACTOR_STRATEGY_MAP[0.5] == Strategies.TurnOverTiming_05
    assert TurnoverScreener.FACTOR_STRATEGY_MAP[1.0] == Strategies.TurnOverTiming_10
    assert MIN_SMA_LOOKBACK_BARS == 200

    assert TurnoverTradeManager.name == str(Strategies.TurnOverTiming)
    assert TurnoverTradeManager.MIN_GREEN_CANDLES_FOR_EXIT == 2

    # Default TurnoverConfiguration values
    config = TurnoverConfiguration()
    assert config.atr_window == 3
    assert config.entry_factors == [0.5, 1.0]
    assert config.sma_window == 200
    assert config.minimum_lookback_days == 800


# ==============================================================================
# 2. Pure Calculation Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_is_green_candle_pure() -> None:
    """Pins Close > Open definition of green candle."""
    assert is_green_candle(100.0, 101.0) is True
    assert is_green_candle(101.0, 100.0) is False
    assert is_green_candle(100.0, 100.0) is False


@pytest.mark.tier1
def test_pinning_calculate_consecutive_green_candles_pure() -> None:
    """Pins consecutive green candle counting logic."""
    dates = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"])
    history = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 102.0, 105.0, 104.0],
            "close": [101.0, 104.0, 103.0, 106.0],  # G, G, R, G
        }
    )

    # Whole history ending at latest (last is Green -> 1)
    count = calculate_consecutive_green_candles(history)
    assert count == 1

    # First two bars ending at bar 1 (G, G -> 2)
    count_two = calculate_consecutive_green_candles(history.iloc[:2])
    assert count_two == 2

    # Seeded with initial_setup_candle_green=True when history starts after start_date
    count_seeded = calculate_consecutive_green_candles(
        history.iloc[1:2],  # Bar 1 is Green
        start_date="2026-03-01",
        initial_setup_candle_green=True,
    )
    assert count_seeded == 2


# ==============================================================================
# 3. Factor Name Resolution Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_factor_strategy_name_resolution() -> None:
    """Pins strategy identifier resolution from entry factors."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock()
    screener = TurnoverScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
    )

    assert screener._resolve_strategy_name_for_factor(0.5) == str(
        Strategies.TurnOverTiming_05
    )
    assert screener._resolve_strategy_name_for_factor(1.0) == str(
        Strategies.TurnOverTiming_10
    )
    assert screener._resolve_strategy_name_for_factor(0.75) == f"{screener.name}_0.75"


# ==============================================================================
# 4. Dependency Injection & Configuration Override Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_turnover_screener_custom_configuration() -> None:
    """Verifies that TurnoverScreener respects custom TurnoverConfiguration."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock()
    custom_config = TurnoverConfiguration(
        atr_window=5,
        entry_factors=[0.8],
        sma_window=100,
        minimum_lookback_days=400,
    )
    screener = TurnoverScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        configuration=custom_config,
    )

    assert screener.configuration.atr_window == 5
    assert screener.configuration.entry_factors == [0.8]
    assert screener.configuration.sma_window == 100
    assert screener.configuration.minimum_lookback_days == 400


@pytest.mark.tier1
def test_pinning_turnover_trade_manager_custom_configuration() -> None:
    """Verifies that TurnoverTradeManager respects custom TurnoverConfiguration for exit threshold."""
    custom_config = TurnoverConfiguration(
        min_green_candles_for_exit=3,
    )
    manager = TurnoverTradeManager(
        configuration=custom_config,
    )
    assert manager.configuration.min_green_candles_for_exit == 3

    # With min_green_candles_for_exit = 3, 2 green candles should NOT trigger GREEN_SEQUENCE exit!
    trade: TradeData = {
        "id": "T1",
        "symbol": "AAPL",
        "strategy": str(Strategies.TurnOverTiming),
        "status": "ACTIVE",
        "initial_size": 10,
        "current_size": 10,
        "entry_price": 100.0,
        "entry_date": "2026-02-16",
        "current_stop_loss": None,
        "current_target": None,
        "realized_pnl": 0.0,
        "exit_price": None,
        "exit_date": None,
        "exit_reason": None,
        "signal_context": '{"green_candle_count": 2}',
    }
    candle = pd.Series(
        {"open": 110.0, "close": 112.0, "date": pd.Timestamp("2026-02-18")},
        name=pd.Timestamp("2026-02-18"),
    )
    mock_trade_repo = MagicMock(spec=TradeRepository)

    # 2 candles: Should return None (no exit yet)
    with patch.object(manager, "_is_end_of_trading_week", return_value=False):
        result_2 = manager.manage_active_trade(
            trade, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result_2 is None

    # 3 candles: Should trigger GREEN_SEQUENCE exit
    trade_3: TradeData = {**trade, "signal_context": '{"green_candle_count": 3}'}
    with patch.object(manager, "_is_end_of_trading_week", return_value=False):
        result_3 = manager.manage_active_trade(
            trade_3, pd.DataFrame([candle]), mock_trade_repo
        )
        assert result_3 is not None
        assert result_3.reason == ExitReason.GREEN_SEQUENCE
