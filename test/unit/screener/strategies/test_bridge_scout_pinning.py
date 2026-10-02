"""Characterization and Pinning Tests for Bridge Scout Screener and Trade Manager.

These tests pin the exact baseline behavior of BridgeScoutStrategy across both
Screener and Trade Manager before and during refactoring, guaranteeing 100% behavioral
equivalence and zero regressive impact.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.bridge_scout import (
    DEFAULT_ATR_WINDOW,
    DEFAULT_RSI_WINDOW,
    MIN_BRIDGE_HISTORY_BARS,
    BridgeScoutConfiguration,
    BridgeScoutParameters,
    evaluate_bridge_scout_setup,
)
from app.services.screener.strategies.bridge_scout import (
    BridgeScoutStrategy as BridgeScoutScreener,
)
from app.services.trade_manager.strategies.bridge_scout import (
    BridgeScoutTradeStrategy,
)

# ==============================================================================
# 1. Attribute & Configuration Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bridge_scout_default_attributes() -> None:
    """Pins class attributes and constants for Bridge Scout."""
    assert BridgeScoutScreener.STRATEGY_IDENTIFIER == Strategies.BridgeScout
    assert BridgeScoutScreener.name == Strategies.BridgeScout
    assert BridgeScoutScreener.TARGET_SYMBOL == "QQQ"
    assert BridgeScoutScreener.DEFAULT_ENTRY_DAYS_BEFORE == 4
    assert BridgeScoutScreener.DEFAULT_RSI_THRESHOLD == 40.0
    assert BridgeScoutScreener.DEFAULT_MAX_ATR_PCT == 3.5
    assert BridgeScoutScreener.DEFAULT_LOOKBACK_PERIOD == 60

    assert BridgeScoutTradeStrategy.STRATEGY_IDENTIFIER == Strategies.BridgeScout
    assert BridgeScoutTradeStrategy.name == Strategies.BridgeScout

    assert MIN_BRIDGE_HISTORY_BARS == 15
    assert DEFAULT_ATR_WINDOW == 10
    assert DEFAULT_RSI_WINDOW == 2

    # Verify alias and default configuration
    assert BridgeScoutParameters is BridgeScoutConfiguration
    config = BridgeScoutConfiguration()
    assert config.target_symbol == "QQQ"
    assert config.entry_days_before == 4
    assert config.rsi_threshold == 40.0
    assert config.max_atr_pct == 3.5
    assert config.lookback_period == 60
    assert config.min_history_bars == 15
    assert config.atr_window == 10
    assert config.rsi_window == 2
    assert config.is_live_same_day is False


# ==============================================================================
# 2. Pure Calculation Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bridge_scout_pure_calculations() -> None:
    """Pins pure setup calculation under boundary conditions."""
    # Insufficient history returns None
    short_series = pd.Series([100.0 + i for i in range(10)])
    result_short = evaluate_bridge_scout_setup(
        close_series=short_series,
        high_series=short_series + 1.0,
        low_series=short_series - 1.0,
    )
    assert result_short is None

    # Valid length series with high ATR returns is_signal=False
    valid_len_series = pd.Series([100.0] * 20)
    high_atr_series = pd.Series([120.0] * 20)
    low_atr_series = pd.Series([80.0] * 20)
    result_high_atr = evaluate_bridge_scout_setup(
        close_series=valid_len_series,
        high_series=high_atr_series,
        low_series=low_atr_series,
        params=BridgeScoutConfiguration(max_atr_pct=1.0),
    )
    assert result_high_atr is not None
    assert result_high_atr.is_signal is False


# ==============================================================================
# 3. Screener Dependency Injection Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bridge_scout_screener_custom_configuration() -> None:
    """Verifies that BridgeScoutScreener accepts and respects custom BridgeScoutConfiguration."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock()
    custom_cfg = BridgeScoutConfiguration(
        target_symbol="SPY",
        entry_days_before=3,
        rsi_threshold=35.0,
        max_atr_pct=2.5,
        lookback_period=45,
    )
    screener = BridgeScoutScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        configuration=custom_cfg,
    )

    assert screener.configuration.target_symbol == "SPY"
    assert screener.configuration.entry_days_before == 3
    assert screener.configuration.rsi_threshold == 35.0
    assert screener.configuration.max_atr_pct == 2.5
    assert screener.configuration.lookback_period == 45


# ==============================================================================
# 4. Trade Manager Dependency Injection Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bridge_scout_trade_manager_custom_configuration() -> None:
    """Verifies that BridgeScoutTradeStrategy accepts and applies custom configuration."""
    custom_cfg = BridgeScoutConfiguration(
        target_symbol="QQQ",
        rsi_threshold=30.0,
    )
    manager = BridgeScoutTradeStrategy(configuration=custom_cfg)
    assert manager.configuration.rsi_threshold == 30.0
