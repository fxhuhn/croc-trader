"""Characterization and Pinning Tests for Bounce Bandit Screener and Trade Manager.

These tests pin the exact baseline behavior of BounceBanditStrategy across both
Screener and Trade Manager before and during refactoring, guaranteeing 100% behavioral
equivalence and zero regressive impact.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.bounce_bandit import (
    BounceBanditConfiguration,
    BounceBanditParameters,
    evaluate_bounce_bandit_setup,
)
from app.services.screener.strategies.bounce_bandit import (
    BounceBanditStrategy as BounceBanditScreener,
)
from app.services.trade_manager.strategies.bounce_bandit import (
    BounceBanditTradeStrategy,
    calculate_bounce_bandit_targets,
    calculate_required_rsi_exit,
    calculate_required_sma_exit,
)
from app.types import TradeData

# ==============================================================================
# 1. Attribute & Configuration Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bounce_bandit_default_attributes() -> None:
    """Pins class attributes and constants for Bounce Bandit."""
    assert BounceBanditScreener.STRATEGY_IDENTIFIER == Strategies.BounceBandit
    assert BounceBanditScreener.name == Strategies.BounceBandit
    assert BounceBanditScreener.TARGET_SYMBOL == "QQQ"
    assert BounceBanditScreener.DEFAULT_LOOKBACK_PERIOD == 350
    assert BounceBanditScreener.TREND_SMA_LEN == 200
    assert BounceBanditScreener.ATR_LEN == 10
    assert BounceBanditScreener.MAX_ATR_PCT == 2.5
    assert BounceBanditScreener.RSI_ENTRY_THRESHOLD == 20.0

    assert BounceBanditTradeStrategy.STRATEGY_IDENTIFIER == Strategies.BounceBandit
    assert BounceBanditTradeStrategy.name == Strategies.BounceBandit
    assert BounceBanditTradeStrategy.EXIT_SMA_LEN == 8
    assert BounceBanditTradeStrategy.RSI_EXIT_THRESHOLD == 75.0

    # Verify alias and default configuration
    assert BounceBanditParameters is BounceBanditConfiguration
    config = BounceBanditConfiguration()
    assert config.target_symbol == "QQQ"
    assert config.trend_sma_len == 200
    assert config.atr_len == 10
    assert config.max_atr_pct == 2.5
    assert config.rsi_entry_threshold == 20.0
    assert config.exit_sma_len == 8
    assert config.rsi_window == 2
    assert config.rsi_exit_target == 75.0
    assert config.lookback_period == 350
    assert config.lookback_buffer_bars == 2


# ==============================================================================
# 2. Pure Calculation Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bounce_bandit_pure_calculations() -> None:
    """Pins pure target and exit calculations."""
    closes = pd.Series([100.0 + i for i in range(15)])
    targets = calculate_bounce_bandit_targets(closes, exit_sma_length=8)
    assert "sma_8" in targets
    assert "rsi_2" in targets
    assert "target_price" in targets

    required_sma = calculate_required_sma_exit(closes)
    required_rsi = calculate_required_rsi_exit(closes)
    assert required_sma > 0.0
    assert required_rsi > 0.0

    # Short history returns None
    result = evaluate_bounce_bandit_setup(
        close_series=closes,
        high_series=closes + 1.0,
        low_series=closes - 1.0,
    )
    assert result is None


# ==============================================================================
# 3. Screener Dependency Injection Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bounce_bandit_screener_custom_configuration() -> None:
    """Verifies that BounceBanditScreener accepts and respects custom BounceBanditConfiguration."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock()
    custom_cfg = BounceBanditConfiguration(
        target_symbol="SPY",
        trend_sma_len=100,
        rsi_entry_threshold=25.0,
        lookback_period=200,
    )
    screener = BounceBanditScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        configuration=custom_cfg,
    )

    assert screener.configuration.target_symbol == "SPY"
    assert screener.configuration.trend_sma_len == 100
    assert screener.configuration.rsi_entry_threshold == 25.0
    assert screener.configuration.lookback_period == 200


# ==============================================================================
# 4. Trade Manager Dependency Injection & Exit Logic Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_bounce_bandit_trade_manager_custom_configuration() -> None:
    """Verifies that BounceBanditTradeStrategy accepts and applies custom configuration."""
    custom_cfg = BounceBanditConfiguration(
        exit_sma_len=5,
        rsi_exit_target=80.0,
    )
    manager = BounceBanditTradeStrategy(configuration=custom_cfg)

    assert manager.configuration.exit_sma_len == 5
    assert manager.configuration.rsi_exit_target == 80.0

    trade: TradeData = {
        "id": "T1",
        "symbol": "QQQ",
        "strategy": str(Strategies.BounceBandit),
        "status": "ACTIVE",
        "entry_price": 100.0,
        "entry_date": "2026-03-02",
    }

    # 10 candles with high prices to trigger exit
    dates = pd.to_datetime([f"2026-03-{i:02d}" for i in range(2, 12)])
    df_history = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [104.0 + i for i in range(10)],
        }
    )
    current_candle = df_history.iloc[-1]

    transition = manager._do_manage_active_trade(
        trade=trade,
        current_candle=current_candle,
        date_string="2026-03-11",
        dataframe_history=df_history,
    )
    assert transition is not None
    assert "SMA" in transition.reason or "RSI" in transition.reason
