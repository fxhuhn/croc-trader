"""Characterization and Pinning Tests for DipBuyer Screener and Trade Manager.

These tests pin the exact baseline behavior of DipBuyerStrategy across both
Screener and Trade Manager before and during refactoring, guaranteeing 100% behavioral
equivalence and zero regressive impact.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import ExitReason, Strategies
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.dip_buyer import (
    CALCULATION_WINDOW_SIZE,
    MIN_DIP_HISTORY_BARS,
    PRICE_DROP_DAYS,
    VOLUME_SMA_WINDOW,
    DipBuyerConfig,
    DipBuyerConfiguration,
)
from app.services.screener.strategies.dip_buyer import (
    DipBuyerStrategy as DipBuyerScreener,
)
from app.services.trade_manager.strategies.dip_buyer import (
    DipBuyerStrategy as DipBuyerTradeManager,
)
from app.types import TradeData

# ==============================================================================
# 1. Attribute & Configuration Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_dip_buyer_default_attributes_and_constants() -> None:
    """Pins class attributes and module-level constants for Dip Buyer."""
    assert DipBuyerScreener.name == str(Strategies.DipBuyer)
    assert MIN_DIP_HISTORY_BARS == 2
    assert VOLUME_SMA_WINDOW == 20
    assert PRICE_DROP_DAYS == 3
    assert CALCULATION_WINDOW_SIZE == 250

    assert DipBuyerTradeManager.name == Strategies.DipBuyer
    assert DipBuyerTradeManager.TIME_STOP_DAYS == 8
    assert DipBuyerTradeManager.MIN_HISTORY_FOR_PREVIOUS_CANDLE == 2
    assert DipBuyerTradeManager.EXIT_TP_FACTOR == 0.8

    # Default configuration values
    config = DipBuyerConfiguration()
    assert config.min_volume == 1_000_000
    assert config.min_price == 5.0
    assert config.atr_window == 5
    assert config.entry_factor == 1.0
    assert config.sma_trend_window == 200
    assert config.min_volatility_ratio == 0.03
    assert config.max_ibs == 0.2
    assert config.max_atr_ratio_3day == -1.0
    assert config.exit_tp_factor == 0.8
    assert config.lookback_days == 600
    assert config.time_stop_days == 8
    assert config.volume_sma_window == 20
    assert config.price_drop_days == 3
    assert config.calculation_window_size == 250

    # Backwards-compatible uppercase property access
    assert config.MIN_VOLUME == 1_000_000
    assert config.MIN_PRICE == 5.0
    assert config.ATR_WINDOW == 5
    assert config.ENTRY_FACTOR == 1.0
    assert config.SMA_TREND_WINDOW == 200
    assert config.MIN_VOLATILITY_RATIO == 0.03
    assert config.MAX_IBS == 0.2
    assert config.MAX_ATR_RATIO_3DAY == -1.0
    assert config.EXIT_TP_FACTOR == 0.8
    assert config.LOOKBACK_DAYS == 600


# ==============================================================================
# 2. Backwards-Compatible Keyword Instantiation
# ==============================================================================


@pytest.mark.tier1
def test_pinning_dip_buyer_config_legacy_uppercase_kwargs() -> None:
    """Pins ability to instantiate config using legacy uppercase kwargs."""
    config = DipBuyerConfig(
        MIN_PRICE=10.0,
        MIN_VOLUME=500_000,
        SMA_TREND_WINDOW=150,
        ATR_WINDOW=3,
        MAX_ATR_RATIO_3DAY=-0.5,
        MAX_IBS=0.25,
        ENTRY_FACTOR=1.2,
        EXIT_TP_FACTOR=0.9,
    )
    assert config.min_price == 10.0
    assert config.min_volume == 500_000
    assert config.sma_trend_window == 150
    assert config.atr_window == 3
    assert config.max_atr_ratio_3day == -0.5
    assert config.max_ibs == 0.25
    assert config.entry_factor == 1.2
    assert config.exit_tp_factor == 0.9

    # Uppercase accessors return matching values
    assert config.MIN_PRICE == 10.0
    assert config.MIN_VOLUME == 500_000
    assert config.SMA_TREND_WINDOW == 150
    assert config.ATR_WINDOW == 3
    assert config.MAX_ATR_RATIO_3DAY == -0.5
    assert config.MAX_IBS == 0.25
    assert config.ENTRY_FACTOR == 1.2
    assert config.EXIT_TP_FACTOR == 0.9


# ==============================================================================
# 3. Screener Dependency Injection Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_dip_buyer_screener_configuration_injection() -> None:
    """Verifies that DipBuyerScreener accepts and exposes DipBuyerConfiguration."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock()
    custom_cfg = DipBuyerConfiguration(
        min_volume=2_000_000,
        min_price=15.0,
        atr_window=7,
    )
    screener = DipBuyerScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        config=custom_cfg,
    )

    assert screener.config.min_volume == 2_000_000
    assert screener.configuration.min_volume == 2_000_000
    assert screener.configuration.min_price == 15.0
    assert screener.configuration.atr_window == 7


# ==============================================================================
# 4. Trade Manager Dependency Injection & Exit Logic Pinning
# ==============================================================================


@pytest.mark.tier1
def test_pinning_dip_buyer_trade_manager_configuration_injection() -> None:
    """Verifies that DipBuyerTradeManager accepts and applies custom configuration."""
    custom_cfg = DipBuyerConfiguration(
        exit_tp_factor=1.2,
        time_stop_days=5,
    )
    manager = DipBuyerTradeManager(configuration=custom_cfg)

    assert manager.configuration.exit_tp_factor == 1.2
    assert manager.configuration.time_stop_days == 5

    # Check entry target calculation: fill_price + (atr * exit_tp_factor)
    trade: TradeData = {
        "id": "T1",
        "symbol": "AAPL",
        "strategy": str(Strategies.DipBuyer),
        "status": "CREATED",
        "entry_price": 100.0,
        "signal_context": '{"date": "2026-03-02", "atr5": 5.0}',
    }
    candle = pd.Series(
        {
            "open": 100.0,
            "low": 99.0,
            "high": 102.0,
            "close": 101.0,
            "date": pd.Timestamp("2026-03-03"),
        }
    )
    df_history = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-03-02")},
            {"date": pd.Timestamp("2026-03-03")},
        ]
    )

    transition = manager.check_entry(trade, candle, df_history)
    assert transition is not None
    # 100.0 + (5.0 * 1.2) = 106.0
    assert transition.updates.get("current_target") == 106.0


@pytest.mark.tier1
def test_pinning_dip_buyer_trade_manager_custom_time_stop() -> None:
    """Verifies custom time_stop_days trigger in active trade management."""
    custom_cfg = DipBuyerConfiguration(time_stop_days=4)
    manager = DipBuyerTradeManager(configuration=custom_cfg)

    trade: TradeData = {
        "id": "T1",
        "symbol": "AAPL",
        "strategy": str(Strategies.DipBuyer),
        "status": "ACTIVE",
        "entry_price": 100.0,
        "entry_date": "2026-03-02",
        "current_target": 120.0,
    }
    # 4 trading days held
    dates = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"])
    df_history = pd.DataFrame(
        {
            "date": dates,
            "high": [105.0, 105.0, 105.0, 105.0],
            "close": [101.0, 102.0, 103.0, 104.0],
        }
    )
    current_candle = pd.Series(
        {
            "open": 103.0,
            "high": 105.0,
            "low": 102.0,
            "close": 104.0,
            "date": pd.Timestamp("2026-03-05"),
        }
    )

    transition = manager._do_manage_active_trade(
        trade=trade,
        current_candle=current_candle,
        date_string="2026-03-05",
        dataframe_history=df_history,
    )
    assert transition is not None
    assert transition.reason == ExitReason.TIME_STOP
    assert transition.updates.get("exit_price") == 104.0
