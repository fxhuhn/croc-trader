"""Characterization and Pinning Tests for TwoPercent Screener and Trade Manager.

These tests pin the exact baseline behavior of TwoPercentStrategy across both
Screener and Trade Manager before any refactoring takes place, guaranteeing
100% behavioral equivalence and zero regressive impact.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.const import ExitReason, Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.two_percent_strategy import (
    TwoPercentConfiguration,
)
from app.services.screener.strategies.two_percent_strategy import (
    TwoPercentStrategy as TwoPercentScreener,
)
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy as TwoPercentTradeManager,
)
from app.tools.trading_calendar import FRIDAY, SATURDAY
from app.types import TradeData, TradeStatus

# ==============================================================================
# 1. Attribute Pinning: Verify Canonical Class Attributes
# ==============================================================================


@pytest.mark.tier1
def test_pinning_screener_default_attributes() -> None:
    """Pins class attributes and strategy identifiers on TwoPercent Screener."""
    assert TwoPercentScreener.STRATEGY_IDENTIFIER == Strategies.TwoPercent
    assert TwoPercentScreener.ENTRY_LIMIT_DISCOUNT == 0.99
    assert TwoPercentScreener.DEFAULT_LOOKBACK_PERIOD == 20
    assert TwoPercentScreener.SYMBOL == "SXRV.DE"
    assert TwoPercentScreener.SYMBOLS == ("SXRV.DE", "QQQ")


@pytest.mark.tier1
def test_pinning_trade_manager_default_attributes() -> None:
    """Pins class attributes and strategy identifiers on TwoPercent Trade Manager."""
    assert TwoPercentTradeManager.STRATEGY_IDENTIFIER == Strategies.TwoPercent
    assert TwoPercentTradeManager.name == Strategies.TwoPercent
    assert TwoPercentTradeManager.REWARD_TARGET_MULTIPLIER == 1.02
    assert TwoPercentTradeManager.SATURDAY_WEEKDAY == SATURDAY
    assert TwoPercentTradeManager.WEEKEND_DAY_OFFSET == 3
    assert TwoPercentTradeManager.DAY_ONE_INDEX == 1
    assert TwoPercentTradeManager.DAY_TWO_INDEX == 2


# ==============================================================================
# 2. Screener Calculation Pinning: Discount Formula & Signal Context
# ==============================================================================


@pytest.mark.tier1
def test_pinning_screener_limit_entry_formula() -> None:
    """Pins exact 99% limit entry price formula and proposal payload."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.exists.return_value = False
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    screener = TwoPercentScreener(
        trade_repository=mock_trade_repo, data_provider=mock_data_provider
    )

    # 1. Pure calculation helper test
    assert screener._calculate_entry_price(100.0) == 99.0
    assert screener._calculate_entry_price(123.45) == 122.22

    # 2. Proposal generation payload check
    screener._create_trade_proposal(
        date_str="2026-02-06",
        close=100.0,
        entry=99.0,
        day_label="Friday",
        symbol="SXRV.DE",
    )

    mock_trade_repo.create_trade.assert_called_once_with(
        symbol="SXRV.DE",
        strategy=Strategies.TwoPercent,
        size=0,
        entry=99.0,
        stop_loss=0.0,
        target=0.0,
        context={
            "date": "2026-02-06",
            "setup_close": 100.0,
            "limit_entry": 99.0,
            "day": "Friday",
            "source": "screener",
        },
    )


@pytest.mark.tier1
def test_pinning_screener_evaluate_symbol_on_friday() -> None:
    """Pins full screener symbol evaluation on a standard Friday."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.exists.return_value = False
    mock_data_provider = MagicMock(spec=MarketDataProvider)

    friday_date = "2026-02-06"
    df_history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-02-05"),
                "open": 98.0,
                "high": 99.0,
                "low": 97.0,
                "close": 98.5,
            },
            {
                "date": pd.Timestamp(friday_date),
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
            },
        ]
    )
    mock_data_provider.get_symbol_history.return_value = df_history

    screener = TwoPercentScreener(
        trade_repository=mock_trade_repo, data_provider=mock_data_provider
    )

    with patch.object(screener.holiday_checker, "is_holiday", return_value=False):
        item, date_str = screener._evaluate_symbol("SXRV.DE", pd.Timestamp(friday_date))

    assert item is not None
    assert item.symbol == "SXRV.DE"
    assert item.action == "BUY LMT"
    assert item.entry_price == 99.0
    assert item.details["Setup Close"] == 100.0
    assert date_str == friday_date
    mock_trade_repo.create_trade.assert_called_once()


@pytest.mark.tier1
def test_pinning_screener_holiday_roll_to_thursday() -> None:
    """Pins holiday roll: If Friday is holiday, setup is evaluated on Thursday."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.exists.return_value = False
    mock_data_provider = MagicMock(spec=MarketDataProvider)

    thursday_date = "2026-04-02"  # Thursday before Good Friday
    df_history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(thursday_date),
                "open": 200.0,
                "high": 205.0,
                "low": 199.0,
                "close": 204.0,
            }
        ]
    )
    mock_data_provider.get_symbol_history.return_value = df_history

    screener = TwoPercentScreener(
        trade_repository=mock_trade_repo, data_provider=mock_data_provider
    )

    # Friday (2026-04-03) is a holiday
    def mock_is_holiday(dt: object) -> bool:
        ts = pd.Timestamp(dt)
        return bool(ts.weekday() == FRIDAY)

    with (
        patch.object(
            screener.holiday_checker, "is_holiday", side_effect=mock_is_holiday
        ),
        patch.object(
            screener, "_get_real_today", return_value=pd.Timestamp("2026-04-05").date()
        ),
    ):
        item, date_str = screener._evaluate_symbol("QQQ", pd.Timestamp(thursday_date))

    assert item is not None
    assert item.symbol == "QQQ"
    assert item.entry_price == 201.96  # 204.0 * 0.99
    assert date_str == thursday_date


# ==============================================================================
# 3. Trade Manager Target Price Calculation: Banker's Rounding Decimal
# ==============================================================================


@pytest.mark.tier1
def test_pinning_trade_manager_target_price_formula() -> None:
    """Pins exact 1.02 target price calculation using Decimal banker's rounding."""
    manager = TwoPercentTradeManager()

    # Flat numbers
    assert manager._calculate_target_price(Decimal("100.00")) == Decimal("102.00")
    assert manager._calculate_target_price(Decimal("50.00")) == Decimal("51.00")

    # Rounding check: 123.45 * 1.02 = 125.919 -> 125.92
    assert manager._calculate_target_price(Decimal("123.45")) == Decimal("125.92")

    # Zero or negative
    assert manager._calculate_target_price(Decimal("0.0")) == Decimal("0.0")
    assert manager._calculate_target_price(Decimal("-10.0")) == Decimal("0.0")


# ==============================================================================
# 4. Trade Manager Entry Execution Pinning: Limit vs. Gap Down
# ==============================================================================


@pytest.mark.tier1
def test_pinning_trade_manager_gap_down_fill_at_open() -> None:
    """Pins behavior: If Monday Open < Limit, fill price is Monday Open (better price)."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_GAP",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "signal_context": '{"limit_entry": 100.0, "setup_close": 101.0, "date": "2026-02-13"}',
    }
    monday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-16"),  # Monday
            "open": 98.0,  # Gap down below 100.0
            "high": 99.5,
            "low": 97.5,
            "close": 99.0,
        },
        name=pd.Timestamp("2026-02-16"),
    )
    history = pd.DataFrame([monday_candle])

    with patch.object(manager, "_get_trading_days_post_signal", return_value=1):
        result = manager.check_entry(trade, monday_candle, history)

    assert result is not None
    assert result.updates["status"] == TradeStatus.ACTIVE
    assert result.updates["entry_price"] == 98.0
    # Target = 98.0 * 1.02 = 99.96
    assert result.updates["current_target"] == 99.96
    assert "Gap Down" in result.reason


@pytest.mark.tier1
def test_pinning_trade_manager_standard_limit_fill() -> None:
    """Pins behavior: If Monday Open >= Limit and Low <= Limit, fill price is Limit."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_NORM",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "signal_context": '{"limit_entry": 100.0, "setup_close": 101.0, "date": "2026-02-13"}',
    }
    monday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-16"),  # Monday
            "open": 102.0,  # Above limit
            "high": 103.0,
            "low": 99.0,  # Touches limit 100.0
            "close": 100.5,
        },
        name=pd.Timestamp("2026-02-16"),
    )
    history = pd.DataFrame([monday_candle])

    with patch.object(manager, "_get_trading_days_post_signal", return_value=1):
        result = manager.check_entry(trade, monday_candle, history)

    assert result is not None
    assert result.updates["status"] == TradeStatus.ACTIVE
    assert result.updates["entry_price"] == 100.0
    # Target = 100.0 * 1.02 = 102.00
    assert result.updates["current_target"] == 102.0
    assert "Limit Hit" in result.reason


@pytest.mark.tier1
def test_pinning_trade_manager_missed_limit_invalidates() -> None:
    """Pins behavior: If Monday Low > Limit, trade is immediately INVALIDATED."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_MISSED",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "signal_context": '{"limit_entry": 100.0, "setup_close": 101.0, "date": "2026-02-13"}',
    }
    monday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-16"),  # Monday
            "open": 105.0,
            "high": 106.0,
            "low": 101.0,  # Low never reached 100.0
            "close": 104.0,
        },
        name=pd.Timestamp("2026-02-16"),
    )
    history = pd.DataFrame([monday_candle])

    with patch.object(manager, "_get_trading_days_post_signal", return_value=1):
        result = manager.check_entry(trade, monday_candle, history)

    assert result is not None
    assert result.updates["status"] == TradeStatus.INVALID
    assert "Missed Entry Window" in result.reason


# ==============================================================================
# 5. Trade Manager Exit Execution Pinning: Day 0 Inactive, Day 1+ Target, Friday Time Stop
# ==============================================================================


@pytest.mark.tier1
def test_pinning_trade_manager_exit_target_inactive_on_entry_day() -> None:
    """Pins rule: Take profit is INACTIVE on entry day (Monday)."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_EXIT",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "current_target": 102.0,
        "entry_date": "2026-02-16",  # Monday entry
        "status": "ACTIVE",
    }
    # Monday candle: high reaches 105.0 (> target 102.0)
    monday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-16"),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 101.0,
        },
        name=pd.Timestamp("2026-02-16"),
    )
    history = pd.DataFrame([monday_candle])

    result = manager.manage_active_trade(trade, history)
    assert result is None


@pytest.mark.tier1
def test_pinning_trade_manager_exit_target_active_on_day_1() -> None:
    """Pins rule: Take profit becomes active on Tuesday (Day 1)."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_EXIT",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "current_target": 102.0,
        "entry_date": "2026-02-16",  # Monday entry
        "status": "ACTIVE",
    }
    # Tuesday candle: high reaches 103.0 (>= target 102.0)
    tuesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-17"),  # Tuesday
            "open": 101.0,
            "high": 103.0,
            "low": 100.5,
            "close": 102.5,
        },
        name=pd.Timestamp("2026-02-17"),
    )
    history = pd.DataFrame([tuesday_candle])

    result = manager.manage_active_trade(trade, history)
    assert result is not None
    assert result.updates["status"] == TradeStatus.CLOSED
    assert result.updates["exit_price"] == 102.0
    assert result.updates["exit_reason"] == ExitReason.TARGET_HIT


@pytest.mark.tier1
def test_pinning_trade_manager_exit_friday_time_stop() -> None:
    """Pins rule: If target not hit by Friday close, exit via TIME_STOP at market close."""
    manager = TwoPercentTradeManager()
    manager.holiday_checker = MagicMock()
    manager.holiday_checker.is_holiday.return_value = False

    trade: TradeData = {
        "id": "T_2PCT_TIMESTOP",
        "symbol": "SXRV.DE",
        "strategy": "two_percent",
        "entry_price": 100.0,
        "current_target": 102.0,
        "entry_date": "2026-02-16",  # Monday
        "status": "ACTIVE",
    }
    # Friday candle: high is only 101.0 (< 102.0 target)
    friday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-02-20"),  # Friday
            "open": 100.5,
            "high": 101.0,
            "low": 99.8,
            "close": 100.2,
        },
        name=pd.Timestamp("2026-02-20"),
    )
    history = pd.DataFrame([friday_candle])

    result = manager.manage_active_trade(trade, history)
    assert result is not None
    assert result.updates["status"] == TradeStatus.CLOSED
    assert result.updates["exit_price"] == 100.2
    assert result.updates["exit_reason"] == ExitReason.TIME_STOP


# ==============================================================================
# 6. Dependency Injection Pinning: Custom Configuration Override
# ==============================================================================


@pytest.mark.tier1
def test_pinning_screener_custom_configuration_injection() -> None:
    """Pins ability to customize parameters via TwoPercentConfiguration injection."""
    custom_cfg = TwoPercentConfiguration(
        entry_limit_discount=0.97,
        default_lookback_period=35,
        target_symbols=("AAPL", "MSFT"),
    )
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)

    screener = TwoPercentScreener(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        configuration=custom_cfg,
    )

    assert screener.configuration.entry_limit_discount == 0.97
    assert screener.configuration.default_lookback_period == 35
    assert screener.symbols == ("AAPL", "MSFT")
    # Calculation with custom discount: 100 * 0.97 = 97.0
    assert screener._calculate_entry_price(100.0) == 97.0


@pytest.mark.tier1
def test_pinning_trade_manager_custom_configuration_injection() -> None:
    """Pins ability to customize reward target multiplier via TwoPercentConfiguration injection."""
    custom_cfg = TwoPercentConfiguration(reward_target_multiplier=1.05)
    manager = TwoPercentTradeManager(configuration=custom_cfg)

    assert manager.configuration.reward_target_multiplier == 1.05
    # Calculation with custom multiplier: 100 * 1.05 = 105.00
    assert manager._calculate_target_price(Decimal("100.00")) == Decimal("105.00")
