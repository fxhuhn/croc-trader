"""Wounded Bull Sprint Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Market On Open (MOO) on the trading session following setup generation (Bar t+1).
   - Order: order_type='MKT', tif='OPG'.
   - Position size: allocated budget / entry price (integral shares).
2. Exits (ACTIVE -> CLOSED):
   - Exit 1 (clexit): Close > High[1] and not WBS_setup (profit taking at NextOpen).
   - Exit 2 (TE): BarsHeld >= 6 (time exit at NextOpen).
   - Order: order_type='MKT', tif='OPG'.
   - Execution price: Open of the subsequent trading session.
"""

import datetime
import logging
from decimal import ROUND_FLOOR, Decimal
from typing import final, override

import pandas as pd

from ....const import ExitReason, Strategies
from ....models import Order
from ....tools.indicators import calculate_atr, calculate_roc, calculate_sma
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.trading_calendar import (
    get_remaining_trading_days_in_month,
    get_trading_day_of_month,
)
from ....types import TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, OrderOptions

logger = logging.getLogger(__name__)

DEFAULT_MAX_HOLDING_BARS: int = 6
DEFAULT_SIGNAL_DAY_THRESHOLD: int = 14
DEFAULT_MIN_DAYS_LEFT_IN_MONTH: int = 3
MIN_HISTORY_FOR_PREV_HIGH: int = 2
MIN_HISTORY_FOR_SMA200: int = 200


def calculate_wbs_bars_held(
    entry_date: datetime.date | None,
    dates_series: pd.Series,
) -> int:
    """Pure calculation: Computes completed trading bars held since entry date (Scenario A).

    - Day of entry (Open fill, Close): 1 bar held.
    - Day 6 EOD: 6 bars held (triggers TE for Day 7 Open).
    """
    if entry_date is None or dates_series.empty:
        return 0

    valid_dates = pd.to_datetime(dates_series).dt.date
    bars_from_entry = valid_dates[valid_dates >= entry_date]
    return len(bars_from_entry)


def is_wbs_setup_active(
    dataframe_history: pd.DataFrame,
    holiday_checker: MarketHolidayChecker | None = None,
    min_price: float = 5.0,
    min_volume_ma: float = 1_000_000.0,
    min_volatility: float = 0.025,
) -> bool:
    """Pure calculation: Evaluates whether the symbol meets WBS technical and calendar setup criteria.

    Criteria:
    1. Close > 5.0
    2. 20-day Volume SMA > 1,000,000
    3. Close > 200-day Close SMA (structural uptrend)
    4. ATR(3) / Close > 2.5% (volatility)
    5. ROC(10) < 0 (short-term pullback)
    6. Calendar window: TradingDayOfMonth > 14 and remaining trading days in month >= 3
    """
    if dataframe_history.empty or len(dataframe_history) < MIN_HISTORY_FOR_SMA200:
        return False

    close_series = dataframe_history["close"].astype(float)
    volume_series = dataframe_history["volume"].astype(float)
    high_series = dataframe_history["high"].astype(float)
    low_series = dataframe_history["low"].astype(float)

    current_close = float(close_series.iloc[-1])
    volume_sma_series = calculate_sma(volume_series, 20)
    trend_sma_series = calculate_sma(close_series, 200)
    atr_series = calculate_atr(high_series, low_series, close_series, 3)
    roc_10_series = calculate_roc(close_series, 10)

    is_technical_valid = (
        current_close > min_price
        and not volume_sma_series.empty
        and float(volume_sma_series.iloc[-1]) > min_volume_ma
        and not trend_sma_series.empty
        and current_close > float(trend_sma_series.iloc[-1])
        and not atr_series.empty
        and (float(atr_series.iloc[-1]) / current_close) > min_volatility
        and not roc_10_series.empty
        and float(roc_10_series.iloc[-1]) < 0.0
    )
    if not is_technical_valid:
        return False

    last_date_val = dataframe_history.iloc[-1]["date"]
    candle_date = (
        pd.Timestamp(last_date_val).date()
        if isinstance(last_date_val, str)
        else last_date_val.date()
    )
    checker = holiday_checker or MarketHolidayChecker()
    tdom = get_trading_day_of_month(candle_date, checker)
    td_left = get_remaining_trading_days_in_month(candle_date, checker) - 1

    return (
        tdom > DEFAULT_SIGNAL_DAY_THRESHOLD
        and td_left >= DEFAULT_MIN_DAYS_LEFT_IN_MONTH
    )


def evaluate_wbs_exit(
    bars_held: int,
    current_close: float,
    previous_high: float,
    is_setup_today: bool = False,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
) -> tuple[bool, str | None]:
    """Pure calculation: Evaluates WBS exit conditions at EOD without side effects.

    Priority:
    1. clexit: Close > High[1] and not WBS_setup
    2. TE: BarsHeld >= 6

    Returns:
        tuple[bool, str | None]: (should_exit, reason_code)
    """
    if current_close > previous_high and not is_setup_today:
        return True, "clexit"

    if bars_held >= max_holding_bars:
        return True, "TE"

    return False, None


def calculate_wbs_position_quantity(
    allocated_budget: Decimal,
    entry_price: Decimal,
) -> int:
    """Calculates integer share quantity using strict Decimal floor rounding."""
    if entry_price <= Decimal("0") or allocated_budget <= Decimal("0"):
        return 0

    raw_quantity = (allocated_budget / entry_price).quantize(
        Decimal("1"), rounding=ROUND_FLOOR
    )
    return int(raw_quantity)


@final
class WoundedBullSprintTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for Wounded Bull Sprint."""

    STRATEGY_IDENTIFIER = Strategies.WoundedBullSprint
    name = Strategies.WoundedBullSprint
    max_holding_bars = DEFAULT_MAX_HOLDING_BARS

    def __init__(
        self,
        max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
        holiday_checker: MarketHolidayChecker | None = None,
    ) -> None:
        """Initializes Wounded Bull Sprint trade manager strategy."""
        super().__init__()
        self.max_holding_bars = max_holding_bars
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    def _get_entry_date(self, trade: TradeData) -> datetime.date | None:
        """Resolves entry date as a datetime.date object."""
        entry_date_raw = trade.get("entry_date")
        if not entry_date_raw:
            return None
        try:
            return datetime.date.fromisoformat(str(entry_date_raw)[:10])
        except (ValueError, TypeError):
            return None

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """Provides daily updates to the trade context, tracking bars_held statelessly."""
        if dataframe_history.empty:
            return {}

        entry_date = self._get_entry_date(trade)
        dates_series = dataframe_history["date"]
        bars_held = calculate_wbs_bars_held(entry_date, dates_series)
        latest_date_str = str(dataframe_history.iloc[-1]["date"])

        updates: dict[str, object] = {
            "bars_held": bars_held,
            "last_processed_date": latest_date_str,
        }

        if len(dataframe_history) >= MIN_HISTORY_FOR_PREV_HIGH:
            current_close = float(dataframe_history.iloc[-1]["close"])
            prev_high = float(dataframe_history.iloc[-2]["high"])
            is_setup_today = is_wbs_setup_active(
                dataframe_history,
                self.holiday_checker,
            )
            should_exit, exit_code = evaluate_wbs_exit(
                bars_held=bars_held,
                current_close=current_close,
                previous_high=prev_high,
                is_setup_today=is_setup_today,
                max_holding_bars=self.max_holding_bars,
            )
            if should_exit:
                updates["pending_exit"] = exit_code
            else:
                updates["pending_exit"] = None

        return updates

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on the next trading session open following setup (NextOpen MOO Entry)."""
        candle_date = (
            pd.Timestamp(candle["date"]).date()
            if isinstance(candle["date"], str)
            else candle["date"].date()
        )

        setup_date = self._get_setup_date(trade)
        if setup_date and candle_date <= setup_date:
            return None

        open_price = float(candle.get("open") or trade.get("entry_price") or 0.0)
        if open_price <= 0:
            return None

        date_string = candle_date.strftime("%Y-%m-%d")
        return self._execute_activation(
            trade,
            open_price,
            "Wounded Bull Sprint MOO Entry",
            date_string,
        )

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages NextOpen Exits (clexit or TE) on current candle open if triggered prior."""
        context = self._get_full_context(trade)
        entry_date = self._get_entry_date(trade)
        if not entry_date:
            return None

        history_prior = dataframe_history[
            dataframe_history["date"] < current_candle["date"]
        ]

        if not history_prior.empty and len(history_prior) >= 1:
            bars_held_prior = calculate_wbs_bars_held(entry_date, history_prior["date"])
            if len(history_prior) >= MIN_HISTORY_FOR_PREV_HIGH:
                prior_close = float(history_prior.iloc[-1]["close"])
                prior_prev_high = float(history_prior.iloc[-2]["high"])
            else:
                prior_close = float(history_prior.iloc[-1]["close"])
                prior_prev_high = float(history_prior.iloc[-1]["high"])

            is_setup_prior = False
            if latest_leaders is not None:
                is_setup_prior = str(trade.get("symbol", "")) in latest_leaders
            if not is_setup_prior:
                is_setup_prior = is_wbs_setup_active(
                    history_prior,
                    self.holiday_checker,
                )

            should_exit_prior, exit_code_prior = evaluate_wbs_exit(
                bars_held=bars_held_prior,
                current_close=prior_close,
                previous_high=prior_prev_high,
                is_setup_today=is_setup_prior,
                max_holding_bars=self.max_holding_bars,
            )

            if should_exit_prior and exit_code_prior:
                exit_fill_price = float(current_candle["open"])
                reason_name = (
                    ExitReason.TAKE_PROFIT
                    if exit_code_prior == "clexit"
                    else ExitReason.TIME_STOP
                )
                logger.info(
                    "Closing WBS trade ID %s on %s: %s at Open (fill=%.2f)",
                    trade.get("id"),
                    date_string,
                    exit_code_prior,
                    exit_fill_price,
                )
                return self._close_trade(
                    trade,
                    exit_fill_price,
                    reason_name,
                    date_string,
                )

        pending_exit = context.get("pending_exit")
        if pending_exit:
            exit_fill_price = float(current_candle["open"])
            reason_name = (
                ExitReason.TAKE_PROFIT
                if pending_exit == "clexit"
                else ExitReason.TIME_STOP
            )
            return self._close_trade(
                trade,
                exit_fill_price,
                reason_name,
                date_string,
            )

        return None

    @override
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates MKT / OPG NextOpen entry order for CREATED trades."""
        entry_price_val = self._extract_entry_price(trade)
        if entry_price_val <= 0.0 and not dataframe_history.empty:
            entry_price_val = float(dataframe_history.iloc[-1]["close"])

        trade_budget_val = self._get_strategy_budget(trade, budget)
        if entry_price_val <= 0 or trade_budget_val <= 0:
            return None

        entry_price = Decimal(str(entry_price_val))
        trade_budget = Decimal(str(trade_budget_val))

        quantity = calculate_wbs_position_quantity(trade_budget, entry_price)
        if quantity < 1:
            return None

        symbol = str(trade["symbol"])
        return self._create_entry_order(
            symbol=symbol,
            quantity=quantity,
            entry_price=Decimal("0.0"),
            options=OrderOptions(
                order_type="MKT",
                time_in_force="OPG",
                order_id=f"{symbol}_{self.name}",
            ),
        )

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates MKT / OPG NextOpen exit order for ACTIVE trades if clexit or TE triggered."""
        quantity = int(trade.get("current_size") or 0)
        if quantity <= 0 or dataframe_history.empty:
            return None

        entry_date = self._get_entry_date(trade)
        dates_series = dataframe_history["date"]
        bars_held = calculate_wbs_bars_held(entry_date, dates_series)

        if len(dataframe_history) < MIN_HISTORY_FOR_PREV_HIGH:
            return None

        current_close = float(dataframe_history.iloc[-1]["close"])
        prev_high = float(dataframe_history.iloc[-2]["high"])
        is_setup_today = is_wbs_setup_active(
            dataframe_history,
            self.holiday_checker,
        )

        should_exit, _ = evaluate_wbs_exit(
            bars_held=bars_held,
            current_close=current_close,
            previous_high=prev_high,
            is_setup_today=is_setup_today,
            max_holding_bars=self.max_holding_bars,
        )

        if not should_exit:
            return None

        symbol = str(trade["symbol"])
        return self._create_exit_order(
            symbol=symbol,
            quantity=quantity,
            price=Decimal("0.0"),
            options=OrderOptions(
                order_type="MKT",
                time_in_force="OPG",
                order_id=f"{symbol}_{self.name}_EXIT",
            ),
        )


__all__ = [
    "DEFAULT_MAX_HOLDING_BARS",
    "WoundedBullSprintTradeStrategy",
    "calculate_wbs_bars_held",
    "calculate_wbs_position_quantity",
    "evaluate_wbs_exit",
    "is_wbs_setup_active",
]
