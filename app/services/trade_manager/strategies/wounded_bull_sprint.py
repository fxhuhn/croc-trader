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
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.symbol_filter import SymbolFilter
from ....types import TradeData
from ...screener.strategies.wounded_bull_sprint import (
    DEFAULT_MAX_POSITIONS,
    normalize_trading_date,
)
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, OrderOptions

logger = logging.getLogger(__name__)

DEFAULT_MAX_HOLDING_BARS: int = 6
MIN_HISTORY_FOR_PREV_HIGH: int = 2


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
    if bars_held < 1:
        return False, None

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
    max_positions = DEFAULT_MAX_POSITIONS

    def __init__(
        self,
        max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
        holiday_checker: MarketHolidayChecker | None = None,
        max_positions: int = DEFAULT_MAX_POSITIONS,
    ) -> None:
        """Initializes Wounded Bull Sprint trade manager strategy."""
        super().__init__()
        self.max_holding_bars = max_holding_bars
        self.holiday_checker = holiday_checker or MarketHolidayChecker()
        self.max_positions = max_positions

    def _get_entry_date(self, trade: TradeData) -> datetime.date | None:
        """Resolves entry date as a datetime.date object."""
        return normalize_trading_date(trade.get("entry_date"))

    @staticmethod
    def _is_symbol_in_setups(symbol: str, qualifying_setups: set[str] | None) -> bool:
        """Checks case-insensitively whether symbol belongs to qualifying WBS setups."""
        if not qualifying_setups:
            return False
        normalized_symbol = symbol.upper().strip()
        return normalized_symbol in {s.upper().strip() for s in qualifying_setups}

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        *,
        qualifying_setups: set[str] | None = None,
    ) -> dict[str, object]:
        """Provides daily updates to the trade context, tracking bars_held statelessly."""
        if dataframe_history.empty:
            return {}

        entry_date = self._get_entry_date(trade)
        dates_series = dataframe_history["date"]
        bars_held = calculate_wbs_bars_held(entry_date, dates_series)
        latest_date = normalize_trading_date(dataframe_history.iloc[-1]["date"])
        latest_date_str = (
            latest_date.strftime("%Y-%m-%d")
            if latest_date is not None
            else str(dataframe_history.iloc[-1]["date"])[:10]
        )

        updates: dict[str, object] = {
            "bars_held": bars_held,
            "last_processed_date": latest_date_str,
        }

        if len(dataframe_history) >= MIN_HISTORY_FOR_PREV_HIGH:
            current_close = float(dataframe_history.iloc[-1]["close"])
            prev_high = float(dataframe_history.iloc[-2]["high"])
            is_setup_today = self._is_symbol_in_setups(
                str(trade.get("symbol", "")), qualifying_setups
            )
            should_exit, exit_code = evaluate_wbs_exit(
                bars_held=bars_held,
                current_close=current_close,
                previous_high=prev_high,
                is_setup_today=is_setup_today,
                max_holding_bars=self.max_holding_bars,
            )
            updates["pending_exit"] = exit_code if should_exit else None

        return updates

    def _check_entry_blocking_conditions(
        self,
        trade: TradeData,
        date_string: str,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None,
    ) -> TradeTransition | None:
        """Evaluates expiry, share-class duplicate, and portfolio capacity entry guards."""
        if self._get_trading_days_post_signal(trade, dataframe_history) > 1:
            return self._expire_trade(trade, date_string)

        if active_symbols is None:
            return None

        symbol = str(trade.get("symbol", "")).upper().strip()
        normalized_active = {s.upper().strip() for s in active_symbols}
        related_symbols = SymbolFilter().get_related_symbols(symbol)
        if related_symbols & normalized_active:
            return self._reject_setup(trade, date_string, "Position already active")
        if len(normalized_active) >= self.max_positions:
            return self._reject_setup(trade, date_string, "Max positions reached")
        return None

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on the next trading session open following setup (NextOpen MOO Entry)."""
        candle_date = normalize_trading_date(candle["date"])
        if candle_date is None:
            return None

        setup_date = self._get_setup_date(trade)
        if setup_date and candle_date <= setup_date:
            return None

        date_string = candle_date.strftime("%Y-%m-%d")
        blocking_transition = self._check_entry_blocking_conditions(
            trade, date_string, dataframe_history, active_symbols
        )
        if blocking_transition is not None:
            return blocking_transition

        open_price = float(candle.get("open") or trade.get("entry_price") or 0.0)
        if open_price <= 0:
            return None

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
        entry_date = self._get_entry_date(trade)
        candle_date = normalize_trading_date(current_candle["date"])
        if entry_date is None or candle_date is None or candle_date <= entry_date:
            return None

        history_dates = pd.to_datetime(dataframe_history["date"]).dt.date
        history_prior = dataframe_history[history_dates < candle_date]

        if not history_prior.empty:
            bars_held_prior = calculate_wbs_bars_held(entry_date, history_prior["date"])
            prior_close = float(history_prior.iloc[-1]["close"])
            prior_prev_high = (
                float(history_prior.iloc[-2]["high"])
                if len(history_prior) >= MIN_HISTORY_FOR_PREV_HIGH
                else float(history_prior.iloc[-1]["high"])
            )
            is_setup_prior = self._is_symbol_in_setups(
                str(trade.get("symbol", "")), latest_leaders
            )

            should_exit_prior, exit_code_prior = evaluate_wbs_exit(
                bars_held=bars_held_prior,
                current_close=prior_close,
                previous_high=prior_prev_high,
                is_setup_today=is_setup_prior,
                max_holding_bars=self.max_holding_bars,
            )

            if should_exit_prior and exit_code_prior:
                return self._build_exit_transition(
                    trade, current_candle, date_string, exit_code_prior
                )

        context = self._get_full_context(trade)
        pending_exit = context.get("pending_exit")
        last_processed_date = normalize_trading_date(context.get("last_processed_date"))
        if (
            isinstance(pending_exit, str)
            and pending_exit
            and last_processed_date is not None
            and candle_date > last_processed_date
        ):
            return self._build_exit_transition(
                trade, current_candle, date_string, pending_exit
            )

        return None

    def _build_exit_transition(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        exit_code: str,
    ) -> TradeTransition:
        """Constructs a CLOSED transition filled at the current session Open."""
        exit_fill_price = float(current_candle["open"])
        reason_name = (
            ExitReason.TAKE_PROFIT if exit_code == "clexit" else ExitReason.TIME_STOP
        )
        logger.info(
            "Closing WBS trade ID %s on %s: %s at Open (fill=%.2f)",
            trade.get("id"),
            date_string,
            exit_code,
            exit_fill_price,
        )
        return self._close_trade(
            trade,
            exit_fill_price,
            reason_name,
            date_string,
        )

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
        is_setup_today = self._is_symbol_in_setups(
            str(trade.get("symbol", "")), created_symbols
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
]
