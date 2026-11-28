"""TGIM (Thank God It's Monday) Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Limit On Close (LOC) on Monday close (Bar 0) with limit = threshold_price - tick_size
     (matching RealTest strict inequality C < min(C[1], C[2])).
   - Position size calculated from portfolio budget allocation via Decimal floor rounding.
2. Exits (ACTIVE -> CLOSED):
   - Bar 1 (Tuesday): SELL LOC at C[1] + tick_size (ExitReason.TAKE_PROFIT if Tuesday Close > Monday Close).
   - Bar 2 (Wednesday): Unconditional SELL MOC exit (ExitReason.TAKE_PROFIT if Wednesday Close > Tuesday Close,
     otherwise ExitReason.TIME_STOP).
"""

import logging
from decimal import ROUND_FLOOR, Decimal
from typing import final, override

import pandas as pd

from ....const import ExitReason, Strategies
from ....models import Order
from ....tools.market_holidays import MarketHolidayChecker
from ....tools.trading_calendar import get_last_completed_trading_day
from ....types import TradeData
from ...screener.strategies.tgim import TGIMConfiguration
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, OrderOptions

logger = logging.getLogger(__name__)

MAX_TGIM_HOLDING_BARS: int = 2


def evaluate_tgim_exit(
    bars_held: int,
    current_close: Decimal,
    previous_close: Decimal,
    max_holding_bars: int = MAX_TGIM_HOLDING_BARS,
) -> ExitReason | None:
    """Pure calculation: Evaluates TGIM exit logic without side effects.

    - Bar 1 (Tuesday): Take profit if current_close > previous_close.
    - Bar 2 (Wednesday): Take profit if current_close > previous_close, else Time Stop exit.
    """
    if bars_held < 1:
        return None

    if current_close > previous_close:
        return ExitReason.TAKE_PROFIT

    if bars_held >= max_holding_bars:
        return ExitReason.TIME_STOP

    return None


def calculate_tgim_position_quantity(
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
class TGIMTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'TGIM' strategy.

    Rules:
    1. Entry: LOC entry executed on Monday setup close with limit = threshold_price - tick_size.
    2. Exits:
       - Bar 1 (Tuesday): SELL LOC at C[1] + tick_size (exits if Tuesday Close > Monday Close).
       - Bar 2 (Wednesday): Unconditional SELL MOC exit (TAKE_PROFIT if Wednesday Close > Tuesday Close,
         otherwise TIME_STOP).
    """

    STRATEGY_IDENTIFIER = Strategies.TGIM
    name = Strategies.TGIM
    MAX_HOLDING_BARS = MAX_TGIM_HOLDING_BARS

    def __init__(
        self,
        configuration: TGIMConfiguration | None = None,
        *,
        holiday_checker: MarketHolidayChecker | None = None,
    ) -> None:
        """Initializes the TGIM trade manager strategy with optional configuration and holiday checker."""
        super().__init__()
        self.configuration = configuration or TGIMConfiguration()
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    @override
    def _resolve_position_size(
        self,
        trade: TradeData,
        fill_price: float,
        stop_loss: float = 0.0,
        fallback_budget: float = 0.0,
        fallback_risk: float = 100.0,
    ) -> int:
        """Resolves TGIM position size using strict Decimal floor rounding."""
        existing_size = float(
            trade.get("initial_size") or trade.get("current_size") or 0.0
        )
        if existing_size > 0:
            return int(existing_size)

        if stop_loss > 0 and fill_price != stop_loss:
            return super()._resolve_position_size(
                trade,
                fill_price,
                stop_loss=stop_loss,
                fallback_budget=fallback_budget,
                fallback_risk=fallback_risk,
            )

        budget_val = float(
            self._get_context_value(trade, "budget")
            or trade.get("budget")
            or fallback_budget
        )
        return calculate_tgim_position_quantity(
            Decimal(str(budget_val)),
            Decimal(str(fill_price)),
        )

    def _calculate_bars_held(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None,
    ) -> int:
        """Calculates number of trading bars held since entry date."""
        entry_date_string = trade.get("entry_date")
        if (
            not entry_date_string
            or dataframe_history is None
            or dataframe_history.empty
        ):
            return 0

        entry_date = pd.Timestamp(entry_date_string).date()
        dates = pd.to_datetime(dataframe_history["date"]).dt.date
        history_from_entry = dataframe_history[dates >= entry_date]
        return max(0, len(history_from_entry) - 1)

    def _get_upcoming_execution_bar(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None,
    ) -> int:
        """Calculates 1-based index of the upcoming trading session for order generation.

        - Tuesday pre-market: 0 completed bars held -> returns 1 (Bar 1).
        - Wednesday pre-market: 1 completed bar held -> returns 2 (Bar 2).
        """
        completed_bars = self._calculate_bars_held(trade, dataframe_history)
        return completed_bars + 1

    @override
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates LOC entry order for CREATED trades on their setup_date."""
        if reference_date:
            setup_date = self._get_setup_date(trade)
            if setup_date and pd.Timestamp(reference_date).date() != setup_date:
                return None

        threshold_price_val = trade.get("entry_price") or 0.0
        trade_budget_val = self._get_strategy_budget(trade, budget)

        threshold_price = Decimal(str(threshold_price_val))
        limit_price = threshold_price - self.configuration.tick_size
        if limit_price <= Decimal("0"):
            return None

        trade_budget = Decimal(str(trade_budget_val))
        quantity = calculate_tgim_position_quantity(trade_budget, limit_price)
        if quantity < 1:
            return None

        return self._create_entry_order(
            symbol=trade["symbol"],
            quantity=quantity,
            entry_price=limit_price,
            options=OrderOptions(order_type="LOC", time_in_force="DAY"),
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
        """Generates exit orders for ACTIVE trades (Bar 1: LOC, Bar 2: MOC)."""
        quantity = int(trade.get("current_size") or 0)
        if quantity <= 0 or dataframe_history.empty:
            return None

        if reference_date:
            latest_candle_date = pd.to_datetime(
                dataframe_history["date"].iloc[-1]
            ).date()
            target_execution_date = pd.to_datetime(reference_date).date()
            if latest_candle_date >= target_execution_date:
                raise ValueError(
                    f"Lookahead bias in TGIM order generation: history contains data "
                    f"from {latest_candle_date} on or after execution date "
                    f"{target_execution_date}"
                )
            expected_previous_day = get_last_completed_trading_day(
                target_execution_date, self.holiday_checker
            )
            if latest_candle_date < expected_previous_day:
                raise ValueError(
                    f"Stale market data in TGIM order generation: expected candle for "
                    f"{expected_previous_day} before {target_execution_date}, "
                    f"got {latest_candle_date}"
                )

        upcoming_bar = self._get_upcoming_execution_bar(trade, dataframe_history)
        if upcoming_bar >= self.configuration.max_holding_bars:
            return self._create_exit_order(
                symbol=trade["symbol"],
                quantity=quantity,
                price=Decimal("0.0"),
                options=OrderOptions(order_type="MOC", time_in_force="DAY"),
            )

        raw_previous_close = dataframe_history["close"].iloc[-1]
        if pd.isna(raw_previous_close):
            return None

        previous_close = Decimal(str(raw_previous_close))
        if previous_close <= Decimal("0"):
            return None

        limit_price = previous_close + self.configuration.tick_size
        return self._create_exit_order(
            symbol=trade["symbol"],
            quantity=quantity,
            price=limit_price,
            options=OrderOptions(order_type="LOC", time_in_force="DAY"),
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on Monday close (Bar 0) if Monday Close < threshold.

        Invalidates setup if entry window is missed, position is already active, or condition fails.
        """
        threshold_price = Decimal(str(trade.get("entry_price") or 0.0))
        if threshold_price <= Decimal("0"):
            return None

        current_close = Decimal(str(candle["close"]))
        candle_date = pd.Timestamp(candle["date"]).date()
        date_string = candle_date.strftime("%Y-%m-%d")

        setup_date = self._get_setup_date(trade)
        if setup_date:
            if candle_date < setup_date:
                return None
            if candle_date > setup_date:
                return self._reject_setup(
                    trade,
                    date_string,
                    "Missed Entry Window (Monday Close)",
                )

        if active_symbols and trade["symbol"] in active_symbols:
            return self._reject_setup(
                trade,
                date_string,
                "Duplicate Position (Already Active)",
            )

        if current_close < threshold_price:
            return self._execute_activation(
                trade,
                float(current_close),
                "Monday LOC Entry",
                date_string,
            )

        return self._reject_setup(
            trade,
            date_string,
            "Missed Entry Window (Monday Close)",
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
        """Manages Exits for TGIM:

        - Bar 1 (Tuesday): Exit if Close > Monday Close (ExitReason.TAKE_PROFIT).
        - Bar 2 (Wednesday): Exit if Close > Tuesday Close (ExitReason.TAKE_PROFIT) OR Time Exit (ExitReason.TIME_STOP).
        """
        entry_date_string = trade.get("entry_date")
        if not entry_date_string or dataframe_history.empty:
            return None

        entry_date = pd.Timestamp(entry_date_string).date()
        dates = pd.to_datetime(dataframe_history["date"]).dt.date
        history_from_entry = dataframe_history[dates >= entry_date]

        bars_held = len(history_from_entry) - 1
        if bars_held < 1:
            return None

        current_close = Decimal(str(current_candle["close"]))
        prev_candle = dataframe_history.iloc[-2]
        prev_close = Decimal(str(prev_candle["close"]))

        exit_reason = evaluate_tgim_exit(
            bars_held=bars_held,
            current_close=current_close,
            previous_close=prev_close,
            max_holding_bars=self.configuration.max_holding_bars,
        )

        if exit_reason is not None:
            return self._close_trade(
                trade,
                float(current_close),
                exit_reason,
                date_string,
            )

        return None
