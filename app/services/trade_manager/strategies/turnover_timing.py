import datetime
import json
import logging
from typing import final, override

import pandas as pd

from ....const import ExitReason, Strategies
from ....models import Order, TradeParams
from ....tools.market_holidays import MarketHolidayChecker
from ....types import TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, HolidayCheckerProtocol, OrderOptions

logger = logging.getLogger(__name__)


def is_green_candle(open_price: float, close_price: float) -> bool:
    """Pure calculation: Determines if a candle is green (Close > Open)."""
    return close_price > open_price


def calculate_consecutive_green_candles(
    dataframe_history: pd.DataFrame,
    start_date: pd.Timestamp | datetime.date | str | None = None,
    initial_setup_candle_green: bool = False,
) -> int:
    """Pure calculation: Computes the number of consecutive green candles ending at the latest candle.

    Reconstructs context statelessly from the historical price series, satisfying
    the Stateless Execution Layers invariant. If historical prices in stocks.db are
    revised, this calculation automatically self-heals without accumulator drift.

    Args:
        dataframe_history: Market history containing 'date', 'open', 'close' columns.
        start_date: Optional setup date or entry date to anchor the trade lifecycle.
        initial_setup_candle_green: Baseline flag if setup candle is not present in history.

    Returns:
        int: Consecutive green candles ending at the latest bar in dataframe_history.
    """
    if (
        dataframe_history.empty
        or "open" not in dataframe_history.columns
        or "close" not in dataframe_history.columns
    ):
        return 1 if initial_setup_candle_green else 0

    candles = dataframe_history
    if start_date is not None:
        try:
            start_date_val = pd.Timestamp(start_date).date()
            dates = pd.to_datetime(candles["date"]).dt.date
            filtered_candles = candles[dates >= start_date_val]
            if not filtered_candles.empty:
                candles = filtered_candles
        except (ValueError, TypeError):
            pass

    count = 0
    first_date = pd.Timestamp(candles.iloc[0]["date"]).date()
    start_date_obj = pd.Timestamp(start_date).date() if start_date is not None else None

    # If the history slice starts strictly after start_date, seed with setup candle flag
    if start_date_obj and first_date > start_date_obj and initial_setup_candle_green:
        count = 1

    for row in candles.itertuples():
        open_price = float(row.open)
        close_price = float(row.close)
        if is_green_candle(open_price, close_price):
            count += 1
        else:
            count = 0

    return count


@final
class TurnoverTimingStrategy(BaseTradeStrategy):
    """
    Manages execution for 'TurnoverTiming' Strategy.

    Rules:
    1. Entry: Limit Buy at specific level from signal.
    2. Exit:
       - Early: Two consecutive green candles (+ logic).
       - Time: Friday Close.
    """

    name: str = str(Strategies.TurnOverTiming)
    """Unique identifier for the strategy."""

    MIN_GREEN_CANDLES_FOR_EXIT: int = 2
    """Threshold of consecutive green candles to trigger early exit."""

    def __init__(
        self,
        strategy_name: str | None = None,
        holiday_checker: HolidayCheckerProtocol | None = None,
    ) -> None:
        """Initializes the strategy, optionally overriding the default name or holiday checker.

        Args:
            strategy_name: Optional override for the strategy registry key.
            holiday_checker: Optional holiday checking protocol instance.
        """
        super().__init__()
        if strategy_name:
            self.name = strategy_name
        self._holiday_checker = holiday_checker or MarketHolidayChecker()

    def _resolve_green_candle_count(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> int:
        """Resolves the current green candle count statelessly from price history.

        Args:
            trade: The trade record.
            dataframe_history: Market history for the symbol.

        Returns:
            int: The calculated consecutive green candle count.
        """
        if (
            dataframe_history.empty
            or "open" not in dataframe_history.columns
            or "close" not in dataframe_history.columns
        ):
            context = self._get_full_context(trade)
            return int(str(context.get("green_candle_count") or 0))

        context = self._get_full_context(trade)
        setup_date = self._get_signal_date(trade)
        setup_was_green = bool(context.get("setup_candle_green", False))

        return calculate_consecutive_green_candles(
            dataframe_history=dataframe_history,
            start_date=setup_date,
            initial_setup_candle_green=setup_was_green,
        )

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams:
        """Standardizes TradeParams for turnover strategy.

        Args:
            trade: The trade data dictionary.
            dataframe_history: Optional historical market data.

        Returns:
            TradeParams: Object containing strategy parameters.
        """
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "variant": trade.get("strategy", "Standard"),
                "current_size": float(trade.get("current_size") or 0.0),
            },
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
        return self._generate_budget_entry_order(
            trade=trade,
            budget=budget,
            options=OrderOptions(order_type="LMT", time_in_force="DAY"),
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
        quantity = int(trade.get("current_size") or 0)

        if quantity <= 0:
            return None

        # Recompute green candle count statelessly from history
        green_candle_count = self._resolve_green_candle_count(trade, dataframe_history)

        # a) Green Sequence Exit (TRIGGERED)
        if green_candle_count >= self.MIN_GREEN_CANDLES_FOR_EXIT:
            return self._create_exit_order(
                trade["symbol"],
                quantity,
                options=OrderOptions(order_type="MKT", time_in_force="OPG"),
            )

        # b) Friday Time Stop (holiday-adjusted)
        time_stop_order = self._generate_time_stop_exit_order(
            trade, dataframe_history, self._holiday_checker
        )
        if time_stop_order is not None:
            return time_stop_order

        return None

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Checks if the limit entry was reached on the NEXT trading day only.

        Args:
            trade: The trade data.
            candle: The current market candle.
            dataframe_history: Historical market data.
            active_symbols: Currently active symbols set.

        Returns:
            TradeTransition | None: Result transition or None if no entry occurred.
        """
        limit_price = self._extract_entry_price(trade)

        if limit_price <= 0:
            return None

        # Determine Timestamps
        current_date_timestamp = pd.Timestamp(candle["date"])
        current_date = str(current_date_timestamp.date())
        signal_date_timestamp = self._get_signal_date(trade)

        if not signal_date_timestamp:
            return None

        # Count trading days strictly after signal
        days_post_signal = self._get_trading_days_post_signal(trade, dataframe_history)

        # 1. Too early (Same day) -> count = 0
        if days_post_signal < 1:
            return None

        # 2. Evaluation for the first trading day after setup (Day 1)
        if days_post_signal == 1:
            return self._evaluate_day_one_entry(
                trade, candle, limit_price, current_date
            )

        # 3. Too late (Expired) -> count > 1
        return self._expire_trade(trade, current_date)

    def _evaluate_day_one_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        limit_price: float,
        current_date: str,
    ) -> TradeTransition | None:
        """Evaluates entry triggers strictly on the first day after the signal."""
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if low_price > limit_price:
            # Day 1 finished and price was never below limit -> Expire immediately!
            return self._expire_trade(trade, current_date)

        fill_price = (
            min(open_price, limit_price) if open_price < limit_price else limit_price
        )

        if fill_price <= 0:
            return None

        # Calculate initial green candle context
        close_price = float(candle["close"])
        context = self._get_full_context(trade)

        entry_is_green = is_green_candle(open_price, close_price)
        setup_was_green = bool(context.get("setup_candle_green", False))

        context["green_candle_count"] = (
            2 if (entry_is_green and setup_was_green) else (1 if entry_is_green else 0)
        )
        context["last_processed_date"] = current_date

        return self._execute_activation(
            trade,
            fill_price,
            "LIMIT",
            current_date,
            extra_updates={
                "signal_context": json.dumps(context, default=str, ensure_ascii=False)
            },
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
        """Manages Exits: Multi-Day Green sequence (Next Open) or Time Stop (EOD)."""
        context = self._get_full_context(trade)

        # Idempotency check: Do not process the same candle twice
        last_processed_date = context.get("last_processed_date")
        if last_processed_date == date_string:
            return None

        # 1. Signal-Specific Exit (Green Candles sequence)
        # Rule: If count prior to current candle >= 2, exit at current candle OPEN (Next Open Rule).
        history_prior = dataframe_history[
            dataframe_history["date"] < current_candle["date"]
        ]
        setup_date = self._get_signal_date(trade)
        setup_was_green = bool(context.get("setup_candle_green", False))

        if not history_prior.empty:
            dates = pd.to_datetime(history_prior["date"]).dt.date
            setup_date_obj = pd.Timestamp(setup_date).date() if setup_date else None
            has_setup_in_history = (
                setup_date_obj is not None and (dates <= setup_date_obj).any()
            )

            if has_setup_in_history:
                green_candle_count = calculate_consecutive_green_candles(
                    dataframe_history=history_prior,
                    start_date=setup_date,
                    initial_setup_candle_green=setup_was_green,
                )
            else:
                calculated_count = calculate_consecutive_green_candles(
                    dataframe_history=history_prior,
                    start_date=setup_date,
                    initial_setup_candle_green=setup_was_green,
                )
                green_candle_count = max(
                    calculated_count,
                    int(str(context.get("green_candle_count") or 0)),
                )
        else:
            green_candle_count = int(str(context.get("green_candle_count") or 0))

        # Check for Exit Trigger (Next Open)
        if green_candle_count >= self.MIN_GREEN_CANDLES_FOR_EXIT:
            return self._close_trade(
                trade,
                float(current_candle["open"]),
                ExitReason.GREEN_SEQUENCE,
                date_string,
            )

        # 2. End of Week Time Stop (Friday or Holiday-Thursday Close)
        current_date_timestamp = pd.Timestamp(current_candle["date"])

        if self._is_end_of_trading_week(current_date_timestamp, self._holiday_checker):
            return self._close_trade(
                trade,
                float(current_candle["close"]),
                ExitReason.TIME_STOP,
                date_string,
            )

        # No exit occurred; get_daily_updates handles green candle count tracking.
        return None

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """Provides daily updates to the trade context, tracking green candle count statelessly."""
        if dataframe_history.empty:
            return {}

        current_candle = dataframe_history.iloc[-1]
        date_string = str(current_candle["date"])

        green_candle_count = self._resolve_green_candle_count(trade, dataframe_history)

        return {
            "green_candle_count": green_candle_count,
            "last_processed_date": date_string,
        }
