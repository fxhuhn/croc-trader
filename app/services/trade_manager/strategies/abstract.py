import datetime
import json
import logging
import uuid
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Protocol, final

import pandas as pd

from ....config import settings
from ....models import Order, OrderLeg, TradeParams
from ....types import ExitReason, TradeData, TradeStatus
from ..types import TradeTransition

logger = logging.getLogger(__name__)


class HolidayCheckerProtocol(Protocol):
    """Protocol defining the required holiday checking interface."""

    def is_holiday(self, check_date: datetime.date) -> bool: ...


class BaseTradeStrategy(ABC):
    """
    Interface and Base Class for all trading strategies.
    Provides shared logic for trade activation, exit management, and position sizing.
    """

    FRIDAY_INDEX: int = 4
    THURSDAY_INDEX: int = 3

    def _is_end_of_trading_week(
        self,
        current_date: pd.Timestamp,
        holiday_checker: HolidayCheckerProtocol,
    ) -> bool:
        """Checks if today is the last trading day of the week.

        Accounts for Friday holidays by treating Thursday as week-end.
        This is the single authoritative end-of-week check (DRY).

        Args:
            current_date: The date to evaluate.
            holiday_checker: An object or protocol implementing `is_holiday(date)`.

        Returns:
            bool: True if this is the last trading day of the week.
        """
        if current_date.dayofweek == self.FRIDAY_INDEX:
            return True

        if current_date.dayofweek == self.THURSDAY_INDEX:
            next_day = current_date + pd.Timedelta(days=1)
            is_holiday_fn = getattr(holiday_checker, "is_holiday", None)
            if is_holiday_fn and is_holiday_fn(next_day.date()):
                return True

        return False

    def _generate_time_stop_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        holiday_checker: HolidayCheckerProtocol,
    ) -> Order | None:
        """Generates a weekly time stop (MOC) exit order if the next trading day is the end of the week.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data.
            holiday_checker: An object with an `is_holiday(date)` method.

        Returns:
            Order | None: The exit order if the end of the week is reached, otherwise None.
        """
        quantity = int(trade.get("current_size") or 0)
        if quantity <= 0 or dataframe_history.empty:
            return None

        last_date = pd.Timestamp(dataframe_history.iloc[-1]["date"])
        next_day = last_date + pd.Timedelta(days=1)

        if self._is_end_of_trading_week(next_day, holiday_checker):
            return self._create_exit_order(
                symbol=trade["symbol"],
                quantity=quantity,
                price=Decimal("0.0"),
                order_type="MOC",
                time_in_force="DAY",
            )

        return None

    def _extract_entry_price(self, trade: TradeData) -> float:
        """Extracts and parses the entry limit price from the trade context.

        Args:
            trade: The trade data from the database.

        Returns:
            float: The parsed entry price or 0.0.
        """
        return float(trade.get("entry_price") or 0.0)

    @abstractmethod
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """
        Checks for CREATED trades if the entry was filled.

        Args:
            trade: The trade data from the database.
            candle: The current price candle.
            dataframe_history: Historical price data.
            active_symbols: Set of symbols with currently active positions.

        Returns:
            TradeTransition | None: The computed transition, or None.
        """
        raise NotImplementedError("Subclasses must implement check_entry")

    @final
    def manage_active_trade(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """
        Manages ACTIVE trades (Exit checks).
        Template method that unpacks data and calls _do_manage_active_trade.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data.
            latest_leaders: Set of leader symbols from the strategy.

        Returns:
            TradeTransition | None: The computed transition, or None.
        """
        if dataframe_history.empty:
            return None

        current_candle = dataframe_history.iloc[-1]
        date_string = str(current_candle["date"])

        return self._do_manage_active_trade(
            trade, current_candle, date_string, dataframe_history, latest_leaders
        )

    @abstractmethod
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: "pd.Series",
        date_string: str,
        dataframe_history: "pd.DataFrame",
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """
        Strategy-specific exit logic.
        """
        pass

    @final
    def generate_orders(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """
        Generates orders for the next trading day.
        Template method branching by trade status.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data.
            budget: Total available budget for the trade.
            created_symbols: Set of symbols with currently pending (CREATED) trades.
            reference_date: The target date for which orders are being generated.

        Returns:
            Order | None: The generated order object or None.
        """
        status = trade.get("status", "CREATED")
        if hasattr(status, "value"):
            status = status.value

        if status == "CREATED":
            return self._generate_entry_order(
                trade, dataframe_history, budget, created_symbols, reference_date
            )
        elif status == "ACTIVE":
            return self._generate_exit_order(
                trade, dataframe_history, budget, created_symbols, reference_date
            )
        return None

    @abstractmethod
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        pass

    @abstractmethod
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        pass

    @abstractmethod
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None,
    ) -> TradeParams | None:
        """
        Calculates parameters for logging or display.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data (optional).

        Returns:
            TradeParams | None: The calculated trade parameters.
        """
        raise NotImplementedError("Subclasses must implement get_current_parameters")

    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """
        Calculates daily state updates (e.g. dynamic thresholds, targets) for the trade.
        Base implementation returns empty updates.

        Args:
            trade: The current trade data.
            dataframe_history: The market history for the traded symbol.

        Returns:
            dict[str, object]: A dictionary containing fields to be merged into signal_context.
        """
        return {}

    # --- Shared Logic & Helpers (DRY) ---

    def _create_order(
        self,
        symbol: str,
        quantity: int,
        mode: str,
        entry: OrderLeg | None = None,
        exits: list[OrderLeg] | None = None,
        order_id: str | None = None,
    ) -> Order:
        """Universal order factory helper to construct an Order object."""
        return Order(
            id=order_id or str(uuid.uuid4()),
            symbol=symbol,
            quantity=quantity,
            mode=mode,
            entry=entry,
            exits=exits or [],
        )

    def _create_entry_order(
        self,
        symbol: str,
        quantity: int,
        entry_price: Decimal,
        order_type: str = "LMT",
        time_in_force: str = "DAY",
        order_id: str | None = None,
    ) -> Order:
        """Generates a standard Buy Entry Order."""
        return self._create_order(
            symbol=symbol,
            quantity=quantity,
            mode="Entry",
            entry=OrderLeg(
                action="BUY",
                type=order_type,
                price=entry_price,
                quantity=quantity,
                time_in_force=time_in_force,
            ),
            exits=[],
            order_id=order_id,
        )

    def _create_exit_order(
        self,
        symbol: str,
        quantity: int,
        price: Decimal | None = None,
        order_type: str = "MKT",
        time_in_force: str = "OPG",
        order_id: str | None = None,
    ) -> Order:
        """Generates a standard Sell Exit Order."""
        return self._create_order(
            symbol=symbol,
            quantity=quantity,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type=order_type,
                    price=price if price is not None else Decimal("0.0"),
                    quantity=quantity,
                    time_in_force=time_in_force,
                )
            ],
            order_id=order_id,
        )

    def _get_signal_date(self, trade: TradeData) -> pd.Timestamp | None:
        """Extracts the signal date from the JSON context."""
        date_value = self._get_context_value(trade, "date") or self._get_context_value(
            trade, "setup_date"
        )
        if date_value:
            try:
                return pd.Timestamp(date_value)
            except (ValueError, TypeError) as error:
                logger.warning(
                    "Failed to parse signal date '%s' for trade %s: %s",
                    date_value,
                    trade.get("id"),
                    error,
                )
                return None
        return None

    @staticmethod
    def _get_context_value(
        trade: TradeData | dict[str, object], key: str
    ) -> str | float | int | bool | None:
        """Extracts a single value from the JSON signal_context field.

        Handles both pre-parsed dict and raw JSON string formats
        since the repository layer may or may not have parsed the context.
        """
        context = BaseTradeStrategy._get_full_context(trade)
        value = context.get(key)
        if isinstance(value, str | float | int | bool) or value is None:
            return value
        return str(value)

    @staticmethod
    def _get_full_context(trade: TradeData | dict[str, object]) -> dict[str, object]:
        """Single authoritative parser for the full signal_context dictionary.

        This is the DRY replacement for all per-strategy context parsers.
        It handles both pre-parsed dicts (if the repository already deserialised
        the JSON) and raw JSON strings, so callers never need to branch on the
        type themselves.

        Args:
            trade: The trade data dictionary.

        Returns:
            dict[str, object]: The parsed context, or an empty dict on failure.
        """
        context_data = trade.get("signal_context")
        if not context_data:
            return {}
        if isinstance(context_data, dict):
            return context_data
        try:
            return json.loads(context_data)
        except (json.JSONDecodeError, TypeError) as error:
            logger.warning(
                "Failed to parse signal_context for trade %s: %s",
                trade.get("id"),
                error,
            )
            return {}

    def _get_trading_days_post_signal(
        self, trade: TradeData, dataframe_history: pd.DataFrame
    ) -> int:
        """
        Calculates how many trading days have passed since the signal.

        Args:
            trade: The trade data.
            dataframe_history: Market history including the signal day.

        Returns:
            int: Number of trading days (candles) strictly after the signal date.
        """
        signal_date = self._get_signal_date(trade)
        if not signal_date or dataframe_history.empty:
            return 0

        # Ensure signal_date is compared as Timestamp
        # Use history to count valid trading sessions
        history_dates = pd.to_datetime(dataframe_history["date"])
        return len(dataframe_history[history_dates > signal_date])

    def _calculate_position_size(
        self, fill_price: float, stop_loss: float, risk_amount: float
    ) -> int:
        """
        Calculates position size based on per-share risk.

        Args:
            fill_price: The entry price.
            stop_loss: The initial stop loss.
            risk_amount: The dollar amount to risk.

        Returns:
            int: Number of shares.
        """
        if stop_loss <= 0 or fill_price == stop_loss:
            return 0
        risk_per_share = abs(fill_price - stop_loss)
        return int(risk_amount / risk_per_share)

    def _resolve_position_size(
        self,
        trade: TradeData,
        fill_price: float,
        stop_loss: float = 0.0,
        fallback_budget: float = 0.0,
        fallback_risk: float = 100.0,
    ) -> int:
        """Determines position size via risk-based or budget-based fallback.

        This is a pure computation function (no direct settings lookup or logging side-effects).
        """
        # 1. Pre-existing size from database
        existing_size = float(
            trade.get("initial_size") or trade.get("current_size") or 0.0
        )
        if existing_size > 0:
            return int(existing_size)

        # 2. Risk-based sizing
        if stop_loss > 0 and fill_price != stop_loss:
            risk_amount = float(
                self._get_context_value(trade, "risk_amount")
                or trade.get("risk_amount")
                or fallback_risk
            )
            risk_based_size = self._calculate_position_size(
                fill_price, stop_loss, risk_amount
            )
            if risk_based_size > 0:
                return risk_based_size

        # 3. Budget-based fallback
        if fill_price > 0:
            budget = float(
                self._get_context_value(trade, "budget")
                or trade.get("budget")
                or fallback_budget
            )
            if budget > 0:
                return int(budget / fill_price)

        return 0

    def _execute_activation(
        self,
        trade: TradeData,
        fill_price: float,
        reason: str,
        date_string: str,
        extra_updates: dict[str, object] | None = None,
    ) -> TradeTransition:
        """Moves a trade from CREATED to ACTIVE status."""
        strategy_key = getattr(
            getattr(self, "name", ""), "value", str(getattr(self, "name", ""))
        )

        fallback_budget = settings.app.portfolio.get_budget(strategy_key)
        fallback_risk = settings.app.portfolio.get_risk_amount(strategy_key)

        stop_loss = float(trade.get("current_stop_loss") or 0.0)
        size = self._resolve_position_size(
            trade,
            fill_price,
            stop_loss,
            fallback_budget=fallback_budget,
            fallback_risk=fallback_risk,
        )

        if size <= 0:
            logger.warning(
                "[%s] No budget found for sizing fallback or size calculated is 0. "
                "Check settings.yaml and signal_context.",
                trade.get("symbol"),
            )
            return TradeTransition(
                updates={},
                reason="ERROR: Zero Size",
                message="ERROR: Zero Size",
            )

        if not (trade.get("initial_size") or trade.get("current_size")):
            logger.debug(
                "Using fallback sizing for %s (size: %s)",
                trade.get("symbol"),
                size,
            )

        update_payload: dict[str, object] = {
            "status": TradeStatus.ACTIVE,
            "entry_date": date_string,
            "entry_price": fill_price,
            "initial_size": size,
            "current_size": size,
        }
        if extra_updates:
            update_payload.update(extra_updates)

        return TradeTransition(
            updates=update_payload,
            reason=f"{reason} FILLED @ {fill_price:.2f}",
            message=f"FILLED @ {fill_price:.2f} ({int(size)} Shares)",
        )

    def _execute_immediate_loss(
        self,
        trade: TradeData,
        fill_price: float,
        reason: str,
        stop_loss: float,
        date_string: str,
    ) -> TradeTransition:
        """Handles Day 1 turnarounds: entry and stop hit on same day."""
        strategy_key = getattr(
            getattr(self, "name", ""), "value", str(getattr(self, "name", ""))
        )

        fallback_budget = settings.app.portfolio.get_budget(strategy_key)
        fallback_risk = settings.app.portfolio.get_risk_amount(strategy_key)

        size = self._resolve_position_size(
            trade,
            fill_price,
            stop_loss,
            fallback_budget=fallback_budget,
            fallback_risk=fallback_risk,
        )

        if size <= 0:
            logger.warning(
                "[%s] Sizing failed during immediate loss execution.",
                trade.get("symbol"),
            )
            return TradeTransition(
                updates={},
                reason="ERROR: Zero Size",
                message="ERROR: Zero Size",
            )

        decimal_fill_price = Decimal(str(fill_price))
        decimal_stop_loss = Decimal(str(stop_loss))
        decimal_quantity = Decimal(str(int(size)))

        direction = str(self._get_context_value(trade, "direction") or "long").lower()
        if direction == "short":
            pnl_chunk = (decimal_fill_price - decimal_stop_loss) * decimal_quantity
        else:
            pnl_chunk = (decimal_stop_loss - decimal_fill_price) * decimal_quantity

        profit_and_loss = float(pnl_chunk)

        return TradeTransition(
            updates={
                "status": TradeStatus.CLOSED,
                "entry_date": date_string,
                "entry_price": fill_price,
                "initial_size": size,
                "current_size": size,
                "exit_date": date_string,
                "exit_price": stop_loss,
                "exit_reason": ExitReason.STOP_LOSS,
                "realized_pnl": profit_and_loss,
            },
            reason=f"{reason} FILLED @ {fill_price:.2f} -> STOPPED @ {stop_loss:.2f}",
            message=(
                f"FILLED @ {fill_price:.2f} -> STOPPED @ {stop_loss:.2f} "
                f"(PnL: {profit_and_loss:.2f})"
            ),
        )

    def _close_trade(
        self,
        trade: TradeData,
        exit_price: float,
        reason: str,
        date_string: str,
    ) -> TradeTransition:
        """Closes a trade and calculates final PnL."""
        entry_price = float(trade.get("entry_price") or 0.0)
        current_size = float(trade.get("current_size") or 0.0)
        decimal_entry_price = Decimal(str(entry_price))
        decimal_exit_price = Decimal(str(exit_price))
        decimal_quantity = Decimal(str(int(current_size)))

        direction = str(self._get_context_value(trade, "direction") or "long").lower()
        if direction == "short":
            pnl_chunk = (decimal_entry_price - decimal_exit_price) * decimal_quantity
        else:
            pnl_chunk = (decimal_exit_price - decimal_entry_price) * decimal_quantity

        total_pnl = float(Decimal(str(trade.get("realized_pnl") or "0.0")) + pnl_chunk)

        return TradeTransition(
            updates={
                "status": TradeStatus.CLOSED,
                "exit_reason": reason,
                "exit_price": exit_price,
                "exit_date": date_string,
                "realized_pnl": total_pnl,
                "current_size": 0,
            },
            reason=reason,
            message=f"{reason} @ {exit_price:.2f} (PnL: {total_pnl:.2f})",
        )

    def _invalidate_trade(
        self,
        trade: TradeData,
        low_price: float,
        stop_loss: float,
        date_string: str,
    ) -> TradeTransition:
        """Invalidates a setup before entry.

        Args:
            trade: The trade.
            low_price: The low of the day.
            stop_loss: The stop loss level.
            date_string: The date.

        Returns:
            TradeTransition: Computed state updates.
        """
        outcome_reason = (
            f"SETUP INVALIDATED: Low {low_price:.2f} <= Stop {stop_loss:.2f}"
        )
        return TradeTransition(
            updates={
                "status": TradeStatus.INVALID,
                "exit_reason": ExitReason.INVALIDATED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason=outcome_reason,
            message=outcome_reason,
        )

    def _expire_trade(self, trade: TradeData, date_string: str) -> TradeTransition:
        """Signals expiration when entry isn't hit within time limit.

        Unified transition to TradeStatus.INVALID / ExitReason.INVALIDATED.
        """
        return TradeTransition(
            updates={
                "status": TradeStatus.INVALID,
                "exit_reason": ExitReason.INVALIDATED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason="EXPIRED (Time Limit Exceeded)",
            message="EXPIRED",
        )

    def _reject_setup(
        self,
        trade: TradeData,
        date_string: str,
        reason: str,
    ) -> TradeTransition:
        """Rejects a setup that never became active (e.g. missed entry window).

        Args:
            trade: The trade data.
            date_string: Current date.
            reason: Detailed reason for rejection.

        Returns:
            TradeTransition: Computed state updates.
        """
        return TradeTransition(
            updates={
                "status": TradeStatus.INVALID,
                "exit_reason": ExitReason.INVALIDATED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason=f"REJECTED: {reason}",
            message=f"REJECTED: {reason}",
        )

    def _get_strategy_budget(
        self, trade: TradeData, override_budget: float = 0.0
    ) -> float:
        """Centralized DRY helper to resolve budget for the strategy."""
        strategy_key = getattr(self.name, "value", str(self.name))
        config_budget = settings.app.portfolio.get_budget(strategy_key)
        return float(trade.get("budget") or override_budget or config_budget)
