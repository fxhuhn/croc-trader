import json
import logging
from typing import override, final

import pandas as pd

from ....types import EntryReason, ExitReason, TradeData, TradeStatus
from ....models import TradeParams, Order, OrderLeg
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)


@final
class SplitTargetStrategy(BaseTradeStrategy):
    """
    Manager for Croc Split Targets (TP1/TP3).

    Logic:
    - Entry: Breakout (Stop Buy).
    - Exit 1 (TP1): Sell 50% of position. Move SL to Entry.
    - Exit 2 (TP3): Sell remaining position.
    - Gap-Over Fix: If price gaps over TP3, close the full position immediately.
    """

    name = Strategies.SplitTarget

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
        repository: TradeRepository | None = None,
    ) -> TradeParams:
        """
        Calculates current strategy parameters for display/logging.

        Args:
            trade: The trade data structure.
            dataframe_history: Historical price data (unused here).
            repository: The trade repository (unused here).

        Returns:
            TradeParams: Container with stop_loss, targets, and phase info.
        """
        context = self._get_context(dict(trade))
        is_phase_2 = context.get("is_phase_2", False)

        take_profit_1 = float(context.get("take_profit_1") or context.get("tp1") or 0.0)
        take_profit_3 = float(context.get("take_profit_3") or context.get("tp3") or 0.0)

        # If TP1 is hit, Phase 2 targets TP3
        current_target = take_profit_3 if is_phase_2 else take_profit_1

        return TradeParams(
            stop_loss=float(trade.get("current_stop_loss") or 0.0),
            take_profit_1=current_target,
            extras={
                "entry_limit": float(trade.get("entry_price") or 0.0),
                "current_size": float(trade.get("current_size") or 0.0),
                "phase": "2 (TP3)" if is_phase_2 else "1 (TP1)",
            },
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """Checks for a breakout entry (Stop Buy) and validates the stop loss.

        Orchestrates the entry check by delegating to specialized helpers
        for signal expiration, fill detection, and same-day target checks.

        Args:
            trade: The trade data.
            candle: The current price candle.
            dataframe_history: Historical price data.
            repository: The trade repository.

        Returns:
            str | None: Status message if filled/invalidated, else None.
        """
        entry_price = float(trade.get("entry_price") or 0.0)
        stop_loss = float(trade.get("current_stop_loss") or 0.0)

        if entry_price <= 0:
            return None

        date_string = str(candle["date"])

        # 1. Signal expiration check — returns False to skip, str message, or None
        expiration_result = self._check_signal_expiration(
            trade, candle, repository, date_string
        )
        if expiration_result is False:
            return None  # Too early to act on this signal
        if expiration_result is not None:
            return expiration_result

        # 2. Detect fill and stop invalidation
        context = self._get_context(trade)
        direction = str(context.get("direction", "long")).lower()
        filled, fill_price, reason = self._detect_entry_fill(
            candle, entry_price, direction
        )
        is_stop_hit = self._is_stop_invalidated(candle, stop_loss, direction)

        if not filled:
            if is_stop_hit:
                low_price = float(candle["low"])
                return self._invalidate_trade(
                    trade, repository, low_price, stop_loss, date_string
                )
            return None

        # 3. Same-day exit checks (stop has priority over target)
        if is_stop_hit:
            return self._execute_immediate_loss(
                trade, repository, fill_price, reason, stop_loss, date_string
            )

        tp3_exit_price = self._check_same_day_tp3(candle, context, direction)
        if tp3_exit_price is not None:
            return self._execute_immediate_target(
                trade,
                repository,
                fill_price,
                reason,
                tp3_exit_price,
                stop_loss,
                date_string,
                context,
            )

        return self._execute_activation(
            trade, repository, fill_price, reason, date_string
        )

    def _check_signal_expiration(
        self,
        trade: TradeData,
        candle: pd.Series,
        repository: TradeRepository,
        date_string: str,
    ) -> str | bool | None:
        """Checks if the signal has expired or is too early to act on.

        Returns:
            str: Expiration status message (trade was expired).
            False: Signal is too early — skip without action.
            None: Signal is within the valid window — continue processing.
        """
        signal_date = self._get_signal_date(trade)
        if not signal_date:
            return None

        current_date_obj = pd.Timestamp(candle["date"])
        if current_date_obj.date() <= signal_date.date():
            return False  # Too early — not yet actionable

        if (current_date_obj.date() - signal_date.date()).days > 5:
            return self._expire_trade(trade, repository, date_string)

        return None

    def _detect_entry_fill(
        self,
        candle: pd.Series,
        entry_price: float,
        direction: str,
    ) -> tuple[bool, float, str]:
        """Detects whether the candle triggers a breakout fill.

        Handles both long (gap-up / breakout) and short (gap-down / breakdown)
        entry detection.

        Args:
            candle: The current price candle.
            entry_price: The trigger price for entry.
            direction: Trade direction ("long" or "short").

        Returns:
            Tuple of (filled, fill_price, reason).
        """
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if direction == "long":
            if open_price >= entry_price:
                return True, open_price, EntryReason.GAP_UP
            if high_price >= entry_price:
                return True, entry_price, EntryReason.BREAKOUT
        else:
            if open_price <= entry_price:
                return True, open_price, EntryReason.GAP_DOWN
            if low_price <= entry_price:
                return True, entry_price, EntryReason.BREAKDOWN

        return False, 0.0, ""

    def _is_stop_invalidated(
        self,
        candle: pd.Series,
        stop_loss: float,
        direction: str,
    ) -> bool:
        """Checks if the stop loss was hit on this candle.

        Args:
            candle: The current price candle.
            stop_loss: The stop loss price level.
            direction: Trade direction ("long" or "short").

        Returns:
            bool: True if stop loss was breached.
        """
        if stop_loss <= 0:
            return False

        if direction == "long":
            return float(candle["low"]) <= stop_loss
        return float(candle["high"]) >= stop_loss

    def _check_same_day_tp3(
        self,
        candle: pd.Series,
        context: dict[str, object],
        direction: str,
    ) -> float | None:
        """Checks if the final target (TP3) was hit on the same candle as entry.

        Args:
            candle: The current price candle.
            context: The parsed signal context.
            direction: Trade direction ("long" or "short").

        Returns:
            float | None: The TP3 exit price if hit, None otherwise.
        """
        take_profit_3 = float(
            context.get("take_profit_3") or context.get("tp3") or 0.0
        )
        if take_profit_3 <= 0:
            return None

        high_price = float(candle["high"])
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if direction == "long" and high_price >= take_profit_3:
            return open_price if open_price > take_profit_3 else take_profit_3

        if direction == "short" and low_price <= take_profit_3:
            return open_price if open_price < take_profit_3 else take_profit_3

        return None


    @override
    def manage_active_trade(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """
        Manages exits: SL, TP1 (Partial), TP3 (Final).
        Implements the 'Gap-Over' fix by checking TP3 before TP1.

        Args:
            trade: The active trade data.
            dataframe_history: Price history.
            repository: The trade repository.

        Returns:
            str | None: Status message if closed/updated, else None.
        """
        if dataframe_history is None or dataframe_history.empty:
            return None

        candle = dataframe_history.iloc[-1]

        # 1. Day-Trading Check
        if not self._is_trade_executable_today(trade, candle):
            return None

        # 2. Extract Context & Market Data
        context = self._get_context(trade)

        # 3. Stop Loss Logic (Priority)
        stop_loss_message = self._check_stop_loss(trade, repository, candle)
        if stop_loss_message:
            return stop_loss_message

        # 4. Target Logic (TP3 -> TP1)
        return self._check_targets(trade, repository, candle, context)

    def _is_trade_executable_today(self, trade: TradeData, candle: pd.Series) -> bool:
        """Checks if the trade can be executed on the current candle date."""
        current_date_obj = pd.Timestamp(candle["date"])
        entry_date_str = trade.get("entry_date")

        if entry_date_str:
            entry_date = pd.Timestamp(entry_date_str)
            if current_date_obj.date() < entry_date.date():
                return False
        return True

    def _check_stop_loss(
        self, trade: TradeData, repository: TradeRepository, candle: pd.Series
    ) -> str | None:
        """Checks and executes Stop Loss logic."""
        stop_loss = float(trade.get("current_stop_loss") or 0.0)
        if stop_loss <= 0:
            return None

        low_price = float(candle["low"])
        high_price = float(candle["high"])
        open_price = float(candle["open"])
        date_string = str(candle["date"])

        direction = str(self._get_context(trade).get("direction", "long")).lower()

        if direction == "long" and low_price <= stop_loss:
            exit_price = stop_loss
            if open_price < stop_loss:
                exit_price = open_price  # Gap down benefit
            return self._close_trade(
                trade, repository, exit_price, ExitReason.STOP_LOSS, date_string
            )

        if direction == "short" and high_price >= stop_loss:
            exit_price = stop_loss
            if open_price > stop_loss:
                exit_price = open_price  # Gap up downside
            return self._close_trade(
                trade, repository, exit_price, ExitReason.STOP_LOSS, date_string
            )

        return None

    def _check_targets(
        self,
        trade: TradeData,
        repository: TradeRepository,
        candle: pd.Series,
        context: dict[str, object],
    ) -> str | None:
        """Evaluates Take Profit levels (TP3 then TP1)."""
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        open_price = float(candle["open"])
        date_string = str(candle["date"])

        take_profit_3 = float(context.get("take_profit_3") or context.get("tp3") or 0.0)
        direction = str(context.get("direction", "long")).lower()

        # Check Final Target (TP3) - GAP OVER FIX
        is_tp3_hit = False
        exit_price = take_profit_3

        if direction == "long" and take_profit_3 > 0 and high_price >= take_profit_3:
            is_tp3_hit = True
            if open_price > take_profit_3:
                exit_price = open_price  # Gap up benefit
        elif direction == "short" and take_profit_3 > 0 and low_price <= take_profit_3:
            is_tp3_hit = True
            if open_price < take_profit_3:
                exit_price = open_price  # Gap down benefit

        if is_tp3_hit:
            return self._close_trade(
                trade, repository, exit_price, ExitReason.TARGET_HIT, date_string
            )

        # Check Partial Target (TP1) if still in Phase 1
        is_phase_2 = bool(context.get("is_phase_2", False))
        take_profit_1 = float(context.get("take_profit_1") or context.get("tp1") or 0.0)

        is_tp1_hit = False
        if not is_phase_2 and take_profit_1 > 0:
            if direction == "long" and high_price >= take_profit_1:
                is_tp1_hit = True
            elif direction == "short" and low_price <= take_profit_1:
                is_tp1_hit = True

        if is_tp1_hit:
            return self._execute_partial_take_profit(
                trade, repository, candle, take_profit_1, context
            )

        return None

    def _execute_partial_take_profit(
        self,
        trade: TradeData,
        repository: TradeRepository,
        candle: pd.Series,
        exit_price_limit: float,
        context: dict[str, object],
    ) -> str:
        """Executes the partial sell logic for TP1."""
        open_price = float(candle["open"])
        direction = str(context.get("direction", "long")).lower()

        exit_price = exit_price_limit
        if direction == "long" and open_price > exit_price:
            exit_price = open_price
        elif direction == "short" and open_price < exit_price:
            exit_price = open_price

        current_size = float(trade.get("current_size") or 0.0)
        quantity_to_sell = int(current_size / 2)
        quantity_remaining = current_size - quantity_to_sell

        if quantity_to_sell <= 0:
            # Too small to split? Close all at TP1
            date_string = str(candle["date"])
            return self._close_trade(
                trade, repository, exit_price, ExitReason.TARGET_HIT, date_string
            )

        entry_price = float(trade.get("entry_price") or 0.0)

        if direction == "short":
            profit_and_loss_chunk = (entry_price - exit_price) * quantity_to_sell
        else:
            profit_and_loss_chunk = (exit_price - entry_price) * quantity_to_sell

        existing_pnl = float(trade.get("realized_pnl") or 0.0)
        new_total_pnl = existing_pnl + profit_and_loss_chunk

        # Move SL to Entry (Break Even)
        new_stop_loss = entry_price

        # Update Context
        context["is_phase_2"] = True

        repository.update_trade(
            trade["id"],
            {
                "current_size": quantity_remaining,
                "current_stop_loss": new_stop_loss,
                "realized_pnl": new_total_pnl,
                "signal_context": json.dumps(context, default=str, ensure_ascii=False),
            },
            reason=(
                f"TP1 HIT @ {exit_price:.2f}. "
                f"Sold {int(quantity_to_sell)}. SL -> {new_stop_loss:.2f}."
            ),
        )

        return (
            f"TP1 HIT @ {exit_price:.2f}. "
            f"Partial Sell {int(quantity_to_sell)}. SL -> BE."
        )

    @override
    def generate_orders(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        repository: TradeRepository,
    ) -> Order | None:
        """
        Generates a Multi-Leg Bracket Order.

        Args:
            trade: The trade data.
            dataframe_history: Price history.
            budget: Available budget.
            repository: Trade repository.

        Returns:
            Order | None: The generated order or None.
        """
        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = float(trade.get("entry_price") or 0.0)
        stop_loss = float(trade.get("current_stop_loss") or 0.0)

        # Get Context for TPs (handle both old and new names)
        context = self._get_context(trade)
        take_profit_1 = float(context.get("take_profit_1") or context.get("tp1") or 0.0)
        take_profit_3 = float(context.get("take_profit_3") or context.get("tp3") or 0.0)

        if entry_price <= 0:
            return None

        # 1. Centralized Quantity Calculation
        database_size = float(trade.get("initial_size") or 0.0)
        if database_size > 0:
            quantity = int(database_size)
        else:
            risk_amount = float(trade.get("risk_amount") or 100.0)
            quantity = self._calculate_position_size(
                entry_price, stop_loss, risk_amount
            )

        if quantity <= 0:
            return None

        direction = str(context.get("direction", "long")).lower()
        entry_action = "SELL" if direction == "short" else "BUY"
        exit_action = "BUY" if direction == "short" else "SELL"

        # 2. Entry Leg
        entry_leg = OrderLeg(
            action=entry_action,
            type="STP",
            price=entry_price,
            quantity=quantity,
            time_in_force="DAY",
        )

        exits = []

        # 3. Exit 1: Stop Loss (Full Quantity)
        if stop_loss > 0:
            exits.append(
                OrderLeg(
                    action=exit_action,
                    type="STP",
                    price=stop_loss,
                    quantity=quantity,
                    time_in_force="GTC",
                )
            )

        # Define half/remaining quantities unconditionally to prevent NameError
        # when take_profit_3 is set but take_profit_1 is zero (SEC-03 fix).
        quantity_half = int(quantity / 2) if take_profit_1 > 0 else 0
        quantity_remaining = quantity - quantity_half

        # 4. Exit 2: TP1 (50%)
        if take_profit_1 > 0 and quantity_half > 0:
            exits.append(
                OrderLeg(
                    action=exit_action,
                    type="LMT",
                    price=take_profit_1,
                    quantity=quantity_half,
                    time_in_force="GTC",
                )
            )

        # 5. Exit 3: TP3 (Remaining)
        if take_profit_3 > 0 and quantity_remaining > 0:
            exits.append(
                OrderLeg(
                    action=exit_action,
                    type="LMT",
                    price=take_profit_3,
                    quantity=quantity_remaining,
                    time_in_force="GTC",
                )
            )

        return Order(
            id=f"{symbol}_{self.name}",
            symbol=symbol,
            quantity=quantity,
            mode="BRACKET_MULTI",
            entry=entry_leg,
            exits=exits,
            last_status="CREATED",
        )

    # --- Private Helpers ---

    def _get_context(self, trade: TradeData) -> dict[str, object]:
        """
        Safe helper to load signal context.

        Args:
            trade: The trade dictionary.

        Returns:
            dict[str, object]: The context dictionary or empty.
        """
        try:
            return json.loads(trade.get("signal_context") or "{}")
        except json.JSONDecodeError as parsing_error:
            logger.warning(
                "Failed to parse signal context for trade %s: %s",
                trade.get("id"),
                parsing_error,
            )
            return {}

    def _execute_immediate_target(
        self,
        trade: TradeData,
        repository: TradeRepository,
        fill_price: float,
        reason: str,
        exit_price: float,
        stop_loss: float,
        date_string: str,
        context: dict[str, object],
    ) -> str:
        """Handles Day 1 massive targets: entry and full target hit on same day.

        Delegates position sizing to the centralized _resolve_position_size
        method to maintain a single source of truth.

        Args:
            trade: The trade dictionary.
            repository: The trade repository.
            fill_price: The entry fill price.
            reason: The entry reason.
            exit_price: The TP3 exit price.
            stop_loss: The stop loss price.
            date_string: The date of the trade.
            context: The parsed signal context.

        Returns:
            str: Status message.
        """
        size = self._resolve_position_size(trade, fill_price, stop_loss)

        if size <= 0:
            return "ERROR: Zero Size"

        direction = str(context.get("direction", "long")).lower()
        if direction == "short":
            profit_and_loss = (fill_price - exit_price) * size
        else:
            profit_and_loss = (exit_price - fill_price) * size

        repository.update_trade(
            trade["id"],
            {
                "status": TradeStatus.CLOSED,
                "entry_date": date_string,
                "entry_price": fill_price,
                "initial_size": size,
                "current_size": 0,
                "exit_date": date_string,
                "exit_price": exit_price,
                "exit_reason": ExitReason.TARGET_HIT,
                "realized_pnl": profit_and_loss,
            },
            reason=f"{reason} FILLED @ {fill_price:.2f} -> TARGET HIT @ {exit_price:.2f}",
        )
        return (
            f"FILLED @ {fill_price:.2f} -> TARGET HIT @ {exit_price:.2f} "
            f"(PnL: {profit_and_loss:.2f})"
        )

