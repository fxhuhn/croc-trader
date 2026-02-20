import json
import logging
from typing import override, final


import pandas as pd

from ....types import EntryReason, ExitReason, TradeData
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
        """
        Checks for a breakout entry (Stop Buy) and validates the stop loss.

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

        current_date_obj = pd.Timestamp(candle["date"])
        date_string = str(candle["date"])

        # 1. Signal Date validation
        signal_date = self._get_signal_date(trade)
        if signal_date:
            if current_date_obj.date() <= signal_date.date():
                return None
            if (current_date_obj.date() - signal_date.date()).days > 5:
                # Expire the setup if not triggered
                return self._expire_trade(trade, repository, date_string)

        # 2. Market Data
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        # 3. Entry Logic
        filled, fill_price, reason = False, 0.0, ""

        if open_price >= entry_price:
            filled, fill_price, reason = True, open_price, EntryReason.GAP_UP
        elif high_price >= entry_price:
            filled, fill_price, reason = True, entry_price, EntryReason.BREAKOUT

        # 4. Stop Loss / Invalidation
        is_stop_hit = stop_loss > 0 and low_price <= stop_loss

        if filled:
            if is_stop_hit:
                # Immediate Loss (Day 1 turnaround)
                return self._execute_immediate_loss(
                    trade, repository, fill_price, reason, stop_loss, date_string
                )
            else:
                return self._execute_activation(
                    trade, repository, fill_price, reason, date_string
                )

        if is_stop_hit:
            return self._invalidate_trade(
                trade, repository, low_price, stop_loss, date_string
            )

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
        low_price = float(candle["low"])
        open_price = float(candle["open"])
        date_string = str(candle["date"])

        if stop_loss > 0 and low_price <= stop_loss:
            exit_price = stop_loss
            if open_price < stop_loss:
                exit_price = open_price  # Gap down

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
        open_price = float(candle["open"])
        date_string = str(candle["date"])

        take_profit_3 = float(context.get("take_profit_3") or context.get("tp3") or 0.0)

        # Check Final Target (TP3) - GAP OVER FIX
        if take_profit_3 > 0 and high_price >= take_profit_3:
            exit_price = take_profit_3
            if open_price > take_profit_3:
                exit_price = open_price  # Gap up benefit

            return self._close_trade(
                trade, repository, exit_price, ExitReason.TARGET_HIT, date_string
            )

        # Check Partial Target (TP1) if still in Phase 1
        is_phase_2 = bool(context.get("is_phase_2", False))
        take_profit_1 = float(context.get("take_profit_1") or context.get("tp1") or 0.0)

        if not is_phase_2 and take_profit_1 > 0:
            if high_price >= take_profit_1:
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

        exit_price = exit_price_limit
        if open_price > exit_price:
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
                "signal_context": json.dumps(context),
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

        # 2. Entry Leg
        entry_leg = OrderLeg(
            action="BUY",
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
                    action="SELL",
                    type="STP",
                    price=stop_loss,
                    quantity=quantity,
                    time_in_force="GTC",
                )
            )

        # 4. Exit 2: TP1 (50%)
        if take_profit_1 > 0:
            quantity_half = int(quantity / 2)
            if quantity_half > 0:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LMT",
                        price=take_profit_1,
                        quantity=quantity_half,
                        time_in_force="GTC",
                    )
                )

        # 5. Exit 3: TP3 (Remaining)
        if take_profit_3 > 0:
            quantity_rem = quantity - int(quantity / 2)
            if quantity_rem > 0:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LMT",
                        price=take_profit_3,
                        quantity=quantity_rem,
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
        except Exception:
            return {}
