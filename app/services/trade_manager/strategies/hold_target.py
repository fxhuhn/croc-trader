import logging
from decimal import Decimal
from typing import override, final

import pandas as pd

from ..types import TradeTransition
from ....types import EntryReason, ExitReason, TradeData
from ....models import TradeParams, Order, OrderLeg
from .abstract import BaseTradeStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)


@final
class HoldTargetStrategy(BaseTradeStrategy):
    """
    Manager for Croc Breakouts (Hold/TP3).
    Aims to ride large trends (3R or more) with a breakout entry.
    Inherits shared logic from BaseTradeStrategy.
    """

    name = Strategies.HoldTarget

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams:
        """
        Calculates current strategy parameters for display/logging.

        Args:
            trade: The trade data structure.
            dataframe_history: Historical price data (unused here).

        Returns:
            TradeParams: Container with stop loss, target, and extras.
        """
        # Note: realized_pnl is kept as is per user requirement
        return TradeParams(
            stop_loss=float(trade.get("current_stop_loss") or 0.0),
            take_profit_1=float(trade.get("current_target") or 0.0),
            extras={
                "entry_limit": float(trade.get("entry_price") or 0.0),
                "current_size": float(trade.get("current_size") or 0.0),
            },
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """
        Checks for a breakout entry (Stop Buy) and validates the stop loss.

        Args:
            trade: The trade data from the database.
            candle: The current price candle.
            dataframe_history: Historical price data.
            active_symbols: Currently active symbols set.

        Returns:
            TradeTransition | None: A transition if filled or invalidated, otherwise None.
        """
        entry_price = float(trade.get("entry_price") or 0.0)
        stop_loss = float(trade.get("current_stop_loss") or 0.0)

        if entry_price <= 0:
            return None

        current_date_obj = pd.Timestamp(candle["date"])
        date_string = str(candle["date"])

        # 1. Signal Date Validation
        # Signal date is used to check entry timing
        signal_date = self._get_signal_date(trade)
        if signal_date:
            # Entry must occur STRICTLY AFTER the signal date
            if current_date_obj.date() <= signal_date.date():
                return None

            # Expiration Check (5 Calendar Days)
            if (current_date_obj.date() - signal_date.date()).days > 5:
                return self._expire_trade(trade, date_string)

        # 2. Market Data
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        # 3. Entry Logic (Stop Buy)
        filled, fill_price, reason = False, 0.0, ""

        if open_price >= entry_price:
            # Gapped over Entry
            filled, fill_price, reason = True, open_price, EntryReason.GAP_UP
        elif high_price >= entry_price:
            # Intraday touched Entry
            filled, fill_price, reason = True, entry_price, EntryReason.BREAKOUT

        # 4. Stop Loss / Invalidation Check
        is_stop_hit = stop_loss > 0 and low_price <= stop_loss

        if filled:
            if is_stop_hit:
                # Day 1 Turnaround: Filled then Stopped on same day
                return self._execute_immediate_loss(
                    trade, fill_price, reason, stop_loss, date_string
                )
            return self._execute_activation(trade, fill_price, reason, date_string)

        if is_stop_hit:
            # Setup Invalidated (Stop hit before Entry)
            return self._invalidate_trade(trade, low_price, stop_loss, date_string)

        return None

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """
        Manages exits for active trades: Stop Loss vs. Target.
        """
        current_date_obj = pd.Timestamp(current_candle["date"])

        # 1. Day-Trading Check (Allow same-day exit)
        entry_date_str = trade.get("entry_date")
        if entry_date_str:
            entry_date = pd.Timestamp(entry_date_str)
            # Only block if date is strictly BEFORE entry date (sanity check)
            if current_date_obj.date() < entry_date.date():
                return None

        # 2. Current Parameters
        stop_loss = float(trade.get("current_stop_loss") or 0.0)
        target = float(trade.get("current_target") or 0.0)

        # 3. Market Data
        low_price = float(current_candle["low"])
        high_price = float(current_candle["high"])
        open_price = float(current_candle["open"])

        # 4. Stop Loss Logic (Check first)
        if stop_loss > 0 and low_price <= stop_loss:
            exit_price = stop_loss
            if open_price < stop_loss:
                exit_price = open_price  # Gap down execution

            return self._close_trade(
                trade, exit_price, ExitReason.STOP_LOSS, date_string
            )

        # 5. Target Logic
        if target > 0 and high_price >= target:
            exit_price = target
            if open_price > target:
                exit_price = open_price  # Gap up execution (Benefit)

            return self._close_trade(
                trade, exit_price, ExitReason.TARGET_HIT, date_string
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
        """
        Generates an Order object for IBKR export.
        """
        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = float(trade.get("entry_price") or 0.0)
        stop_loss = float(trade.get("current_stop_loss") or 0.0)

        if entry_price <= 0:
            logger.warning("[%s] Invalid entry price: %s", symbol, entry_price)
            return None

        # 1. Centralized Quantity Calculation
        database_size = float(trade.get("initial_size") or 0.0)
        if database_size > 0:
            quantity = int(database_size)
        else:
            quantity = self._resolve_position_size(trade, entry_price, stop_loss)

        if quantity <= 0:
            logger.warning(
                "[%s] Calculated quantity is 0 or invalid stop loss setup.", symbol
            )
            return None

        # 2. Entry: STOP BUY
        entry_leg = OrderLeg(
            action="BUY",
            type="STP",
            price=Decimal(str(entry_price)),
            quantity=quantity,
            time_in_force="DAY",
        )

        exits = []

        # 3. Exit 1: Stop Loss (Mandatory)
        if stop_loss > 0:
            exits.append(
                OrderLeg(
                    action="SELL",
                    type="STP",
                    price=Decimal(str(stop_loss)),
                    quantity=quantity,
                    time_in_force="GTC",
                )
            )
        else:
            logger.warning("[%s] Missing stop loss for HoldTarget order.", symbol)
            return None

        # 4. Exit 2: Target (Optional)
        target_price = float(trade.get("current_target") or 0.0)
        if target_price > 0:
            exits.append(
                OrderLeg(
                    action="SELL",
                    type="LMT",
                    price=Decimal(str(target_price)),
                    quantity=quantity,
                    time_in_force="GTC",
                )
            )

        return Order(
            id=f"{symbol}_{self.name}",
            symbol=symbol,
            quantity=quantity,
            mode="BRACKET",
            entry=entry_leg,
            exits=exits,
            last_status="CREATED",
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
        return None
