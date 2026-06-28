import logging
from decimal import Decimal
from typing import override, final
import pandas as pd

from ..types import TradeTransition
from ....types import ExitReason, TradeData
from ....const import Strategies
from ....models import TradeParams, Order, OrderLeg
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


@final
class DipBuyerStrategy(BaseTradeStrategy):
    """Dip Buyer Strategy: Enters on weakness via Limit Order.

    Attributes:
        name: Strategy identifier.
        DEFAULT_BUDGET: Default capital allocation for the trade.
        TIME_STOP_DAYS: Maximum holding period in trading days.
    """

    name: Strategies = Strategies.DipBuyer
    TIME_STOP_DAYS: int = 8

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams | None:
        """Extracts current strategy parameters for display.

        Args:
            trade: Current trade data.
            dataframe_history: Historical price data.

        Returns:
            TradeParams: Extracted parameters for UI display.
        """
        return TradeParams(
            stop_loss=0.0,  # No stop loss
            take_profit_1=float(trade.get("current_target") or 0.0),
            extras={
                "entry_limit": self._extract_entry_price(trade),
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
        """Checks if the limit entry was reached.

        Rules:
        1. Skip signal day (Day 0).
        2. Entry on Next Day (Day 1) via Limit Order.
        3. Invalidate if Day > 1.

        Args:
            trade: Current trade record.
            candle: Latest market candle.
            dataframe_history: Historical data for context.
            active_symbols: Active symbols set.

        Returns:
            TradeTransition | None: Description of transition if triggered.
        """
        if active_symbols and trade.get("symbol") in active_symbols:
            return self._reject_setup(
                trade, str(candle["date"]), "Symbol already active"
            )

        limit_price = self._extract_entry_price(trade)
        if limit_price <= 0:
            return None

        # 1. Date/Session Validation
        # Rules: Skip signal day (Day 0), Entry on Next Day (Day 1), Invalidate if Day > 1
        days_passed = self._get_trading_days_post_signal(trade, dataframe_history)

        if days_passed == 0:
            # Too early (Signal Day)
            return None

        date_string = str(candle["date"])
        if days_passed > 1:
            # Too late: Missed the entry window
            return self._reject_setup(trade, date_string, "Missed Entry Window")

        # 2. Check Fill (Day 1)
        low_price = float(candle["low"])
        open_price = float(candle["open"])

        if low_price > limit_price:
            # Missed entry on the target day -> Invalidate the setup
            return self._invalidate_trade(trade, low_price, limit_price, date_string)

        # 3. Execution (with Gap Down benefit)
        fill_price = (
            min(open_price, limit_price) if open_price < limit_price else limit_price
        )

        return self._execute_activation(trade, fill_price, "LIMIT", date_string)

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages exits: LOC, Target, and Time Stop.

        Exits:
        1. LOC (Limit On Close) - If Close > Previous Day High.
        2. Target (Take Profit) - predefined target hit.
        3. Time Stop - Closed at end of day 8.
        """
        current_date_obj = pd.Timestamp(current_candle["date"])

        # 1. Day Check
        entry_date_str = trade.get("entry_date")
        is_entry_day = False
        if entry_date_str:
            entry_date = pd.Timestamp(entry_date_str).date()
            if current_date_obj.date() < entry_date:
                return None
            is_entry_day = current_date_obj.date() == entry_date

        # 2. Target Logic (Take Profit)
        # Rule: Target can NOT be hit on entry day.
        target_price = float(trade.get("current_target") or 0.0)
        high_price = float(current_candle["high"])
        open_price = float(current_candle["open"])

        if not is_entry_day and target_price > 0 and high_price >= target_price:
            exit_price = max(open_price, target_price)
            return self._close_trade(
                trade, exit_price, ExitReason.TARGET_HIT, date_string
            )

        # 3. LOC (Limit On Close) Logic
        # Rule: Only Limit on Close is possible for same day.
        if len(dataframe_history) >= 2:
            previous_candle = dataframe_history.iloc[-2]
            previous_day_high = float(previous_candle["high"])
            close_price = float(current_candle["close"])

            if close_price > previous_day_high:
                return self._close_trade(trade, close_price, "LOC_HIT", date_string)

        entry_date_str = trade.get("entry_date")
        if entry_date_str:
            # Count trading days since entry
            # User Request: Close AT the 8th trading day
            trading_days_held = len(
                dataframe_history[dataframe_history["date"] >= entry_date_str]
            )
            if trading_days_held >= self.TIME_STOP_DAYS:
                close_price = float(current_candle["close"])
                return self._close_trade(
                    trade,
                    close_price,
                    ExitReason.TIME_STOP,
                    date_string,
                )

        return None

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """Calculates dynamic daily updates, primarily the LOC threshold.

        Args:
            trade: The current trade data.
            dataframe_history: Market history for the symbol.

        Returns:
            dict[str, object]: Updates to be merged into signal_context.
        """
        updates: dict[str, object] = {}
        threshold_loc = self._calculate_threshold_loc(dataframe_history)
        if threshold_loc is not None:
            updates["threshold_loc"] = threshold_loc

        return updates

    def _determine_order_quantity(self, trade: TradeData, budget: float) -> int:
        """Calculates order quantity based on database size or strategy budget."""
        entry_price = self._extract_entry_price(trade)
        if entry_price <= 0:
            return 0

        database_size = float(trade.get("initial_size") or 0.0)
        if database_size > 0:
            return int(database_size)

        trade_budget = self._get_strategy_budget(trade, budget)
        if trade_budget <= 0:
            logger.warning(
                "[%s] Sizing Fallback: No budget found. Check settings.yaml.",
                trade.get("symbol"),
            )
            return 0
        return int(trade_budget / entry_price)

    def _calculate_threshold_loc(self, dataframe_history: pd.DataFrame) -> float | None:
        """Calculates the dynamic LOC threshold based on the previous day's high."""
        if len(dataframe_history) < 1:
            return None
        previous_candle = dataframe_history.iloc[-1]
        previous_day_high = float(previous_candle["high"])
        return round(previous_day_high + 0.01, 2)

    @override
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
    ) -> Order | None:
        """Generates entry bracket orders for CREATED trades."""
        quantity = self._determine_order_quantity(trade, budget)
        if quantity <= 0:
            return None
        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = self._extract_entry_price(trade)

        entry_leg = OrderLeg(
            action="BUY",
            type="LMT",
            price=Decimal(str(entry_price)),
            quantity=quantity,
            time_in_force="DAY",
        )

        exits = []
        context = self._get_full_context(trade)
        threshold_loc = context.get("threshold_loc")
        if threshold_loc is None:
            threshold_loc = self._calculate_threshold_loc(dataframe_history)
        exit_price = threshold_loc if threshold_loc else trade.get("current_target")
        if exit_price:
            exit_value = float(exit_price)
            if exit_value > 0:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LOC",
                        price=Decimal(str(exit_value)),
                        quantity=quantity,
                        time_in_force="DAY",
                    )
                )

        return self._create_order(
            symbol=symbol,
            quantity=quantity,
            mode="BRACKET",
            entry=entry_leg,
            exits=exits,
            order_id=f"{symbol}_{self.name}",
        )

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
    ) -> Order | None:
        """Generates exit orders for ACTIVE trades (Take Profit and EOD LOC exit)."""
        quantity = self._determine_order_quantity(trade, budget)
        if quantity <= 0:
            return None
        symbol = trade.get("symbol", "UNKNOWN")
        exits = []

        # 1. Take Profit Exit Order
        target_price = float(trade.get("current_target") or 0.0)
        if target_price > 0:
            exits.append(
                OrderLeg(
                    action="SELL",
                    type="LMT",
                    price=Decimal(str(target_price)),
                    quantity=quantity,
                    time_in_force="DAY",
                )
            )

        # 2. Limit On Close (LOC) Exit at daily updated threshold (Vortages-Hoch)
        context = self._get_full_context(trade)
        threshold_loc = context.get("threshold_loc")
        if threshold_loc is None:
            threshold_loc = self._calculate_threshold_loc(dataframe_history)
        if threshold_loc:
            threshold_value = float(threshold_loc)
            if threshold_value > 0:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LOC",
                        price=Decimal(str(threshold_value)),
                        quantity=quantity,
                        time_in_force="DAY",
                    )
                )

        if not exits:
            return None

        return self._create_order(
            symbol=symbol,
            quantity=quantity,
            mode="Exit",
            entry=None,
            exits=exits,
            order_id=f"{symbol}_{self.name}_EXIT",
        )
