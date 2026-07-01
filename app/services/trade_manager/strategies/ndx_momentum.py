import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import override, final
import pandas as pd

from ..types import TradeTransition
from ....types import TradeData
from ....const import Strategies
from ....models import TradeParams, Order
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


@dataclass
class _RebalanceCache:
    """Typed cache for the monthly rebalance leader lookup.

    Avoids O(N²) DB queries during a full portfolio rebalance by caching
    the leaders set for a given date key.
    """

    cache_key: str
    latest_signal_date: str
    leaders_symbols: set[str] = field(default_factory=set)


@final
class NDXMomentumTradeStrategy(BaseTradeStrategy):
    """
    NDX Momentum Strategy implementation for monthly rebalancing.

    This strategy manages symbols selected by the NDX Momentum Screener.
    It follows a strict month-start rebalancing rule, entering top performers
    if market conditions are favorable and exiting symbols that fall out of
    the leaders list.

    Attributes:
        DEFAULT_BUDGET: The standard dollar amount allocated per position.
    """

    name = Strategies.NDXMomentum
    _rebalance_cache: _RebalanceCache | None = None

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams:
        """
        Extracts current strategy parameters for display and evaluation.

        Args:
            trade: The trade data to extract parameters from.
            dataframe_history: Optional historical price data.

        Returns:
            A TradeParams object containing the current strategy settings.
        """
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "momentum_score": self._get_context_value(trade, "momentum_score"),
                "qqq_regime": self._get_context_value(trade, "qqq_regime"),
                "regime": f"{self._get_context_value(trade, 'regime')} (UNUSED)",
                "signal_date": self._get_context_value(trade, "date"),
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
        Evaluates and executes entry for a candidate trade.

        This method checks for market regime favorability (QQQ SMA Filter)
        and ensures no duplicate positions exist before activating a trade
        at the current market open.

        Note: The breadth-based combined regime is currently ignored.

        Args:
            trade: The candidate trade data.
            candle: The current daily price candle.
            dataframe_history: The historical data leading up to the target date.
            active_symbols: Set of symbols with currently active positions.

        Returns:
            TradeTransition | None: Computed transition if entry occurred.
        """
        # 1. Regime Check (Using QQQ SMA Filter, ignoring Breadth)
        qqq_regime = self._get_context_value(trade, "qqq_regime")
        symbol = trade.get("symbol")

        if qqq_regime != "BULL":
            return self._reject_setup(
                trade, str(candle["date"]), f"QQQ Regime: {qqq_regime}"
            )

        # 2. Duplicate Check (Active positions in the same strategy)
        if active_symbols and symbol in active_symbols:
            return self._reject_setup(
                trade, str(candle["date"]), "Position already exists"
            )

        # 3. Execution (Market On Open)
        # We ensure at least one trading day has passed since the signal
        # to avoid look-ahead bias (entering on the same day as the screen).
        trading_days_passed = self._get_trading_days_post_signal(
            trade, dataframe_history
        )
        if trading_days_passed < 1:
            return None

        date_string = str(candle["date"])
        fill_price = float(candle["open"])

        return self._execute_activation(
            trade, fill_price, "REBALANCE_ENTRY", date_string
        )

    def _is_month_switch(self, dataframe_history: pd.DataFrame) -> bool:
        """Determines if the current candle represents a month transition.

        Args:
            dataframe_history: Historical price data.

        Returns:
            bool: True if the current date is in a different month/year
                than the previous date.
        """
        if len(dataframe_history) < 2:
            return False

        current_candle = dataframe_history.iloc[-1]
        previous_candle = dataframe_history.iloc[-2]
        current_date = pd.Timestamp(current_candle["date"])
        previous_date = pd.Timestamp(previous_candle["date"])

        return (
            current_date.month != previous_date.month
            or current_date.year != previous_date.year
        )

    def _is_month_switch_order(
        self, dataframe_history: pd.DataFrame, reference_date: str | None
    ) -> bool:
        """Determines if there is a month switch between last historical date and target date.

        Used in order generation to correctly identify a month switch before the new month's
        first daily candle has been recorded in the database.
        """
        if dataframe_history.empty:
            return False

        last_candle_date_str = str(dataframe_history.iloc[-1]["date"])
        try:
            last_date = pd.Timestamp(last_candle_date_str)
        except (ValueError, TypeError):
            return False

        if reference_date:
            try:
                ref_date = pd.Timestamp(reference_date)
                return (
                    ref_date.month != last_date.month or ref_date.year != last_date.year
                )
            except (ValueError, TypeError):
                pass

        # Fallback to the standard candle-to-candle comparison
        return self._is_month_switch(dataframe_history)

    @staticmethod
    def extract_latest_leaders(
        all_strategy_trades: list[dict[str, object]],
    ) -> set[str]:
        """Pure helper to extract the latest signal date's leaders from strategy trades."""
        latest_signal_date = "0000-00-00"
        leaders_symbols: set[str] = set()

        for trade in all_strategy_trades:
            context_data = trade.get("signal_context")
            date_value = None
            if context_data:
                try:
                    if isinstance(context_data, dict):
                        date_value = context_data.get("date")
                    else:
                        date_value = json.loads(context_data).get("date")
                except (json.JSONDecodeError, TypeError) as parse_error:
                    logger.warning(
                        "Failed to parse signal_context for trade %s: %s",
                        trade.get("id"),
                        parse_error,
                    )
            date_str = str(date_value) if date_value else "0000-00-00"
            symbol_str = str(trade.get("symbol", ""))
            if not symbol_str:
                continue
            if date_str > latest_signal_date:
                latest_signal_date = date_str
                leaders_symbols = {symbol_str}
            elif date_str == latest_signal_date:
                leaders_symbols.add(symbol_str)
        return leaders_symbols

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
        Manages an active position during the monthly rebalance cycle.
        """
        if not self._is_month_switch(dataframe_history):
            return None

        symbol = trade.get("symbol")

        leaders_symbols = latest_leaders or set()

        # 2. Check if current trade symbol is still in the latest batch of leaders
        # We only exit if the symbol is NOT in the latest winners list.
        # This allows positions to persist across months without unnecessary turnover.
        if symbol not in leaders_symbols:
            logger.info(
                "[%s] Symbol %s dropped from leaders list in rebalance",
                self.name,
                symbol,
            )
            exit_price = float(current_candle["open"])
            return self._close_trade(trade, exit_price, "REBALANCE_EXIT", date_string)

        return None

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        symbol = trade.get("symbol", "UNKNOWN")

        if not self._is_month_switch_order(dataframe_history, reference_date):
            return None

        leaders_symbols = created_symbols or set()

        if symbol not in leaders_symbols:
            quantity = int(trade.get("current_size") or 0)
            if quantity <= 0:
                return None

            return self._create_exit_order(
                symbol=symbol,
                quantity=quantity,
                price=Decimal("0.0"),
                order_type="MKT",
                time_in_force="OPG",
                order_id=f"{symbol}_{self.name}_EXIT",
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
        symbol = trade.get("symbol", "UNKNOWN")

        # Determine specific budget for this position
        trade_budget = self._get_strategy_budget(trade, budget)

        # Calculate quantity based on the most recent closing price
        if dataframe_history.empty:
            return None

        last_closing_price = float(dataframe_history.iloc[-1]["close"])
        if last_closing_price <= 0:
            return None

        shares_quantity = int(trade_budget / last_closing_price)
        if shares_quantity <= 0:
            return None

        return self._create_entry_order(
            symbol=symbol,
            quantity=shares_quantity,
            entry_price=Decimal("0.0"),
            order_type="MKT",
            time_in_force="OPG",
            order_id=f"{symbol}_{self.name}",
        )
