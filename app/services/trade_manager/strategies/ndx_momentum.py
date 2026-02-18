import logging
from typing import override, final
import pandas as pd

from ....types import TradeStatus, TradeData
from ....const import Strategies
from ....models import TradeParams, Order, OrderLeg
from ....database.repositories.trade import TradeRepository
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)

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
    DEFAULT_BUDGET: float = 2000.0

    @override
    def get_current_params(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame | None = None, 
        repository: TradeRepository | None = None
    ) -> TradeParams:
        """
        Extracts current strategy parameters for display and evaluation.
        
        Args:
            trade: The trade data to extract parameters from.
            dataframe_history: Optional historical price data.
            repository: Optional repository for database access.
            
        Returns:
            A TradeParams object containing the current strategy settings.
        """
        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "momentum_score": self._get_context_value(trade, "momentum_score"),
                "qqq_regime": self._get_context_value(trade, "qqq_regime"),
                "regime": self._get_context_value(trade, "regime"),
                "signal_date": self._get_context_value(trade, "date")
            }
        )

    @override
    def check_entry(
        self, 
        trade: TradeData, 
        candle: pd.Series, 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Evaluates and executes entry for a candidate trade.
        
        This method checks for market regime favorability and ensures no
        duplicate positions exist before activating a trade at the current
        market open.
        
        Args:
            trade: The candidate trade data.
            candle: The current daily price candle.
            dataframe_history: The historical data leading up to the target date.
            repository: The repository for trade state persistence.
            
        Returns:
            A string describing the activation status or None if no entry occurred.
        """
        # 1. Regime Check (Combined signal from screener)
        current_regime = self._get_context_value(trade, "regime")
        symbol = trade.get("symbol")
        
        if current_regime != "BULL":
            return self._reject_setup(
                trade, repository, str(candle["date"]), f"Combined Regime: {current_regime}"
            )

        # 2. Duplicate Check (Active positions in the same strategy)
        active_positions = repository.get_by_status(TradeStatus.ACTIVE)
        for active_trade in active_positions:
            if active_trade["symbol"] == symbol and active_trade["strategy"] == self.name:
                return self._reject_setup(
                    trade, repository, str(candle["date"]), "Position already exists"
                )

        # 3. Execution (Market On Open)
        date_string = str(candle["date"])
        fill_price = float(candle["open"])
        
        return self._execute_activation(
            trade, repository, fill_price, "REBAL_ENTRY", date_string
        )

    @override
    def manage_active_trade(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame, 
        repository: TradeRepository
    ) -> str | None:
        """
        Manages an active position during the monthly rebalance cycle.
        
        This method determines if a position should be maintained or closed
        based on the latest leaders identified on the month-end screener.
        Rebalancing only occurs on the transition to a new month.
        
        Args:
            trade: The active trade data.
            dataframe_history: Historical price data.
            repository: Repository for trade lookups.
            
        Returns:
            A status string if the trade was closed, otherwise None.
        """
        if len(dataframe_history) < 2:
            return None
            
        current_candle = dataframe_history.iloc[-1]
        previous_candle = dataframe_history.iloc[-2]
        current_date = pd.Timestamp(current_candle["date"])
        previous_date = pd.Timestamp(previous_candle["date"])
        
        # Rule: Only rebalance on month-switch (First Trading Day)
        if current_date.month == previous_date.month and current_date.year == previous_date.year:
            return None

        date_string = str(current_candle["date"])
        symbol = trade.get("symbol")
        
        # 1. Fetch latest "Rebalance Winners" from the trades table
        all_strategy_trades = repository.get_all_by_strategy(self.name)
        if not all_strategy_trades:
            return None
            
        # Group by context date to find the latest rebalance batch
        latest_signal_date = "0000-00-00"
        leaders_symbols = set()
        
        for historical_trade in all_strategy_trades:
            signal_context = historical_trade.get("signal_context", {})
            context_date_value = signal_context.get("date", "0000-00-00")
            if context_date_value > latest_signal_date:
                latest_signal_date = context_date_value
                leaders_symbols = {historical_trade["symbol"]}
            elif context_date_value == latest_signal_date:
                leaders_symbols.add(historical_trade["symbol"])
        
        # 2. Check if current trade belongs to the latest batch
        current_signal_context = trade.get("signal_context", {})
        current_signal_date = current_signal_context.get("date", "0000-00-00")
        
        if current_signal_date < latest_signal_date or symbol not in leaders_symbols:
            logger.info("[%s] Symbol %s dropped in rebalance (Current:%s < Latest:%s)", 
                        self.name, symbol, current_signal_date, latest_signal_date)
            exit_price = float(current_candle["open"])
            return self._close_trade(
                trade, repository, exit_price, "REBAL_EXIT", date_string
            )
            
        return None

    @override
    def generate_orders(
        self, 
        trade: TradeData, 
        dataframe_history: pd.DataFrame, 
        budget: float, 
        repository: TradeRepository
    ) -> Order | None:
        """
        Generates formal trading orders for a given trade.
        
        Args:
            trade: The trade data to generate orders for.
            dataframe_history: Historical price data.
            budget: Total allocated budget to use for sizing.
            repository: Repository instance.
            
        Returns:
            An Order object or None if generation failed.
        """
        symbol = trade.get("symbol", "UNKNOWN")
        
        # Determine specific budget for this position
        trade_budget = float(trade.get("budget") or budget or self.DEFAULT_BUDGET)
        
        # Calculate quantity based on the most recent closing price
        if dataframe_history.empty:
            return None
            
        last_closing_price = float(dataframe_history.iloc[-1]["close"])
        if last_closing_price <= 0:
            return None
            
        shares_quantity = int(trade_budget / last_closing_price)
        if shares_quantity <= 0:
            return None
            
        entry_leg_definition = OrderLeg(
            action="BUY",
            type="MKT",
            price=0.0,
            quantity=shares_quantity,
            tif="OPG"
        )
        
        return Order(
            id=f"{symbol}_{self.name}",
            symbol=symbol,
            quantity=shares_quantity,
            mode="SIMPLE",
            entry=entry_leg_definition,
            exits=[],
            last_status="CREATED"
        )
