import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ....database.repositories.trade import TradeRepository
from ....types import TradeStatus, ExitReason
from ....models import Order, TradeParams

logger = logging.getLogger(__name__)


class BaseTradeStrategy(ABC):
    """
    Interface and Base Class for all trading strategies.
    Provides shared logic for trade activation, exit management, and position sizing.
    """

    DEFAULT_BUDGET: float = 2000.0

    @abstractmethod
    def check_entry(
        self,
        trade: dict[str, Any],
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """
        Checks for CREATED trades if the entry was filled.

        Args:
            trade: The trade data from the database.
            candle: The current price candle.
            dataframe_history: Historical price data.
            repository: The trade repository for database updates.

        Returns:
            str | None: A status message if filled/invalidated, else None.
        """
        raise NotImplementedError("Subclasses must implement check_entry")

    @abstractmethod
    def manage_active_trade(
        self,
        trade: dict[str, Any],
        dataframe_history: pd.DataFrame,
        repository: TradeRepository,
    ) -> str | None:
        """
        Manages ACTIVE trades (Exit checks).

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data.
            repository: The trade repository for database updates.

        Returns:
            str | None: A status message if closed/updated, else None.
        """
        raise NotImplementedError("Subclasses must implement manage_active_trade")

    @abstractmethod
    def generate_orders(
        self,
        trade: dict[str, Any],
        dataframe_history: pd.DataFrame,
        budget: float,
        repository: TradeRepository,
    ) -> Order | None:
        """
        Generates orders for the next trading day.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data.
            budget: Total available budget for the trade.
            repository: The trade repository for database access.

        Returns:
            Order | None: The generated order object or None.
        """
        raise NotImplementedError("Subclasses must implement generate_orders")

    @abstractmethod
    def get_current_parameters(
        self,
        trade: dict[str, Any],
        dataframe_history: pd.DataFrame | None,
        repository: TradeRepository | None,
    ) -> TradeParams | None:
        """
        Calculates parameters for logging or display.

        Args:
            trade: The trade data from the database.
            dataframe_history: Historical price data (optional).
            repository: The trade repository (optional).

        Returns:
            TradeParams | None: The calculated trade parameters.
        """
        raise NotImplementedError("Subclasses must implement get_current_parameters")

    # --- Shared Logic & Helpers (DRY) ---

    def _get_signal_date(self, trade: dict[str, Any]) -> pd.Timestamp | None:
        """
        Extracts the signal date from the JSON context.

        Args:
            trade: The trade data dictionary.

        Returns:
            pd.Timestamp | None: The signal date if found.
        """
        date_value = self._get_context_value(trade, "date") or self._get_context_value(
            trade, "setup_date"
        )
        if date_value:
            try:
                return pd.Timestamp(date_value)
            except (ValueError, TypeError):
                return None
        return None

    def _get_context_value(self, trade: dict[str, Any], key: str) -> Any | None:
        """Helper to extract values from the JSON signal_context."""
        try:
            context_data = trade.get("signal_context")
            if not context_data:
                return None

            # If already a dict, use it directly (Repository might have parsed it)
            if isinstance(context_data, dict):
                return context_data.get(key)

            # Otherwise parse from JSON string
            context = json.loads(context_data)
            return context.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    def _get_trading_days_post_signal(
        self, trade: dict[str, Any], dataframe_history: pd.DataFrame
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
        return len(dataframe_history[dataframe_history["date"] > signal_date])

    def _calculate_position_size(
        self, fill_price: float, stop_loss: float, risk_amount: float
    ) -> int:
        """
        Centralized position sizing based on risk.

        Args:
            fill_price: The entry price.
            stop_loss: The initial stop loss.
            risk_amount: The dollar amount to risk.

        Returns:
            int: Number of shares.
        """
        if stop_loss <= 0 or fill_price <= stop_loss:
            logger.warning("Invalid SL or Fill Price for risk calculation.")
            return 0
        risk_per_share = fill_price - stop_loss
        return int(risk_amount / risk_per_share)

    def _execute_activation(
        self,
        trade: dict[str, Any],
        repository: TradeRepository,
        fill_price: float,
        reason: str,
        date_string: str,
        extra_updates: dict[str, Any] | None = None,
    ) -> str:
        """
        Moves a trade from CREATED to ACTIVE status.

        Args:
            trade: The trade dictionary.
            repository: The trade repository.
            fill_price: The price at which the trade filled.
            reason: The reason for fill (e.g. BREAKOUT).
            date_string: The date of activation.
            extra_updates: Optional dictionary of additional fields to update.

        Returns:
            str: Status message.
        """
        # Centralized size calculation
        size = float(trade.get("initial_size") or trade.get("current_size") or 0.0)
        if size <= 0:
            stop_loss = float(trade.get("current_stop_loss") or 0.0)

            # 1. Try Risk-Based sizing (if SL exists)
            if stop_loss > 0 and fill_price > stop_loss:
                risk_amount = float(
                    self._get_context_value(trade, "risk_amount")
                    or trade.get("risk_amount")
                    or 100.0
                )
                size = self._calculate_position_size(fill_price, stop_loss, risk_amount)

            # 2. Try Budget-Based sizing (if size still 0 or no SL)
            if size <= 0 and fill_price > 0:
                budget = float(
                    self._get_context_value(trade, "budget")
                    or trade.get("budget")
                    or self.DEFAULT_BUDGET
                )
                size = int(budget / fill_price)
                if size > 0:
                    logger.debug(
                        f"Using budget-based sizing ({budget}) for {trade.get('symbol')}"
                    )

        if size <= 0:
            return "ERROR: Zero Size"

        update_payload = {
            "status": TradeStatus.ACTIVE,
            "entry_date": date_string,
            "entry_price": fill_price,
            "initial_size": size,
            "current_size": size,
        }
        if extra_updates:
            update_payload.update(extra_updates)

        repository.update_trade(
            trade["id"],
            update_payload,
            reason=f"{reason} FILLED @ {fill_price:.2f}",
        )
        return f"FILLED @ {fill_price:.2f} ({int(size)} Shares)"

    def _execute_immediate_loss(
        self,
        trade: dict[str, Any],
        repository: TradeRepository,
        fill_price: float,
        reason: str,
        stop_loss: float,
        date_string: str,
    ) -> str:
        """
        Handles Day 1 turnarounds: entry and stop hit on same day.

        Args:
            trade: The trade dictionary.
            repository: The trade repository.
            fill_price: The entry price.
            reason: The entry reason.
            stop_loss: The stopped price.
            date_string: The date.

        Returns:
            str: Status message.
        """
        size = float(trade.get("initial_size") or trade.get("current_size") or 0.0)
        if size <= 0:
            if stop_loss > 0 and fill_price > stop_loss:
                risk_amount = float(
                    self._get_context_value(trade, "risk_amount")
                    or trade.get("risk_amount")
                    or 100.0
                )
                size = self._calculate_position_size(fill_price, stop_loss, risk_amount)

            if size <= 0:
                budget = float(
                    self._get_context_value(trade, "budget")
                    or trade.get("budget")
                    or self.DEFAULT_BUDGET
                )
                size = int(budget / fill_price)

        if size <= 0:
            return "ERROR: Zero Size"

        direction = str(self._get_context_value(trade, "direction") or "long").lower()
        if direction == "short":
            profit_and_loss = (fill_price - stop_loss) * size
        else:
            profit_and_loss = (stop_loss - fill_price) * size
            
        repository.update_trade(
            trade["id"],
            {
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
        )
        return (
            f"FILLED @ {fill_price:.2f} -> STOPPED @ {stop_loss:.2f} "
            f"(PnL: {profit_and_loss:.2f})"
        )

    def _close_trade(
        self,
        trade: dict[str, Any],
        repository: TradeRepository,
        exit_price: float,
        reason: str,
        date_string: str,
    ) -> str:
        """
        Closes a trade and calculates final PnL.

        Args:
            trade: The trade dictionary.
            repository: The trade repository.
            exit_price: The price at exit.
            reason: The exit reason.
            date_string: The date of exit.

        Returns:
            str: Status message.
        """
        entry_price = float(trade.get("entry_price") or 0.0)
        current_size = float(trade.get("current_size") or 0.0)
        
        direction = str(self._get_context_value(trade, "direction") or "long").lower()
        if direction == "short":
            pnl_chunk = (entry_price - exit_price) * current_size
        else:
            pnl_chunk = (exit_price - entry_price) * current_size
            
        total_pnl = float(trade.get("realized_pnl") or 0.0) + pnl_chunk

        repository.update_trade(
            trade["id"],
            {
                "status": TradeStatus.CLOSED,
                "exit_reason": reason,
                "exit_price": exit_price,
                "exit_date": date_string,
                "realized_pnl": total_pnl,
                "current_size": 0,
            },
        )
        return f"{reason} @ {exit_price:.2f} (PnL: {total_pnl:.2f})"

    def _invalidate_trade(
        self,
        trade: dict[str, Any],
        repository: TradeRepository,
        low_price: float,
        stop_loss: float,
        date_string: str,
    ) -> str:
        """
        Invalidates a setup before entry.

        Args:
            trade: The trade.
            repository: The repository.
            low_price: The low of the day.
            stop_loss: The stop loss level.
            date_string: The date.

        Returns:
            str: Status message.
        """
        outcome_reason = (
            f"SETUP INVALIDATED: Low {low_price:.2f} <= Stop {stop_loss:.2f}"
        )
        repository.update_trade(
            trade["id"],
            {
                "status": TradeStatus.INVALID,
                "exit_reason": ExitReason.INVALIDATED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason=outcome_reason,
        )
        return outcome_reason

    def _expire_trade(
        self, trade: dict[str, Any], repository: TradeRepository, date_string: str
    ) -> str:
        """
        Signals expiration when entry isn't hit within time limit.

        Args:
            trade: The trade.
            repository: The repository.
            date_string: The date.

        Returns:
            str: Status message.
        """
        repository.update_trade(
            trade["id"],
            {
                "status": TradeStatus.CLOSED,
                "exit_reason": ExitReason.EXPIRED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason="EXPIRED (Time Limit Exceeded)",
        )
        return "EXPIRED"

    def _reject_setup(
        self,
        trade: dict[str, Any],
        repository: TradeRepository,
        date_string: str,
        reason: str,
    ) -> str:
        """
        Rejects a setup that never became active (e.g. missed entry window).

        Args:
            trade: The trade data.
            repository: The repository.
            date_string: Current date.
            reason: Detailed reason for rejection.

        Returns:
            str: Status message.
        """
        repository.update_trade(
            trade["id"],
            {
                "status": TradeStatus.INVALID,
                "exit_reason": ExitReason.INVALIDATED,
                "exit_date": date_string,
                "realized_pnl": 0.0,
            },
            reason=f"REJECTED: {reason}",
        )
        return f"REJECTED: {reason}"
