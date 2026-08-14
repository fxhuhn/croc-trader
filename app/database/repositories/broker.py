"""Repository for accessing Trader Workstation (TWS) execution and settlement data.

Provides read-only access to the tables of `trading.db` including orders,
executions, and settlements, mapping rows to structured TypedDict records.
"""

import logging
from typing import NotRequired, TypedDict, cast

from ...const import Strategies
from .base import BaseRepository

logger = logging.getLogger(__name__)

EXCLUDED_STRATEGIES: tuple[str, ...] = (
    Strategies.SplitTarget.value,
    Strategies.HoldTarget.value,
    "SplitTarget",
    "HoldTarget",
)


class OrderRecord(TypedDict):
    """Structured dictionary representation of an order row."""

    order_id: int
    perm_id: NotRequired[int | None]
    parent_id: NotRequired[int | None]
    trade_group_id: str
    account_id: str
    bracket_role: NotRequired[str | None]
    symbol: str
    sec_type: NotRequired[str | None]
    exchange: NotRequired[str | None]
    action: str
    quantity: float
    order_type: str
    target_price: float
    tif: NotRequired[str | None]
    strategy_name: str
    status: str
    retry_count: NotRequired[int]
    transmitted_at: NotRequired[str | None]
    display_price: NotRequired[float | None]
    display_date: NotRequired[str | None]
    commission: NotRequired[float | None]


class ExecutionRecord(TypedDict):
    """Structured dictionary representation of an execution row."""

    exec_id: str
    order_id: int
    price: float
    qty: float
    commission: float | None
    currency: NotRequired[str | None]
    executed_at: NotRequired[str | None]
    action: NotRequired[str]
    symbol: NotRequired[str]
    bracket_role: NotRequired[str]
    order_type: NotRequired[str]
    target_price: NotRequired[float | None]
    strategy_name: NotRequired[str]


class SettlementRecord(TypedDict):
    """Structured dictionary representation of a settlement row."""

    account_id: str
    trade_group_id: str
    avg_entry_price: float
    avg_exit_price: float
    price_diff_slippage: float
    total_commissions: float
    net_pnl: float
    settled_at: str


class ActivePositionRecord(TypedDict):
    """Structured dictionary representation of an active open position."""

    id: int
    symbol: str
    strategy: str
    entry_date: str
    days_held: int
    current_size: float
    entry_price: float
    current_price: float
    tws_status: str
    tws_orders: list[OrderRecord]
    unrealized_pnl: float
    pnl_percentage: float
    trade_group_id: str


class BrokerRepository(BaseRepository):
    """Handles read-only queries against the TWS trading.db database.

    All methods use synchronous SQL execution via DatabaseSession.
    """

    def get_orders_by_status(self, status: str) -> list[OrderRecord]:
        """Retrieves orders filtered by their current lifecycle status.

        Args:
            status: The order status to filter by (e.g. 'Submitted', 'Error').

        Returns:
            list[OrderRecord]: A list of order records.
        """
        query_string = (
            "SELECT * FROM orders WHERE status = ? ORDER BY transmitted_at DESC"
        )
        rows = self.fetch_all(query_string, (status,))
        return cast(list[OrderRecord], [dict(row) for row in rows])

    def get_all_orders(self) -> list[OrderRecord]:
        """Retrieves all orders from the database.

        Returns:
            list[OrderRecord]: A list of all order records.
        """
        query_string = "SELECT * FROM orders ORDER BY transmitted_at DESC"
        rows = self.fetch_all(query_string)
        return cast(list[OrderRecord], [dict(row) for row in rows])

    def get_executions_for_order(self, order_id: int) -> list[ExecutionRecord]:
        """Retrieves TWS executions (fills/partial fills) associated with an order.

        Args:
            order_id: The TWS order identifier.

        Returns:
            list[ExecutionRecord]: A list of execution records.
        """
        query_string = (
            "SELECT * FROM executions WHERE order_id = ? ORDER BY executed_at ASC"
        )
        rows = self.fetch_all(query_string, (order_id,))
        return cast(list[ExecutionRecord], [dict(row) for row in rows])

    def get_executions_for_trade_group(
        self, trade_group_identifier: str
    ) -> list[ExecutionRecord]:
        """Retrieves executions associated with a trade group.

        Args:
            trade_group_identifier: The trade group identifier (e.g., '768_TurnoverTiming_0.5_TSLA').

        Returns:
            list[ExecutionRecord]: A list of execution records with order details.
        """
        query_string = """
            SELECT e.*, o.strategy_name, o.action, o.symbol, o.bracket_role, o.order_type, o.target_price
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.trade_group_id = ?
            ORDER BY e.executed_at ASC
        """
        rows = self.fetch_all(query_string, (trade_group_identifier,))
        return cast(list[ExecutionRecord], [dict(row) for row in rows])

    def get_settlements(self) -> list[SettlementRecord]:
        """Retrieves closed trade settlements.

        Returns:
            list[SettlementRecord]: A list of settlement records.
        """
        query_string = "SELECT * FROM trades_settlement ORDER BY settled_at DESC"
        rows = self.fetch_all(query_string)
        return cast(list[SettlementRecord], [dict(row) for row in rows])

    def get_net_positions_by_symbol(self) -> dict[str, float]:
        """Calculates current net execution quantities grouped by symbol.

        Aggregates quantity from executions table where action is BUY (positive)
        or SELL (negative). Ignores manual strategies 'SplitTarget' and 'HoldTarget'.

        Returns:
            dict[str, float]: Mapping of symbol to net executed position quantity.
        """
        placeholders = ",".join("?" for _ in EXCLUDED_STRATEGIES)
        query_string = f"""
            SELECT
                o.symbol,
                SUM(CASE WHEN o.action = 'BUY' THEN e.qty ELSE -e.qty END) AS net_quantity
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.strategy_name NOT IN ({placeholders})
            GROUP BY o.symbol
        """  # nosec B608
        rows = self.fetch_all(query_string, EXCLUDED_STRATEGIES)
        return {
            str(row["symbol"]): float(row["net_quantity"])
            for row in rows
            if row["symbol"]
        }

    def get_orders_by_local_trade_id(self, local_trade_id: int) -> list[OrderRecord]:
        """Retrieves orders from trading.db matching a local trade ID prefix.

        Args:
            local_trade_id: The local trade database ID.

        Returns:
            list[OrderRecord]: List of matching order records.
        """
        query_string = "SELECT * FROM orders WHERE trade_group_id LIKE ?"
        rows = self.fetch_all(query_string, (f"{local_trade_id}_%",))
        return cast(list[OrderRecord], [dict(row) for row in rows])

    def get_active_positions(self) -> list[ActivePositionRecord]:
        """Retrieves active positions directly from trading.db executions.

        Returns:
            list[ActivePositionRecord]: Active positions list.
        """
        placeholders = ",".join("?" for _ in EXCLUDED_STRATEGIES)
        active_groups_query = f"""
            SELECT
                o.trade_group_id,
                o.symbol,
                o.strategy_name,
                SUM(CASE WHEN o.action = 'BUY' THEN e.qty ELSE -e.qty END) AS net_quantity
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.strategy_name NOT IN ({placeholders})
            GROUP BY o.trade_group_id
            HAVING net_quantity > 0.0
        """  # nosec B608
        active_group_rows = [
            dict(row)
            for row in self.fetch_all(active_groups_query, EXCLUDED_STRATEGIES)
        ]
        active_positions: list[ActivePositionRecord] = []

        for group_row in active_group_rows:
            trade_group_id = str(group_row["trade_group_id"])
            symbol = str(group_row["symbol"])
            net_quantity = float(group_row["net_quantity"])
            strategy_name = str(group_row["strategy_name"])

            trade_group_executions = self.get_executions_for_trade_group(trade_group_id)
            avg_entry_price = self._calculate_average_entry_price(
                trade_group_executions
            )
            latest_buy = self._resolve_latest_buy_execution(trade_group_executions)
            current_price = self._resolve_latest_price_fallback(
                symbol, trade_group_executions
            )

            tws_status, tws_orders = self._determine_tws_status(trade_group_id)

            parts = trade_group_id.split("_")
            local_id = int(parts[0]) if parts and parts[0].isdigit() else 0

            latest_execution_date = (
                latest_buy["executed_at"][:10]
                if latest_buy and latest_buy.get("executed_at")
                else "-"
            )

            position_record: ActivePositionRecord = {
                "id": local_id,
                "symbol": symbol,
                "strategy": strategy_name,
                "entry_date": latest_execution_date,
                "days_held": 0,
                "current_size": net_quantity,
                "entry_price": avg_entry_price,
                "current_price": current_price,
                "tws_status": tws_status,
                "tws_orders": tws_orders,
                "unrealized_pnl": 0.0,
                "pnl_percentage": 0.0,
                "trade_group_id": trade_group_id,
            }
            active_positions.append(position_record)

        return active_positions

    def _calculate_average_entry_price(
        self, executions: list[ExecutionRecord]
    ) -> float:
        """Calculates average entry price from BUY executions."""
        buy_executions = [
            execution for execution in executions if execution.get("action") == "BUY"
        ]
        total_buy_quantity = sum(
            buy_execution["qty"] for buy_execution in buy_executions
        )
        total_buy_cost = sum(
            buy_execution["qty"] * buy_execution["price"]
            for buy_execution in buy_executions
        )
        return total_buy_cost / total_buy_quantity if total_buy_quantity > 0 else 0.0

    def _resolve_latest_buy_execution(
        self, executions: list[ExecutionRecord]
    ) -> ExecutionRecord | None:
        """Determines the last buy execution or fallback execution."""
        buy_executions = [
            execution for execution in executions if execution.get("action") == "BUY"
        ]
        if buy_executions:
            return buy_executions[-1]
        return executions[-1] if executions else None

    def _resolve_latest_price_fallback(
        self, symbol: str, executions: list[ExecutionRecord]
    ) -> float:
        """Queries the absolute latest execution price for the symbol safely."""
        fallback_price_query = """
            SELECT e.price FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.symbol = ? AND e.price IS NOT NULL
            ORDER BY e.executed_at DESC LIMIT 1
        """
        fallback_price_row = self.fetch_one(fallback_price_query, (symbol,))
        if fallback_price_row and fallback_price_row["price"] is not None:
            return float(fallback_price_row["price"])

        if executions:
            latest_price = executions[-1].get("price")
            if latest_price is not None:
                return float(latest_price)

        return 0.0

    def _determine_tws_status(
        self, trade_group_id: str
    ) -> tuple[str, list[OrderRecord]]:
        """Resolves TWS state and loads associated orders strictly for a trade group."""
        orders_query = """
            SELECT o.*,
                   COALESCE(MAX(e.price), NULLIF(o.target_price, 0)) AS display_price,
                   COALESCE(MAX(e.executed_at), o.transmitted_at) AS display_date,
                   SUM(e.commission) AS commission
            FROM orders o
            LEFT JOIN executions e ON e.order_id = o.order_id
            WHERE o.trade_group_id = ?
            GROUP BY o.order_id
            ORDER BY COALESCE(MAX(e.executed_at), o.transmitted_at) ASC
        """
        tws_orders = cast(
            list[OrderRecord],
            [dict(row) for row in self.fetch_all(orders_query, (trade_group_id,))],
        )
        tws_status = "Filled"

        if tws_orders:
            order_statuses = [order.get("status") for order in tws_orders]
            if "Error" in order_statuses:
                tws_status = "Error"
            elif "Submitted" in order_statuses:
                tws_status = "Submitted"
            elif "PreSubmitted" in order_statuses:
                tws_status = "PreSubmitted"

        return tws_status, tws_orders
