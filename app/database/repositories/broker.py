"""Repository for accessing Trader Workstation (TWS) execution and settlement data.

Provides read-only access to the tables of `trading.db` including orders,
executions, and settlements, mapping rows to structured dictionaries.
"""

import logging
from typing import Any

from .base import BaseRepository

logger = logging.getLogger(__name__)


class BrokerRepository(BaseRepository):
    """Handles read-only queries against the TWS trading.db database.

    All methods use synchronous SQL execution via DatabaseSession.
    """

    def get_orders_by_status(self, status: str) -> list[dict[str, Any]]:
        """Retrieves orders filtered by their current lifecycle status.

        Args:
            status: The order status to filter by (e.g. 'Submitted', 'Error').

        Returns:
            list[dict[str, Any]]: A list of order records as dictionaries.
        """
        query_string = (
            "SELECT * FROM orders WHERE status = ? ORDER BY transmitted_at DESC"
        )
        rows = self.fetch_all(query_string, (status,))
        return [dict(row) for row in rows]

    def get_all_orders(self) -> list[dict[str, Any]]:
        """Retrieves all orders from the database.

        Returns:
            list[dict[str, Any]]: A list of all order records.
        """
        query_string = "SELECT * FROM orders ORDER BY transmitted_at DESC"
        rows = self.fetch_all(query_string)
        return [dict(row) for row in rows]

    def get_executions_for_order(self, order_id: int) -> list[dict[str, Any]]:
        """Retrieves TWS executions (fills/partial fills) associated with an order.

        Args:
            order_id: The TWS order identifier.

        Returns:
            list[dict[str, Any]]: A list of execution records.
        """
        query_string = (
            "SELECT * FROM executions WHERE order_id = ? ORDER BY executed_at ASC"
        )
        rows = self.fetch_all(query_string, (order_id,))
        return [dict(row) for row in rows]

    def get_executions_for_trade_group(
        self, trade_group_identifier: str
    ) -> list[dict[str, Any]]:
        """Retrieves executions associated with a trade group.

        Args:
            trade_group_identifier: The trade group identifier (e.g., '768_TurnoverTiming_0.5_TSLA').

        Returns:
            list[dict[str, Any]]: A list of execution records with their order details.
        """
        query_string = """
            SELECT e.*, o.action, o.symbol, o.bracket_role, o.order_type
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.trade_group_id = ?
            ORDER BY e.executed_at ASC
        """
        rows = self.fetch_all(query_string, (trade_group_identifier,))
        return [dict(row) for row in rows]

    def get_settlements(self) -> list[dict[str, Any]]:
        """Retrieves closed trade settlements.

        Returns:
            list[dict[str, Any]]: A list of settlement records.
        """
        query_string = "SELECT * FROM trades_settlement ORDER BY settled_at DESC"
        rows = self.fetch_all(query_string)
        return [dict(row) for row in rows]

    def get_net_positions_by_symbol(self) -> dict[str, float]:
        """Calculates current net execution quantities grouped by symbol.

        Aggregates quantity from executions table where action is BUY (positive)
        or SELL (negative). Ignores manual strategies 'SplitTarget' and 'HoldTarget'.

        Returns:
            dict[str, float]: Mapping of symbol to net executed position quantity.
        """
        query_string = """
            SELECT
                o.symbol,
                SUM(CASE WHEN o.action = 'BUY' THEN e.qty ELSE -e.qty END) AS net_quantity
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.strategy_name NOT IN ('SplitTarget', 'HoldTarget')
            GROUP BY o.symbol
        """
        rows = self.fetch_all(query_string)
        return {
            str(row["symbol"]): float(row["net_quantity"])
            for row in rows
            if row["symbol"]
        }

    def get_orders_by_local_trade_id(self, local_trade_id: int) -> list[dict[str, Any]]:
        """Retrieves orders from trading.db matching a local trade ID prefix.

        Args:
            local_trade_id: The local trade database ID.

        Returns:
            list[dict[str, Any]]: List of matching order records.
        """
        query_string = "SELECT * FROM orders WHERE trade_group_id LIKE ?"
        rows = self.fetch_all(query_string, (f"{local_trade_id}_%",))
        return [dict(row) for row in rows]

    def get_active_positions(self) -> list[dict[str, Any]]:
        """Retrieves active positions directly from trading.db executions.

        Returns:
            list[dict[str, Any]]: Active positions list.
        """
        active_groups_query = """
            SELECT
                o.trade_group_id,
                o.symbol,
                o.strategy_name,
                SUM(CASE WHEN o.action = 'BUY' THEN e.qty ELSE -e.qty END) AS net_quantity
            FROM executions e
            JOIN orders o ON e.order_id = o.order_id
            WHERE o.strategy_name NOT IN ('SplitTarget', 'HoldTarget')
            GROUP BY o.trade_group_id
            HAVING net_quantity > 0.0
        """
        active_group_rows = [dict(row) for row in self.fetch_all(active_groups_query)]
        active_positions: list[dict[str, Any]] = []

        for group_row in active_group_rows:
            trade_group_id = str(group_row["trade_group_id"])
            symbol = str(group_row["symbol"])
            net_quantity = float(group_row["net_quantity"])
            strategy_name = str(group_row["strategy_name"])

            executions_query = """
                SELECT e.*, o.strategy_name, o.action, o.trade_group_id
                FROM executions e
                JOIN orders o ON e.order_id = o.order_id
                WHERE o.trade_group_id = ?
                ORDER BY e.executed_at ASC
            """
            trade_group_executions = [
                dict(row) for row in self.fetch_all(executions_query, (trade_group_id,))
            ]

            buy_executions = [r for r in trade_group_executions if r["action"] == "BUY"]
            total_buy_qty = sum(b["qty"] for b in buy_executions)
            total_buy_cost = sum(b["qty"] * b["price"] for b in buy_executions)
            avg_entry_price = (
                total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0.0
            )

            latest_buy = (
                buy_executions[-1]
                if buy_executions
                else (trade_group_executions[-1] if trade_group_executions else None)
            )

            parts = trade_group_id.split("_")
            local_id = int(parts[0]) if parts and parts[0].isdigit() else 0

            # Get the latest execution price for this symbol across all executions
            fallback_price_query = """
                SELECT e.price FROM executions e
                JOIN orders o ON e.order_id = o.order_id
                WHERE o.symbol = ?
                ORDER BY e.executed_at DESC LIMIT 1
            """
            fallback_price_row = self.fetch_one(fallback_price_query, (symbol,))
            current_price = (
                fallback_price_row["price"]
                if fallback_price_row
                else (
                    trade_group_executions[-1]["price"]
                    if trade_group_executions
                    else 0.0
                )
            )

            orders_query = """
                SELECT o.*,
                       COALESCE(e.price, NULLIF(o.target_price, 0)) AS display_price,
                       COALESCE(e.executed_at, o.transmitted_at) AS display_date
                FROM orders o
                LEFT JOIN executions e ON e.order_id = o.order_id
                WHERE o.trade_group_id = ?
                ORDER BY COALESCE(e.executed_at, o.transmitted_at) ASC
            """
            tws_orders = [
                dict(row) for row in self.fetch_all(orders_query, (trade_group_id,))
            ]

            tws_status = "Filled"
            if tws_orders:
                order_statuses = [o.get("status") for o in tws_orders]
                if "Error" in order_statuses:
                    tws_status = "Error"
                elif "Submitted" in order_statuses:
                    tws_status = "Submitted"
                elif "PreSubmitted" in order_statuses:
                    tws_status = "PreSubmitted"

            active_positions.append(
                {
                    "id": local_id,
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "entry_date": latest_buy["executed_at"][:10]
                    if latest_buy and latest_buy.get("executed_at")
                    else "-",
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
            )

        return active_positions
