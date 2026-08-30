"""MCP Resources exposing URI-addressable static and dynamic data snapshots."""

import json
import logging

from mcp.server import MCPServer

from .tools.portfolio import get_active_positions, get_portfolio_summary
from .tools.system import get_strategy_list, get_system_health

logger = logging.getLogger(__name__)


def register_resources(server: MCPServer) -> None:
    """Registers domain resources on the MCP server."""

    @server.resource(
        "croc://portfolio/active",
        name="active_positions",
        description="Snapshot of all currently active trading positions.",
        mime_type="application/json",
    )
    def resource_active_positions() -> str:
        """Returns JSON text of active positions."""
        positions = get_active_positions()
        return json.dumps(positions, indent=2, default=str)

    @server.resource(
        "croc://portfolio/summary",
        name="portfolio_summary",
        description="Aggregated portfolio summary, budget utilization, and PnL.",
        mime_type="application/json",
    )
    def resource_portfolio_summary() -> str:
        """Returns JSON text of portfolio summary."""
        summary = get_portfolio_summary()
        return json.dumps(summary, indent=2, default=str)

    @server.resource(
        "croc://system/health",
        name="system_health",
        description="System health and SQLite database timestamps.",
        mime_type="application/json",
    )
    def resource_system_health() -> str:
        """Returns JSON text of system health."""
        health = get_system_health()
        return json.dumps(health, indent=2, default=str)

    @server.resource(
        "croc://strategies",
        name="strategy_registry",
        description="Registered trading strategies and their budget configurations.",
        mime_type="application/json",
    )
    def resource_strategy_registry() -> str:
        """Returns JSON text of strategy configurations."""
        strategies = get_strategy_list()
        return json.dumps(strategies, indent=2, default=str)
