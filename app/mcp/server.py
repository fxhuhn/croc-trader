"""Croc-Trader Model Context Protocol (MCP) Server.

Initializes the domain MCPServer instance and registers all tools,
resources, and prompt templates.
"""

import logging

from mcp.server import MCPServer

from .prompts import register_prompts
from .resources import register_resources
from .tools import (
    actions,
    market,
    orders,
    portfolio,
    screener,
    system,
)

logger = logging.getLogger(__name__)


def create_mcp_server() -> MCPServer:
    """Creates and configures the domain MCPServer instance for Croc-Trader.

    Returns:
        MCPServer: Configured MCPServer with all tools, resources, and prompts.
    """
    server = MCPServer(
        name="croc-trader",
        instructions=(
            "Croc-Trader EOD Portfolio Management and Screener Domain Server. "
            "Provides inspection tools for positions, market prices, screener candidates, "
            "broker executions, and operational actions."
        ),
    )

    # Register domain tools
    portfolio.register(server)
    screener.register(server)
    market.register(server)
    orders.register(server)
    system.register(server)
    actions.register(server)

    # Register resources & prompts
    register_resources(server)
    register_prompts(server)

    logger.debug("Croc-Trader MCPServer created and tools registered.")
    return server
