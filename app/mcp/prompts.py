"""Preconfigured MCP Prompt templates for AI assistant workflows."""

import logging

from mcp.server import MCPServer

logger = logging.getLogger(__name__)


def register_prompts(server: MCPServer) -> None:
    """Registers preconfigured analysis prompts on the MCP server."""

    @server.prompt(
        name="daily-briefing",
        description="Generates a comprehensive End-of-Day briefing of the portfolio, screener candidates, and system health.",
    )
    def daily_briefing() -> str:
        """Returns the daily briefing prompt."""
        return (
            "Please perform a complete daily market and portfolio briefing for Croc-Trader:\n"
            "1. Use `get_portfolio_summary` and `get_active_positions` to review open trades, risk levels, and PnL.\n"
            "2. Use `get_screener_candidates` to inspect today's newly generated setups.\n"
            "3. Use `get_system_health` to verify that market data quotes and signal records are up to date.\n"
            "4. Summarize key actions, open risk, and top candidate setups for tomorrow."
        )

    @server.prompt(
        name="trade-post-mortem",
        description="Performs a deep-dive forensic review of a completed or invalidated trade setup.",
    )
    def trade_post_mortem(trade_id: int) -> str:
        """Returns trade post-mortem prompt template."""
        return (
            f"Please conduct a post-mortem review for trade ID {trade_id}:\n"
            f"1. Use `get_trade_detail(trade_id={trade_id})` to inspect the trade parameters, dates, and audit log history.\n"
            "2. Examine whether entry, stop loss, and exit rules were adhered to according to the strategy playbook.\n"
            "3. Evaluate the realized profit/loss, slippage, and reasons for exit.\n"
            "4. Provide constructive feedback on setup quality and execution efficiency."
        )

    @server.prompt(
        name="strategy-review",
        description="Reviews the historical performance and trade distribution for a given trading strategy.",
    )
    def strategy_review(strategy_name: str) -> str:
        """Returns strategy review prompt template."""
        return (
            f"Please conduct a comprehensive performance review for strategy '{strategy_name}':\n"
            f"1. Use `get_trade_history(strategy='{strategy_name}')` to retrieve closed trades.\n"
            f"2. Use `get_active_positions(strategy='{strategy_name}')` to check currently open risk.\n"
            "3. Calculate win rate, profit factor, average hold duration, and maximum drawdown.\n"
            "4. Highlight any risk concentration or recurring failure patterns."
        )
