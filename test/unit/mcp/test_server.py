"""Unit tests for Croc-Trader MCPServer assembly and registry."""

import asyncio

from mcp.server import MCPServer

from app.mcp.server import create_mcp_server


def test_create_mcp_server_initialization() -> None:
    """Verifies that create_mcp_server returns a properly configured MCPServer instance."""
    server = create_mcp_server()
    assert isinstance(server, MCPServer)
    assert server.name == "croc-trader"
    assert "Croc-Trader" in (server.instructions or "")


def test_mcp_server_tools_registered() -> None:
    """Verifies that all domain tools are registered and have unique names."""
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    tool_names = [tool.name for tool in tools]

    expected_tools = [
        "get_active_positions",
        "get_portfolio_summary",
        "get_trade_history",
        "get_trade_detail",
        "get_screener_candidates",
        "get_turnover_candidates",
        "get_webhook_signals",
        "get_price_history",
        "get_latest_prices",
        "get_symbol_universe",
        "get_broker_active_orders",
        "get_broker_settlements",
        "list_order_csv_files",
        "get_system_health",
        "get_strategy_list",
        "trigger_screener",
        "trigger_single_symbol_debug",
        "trigger_order_generation",
        "trigger_eod_pipeline",
        "trigger_strategy_backfill",
    ]

    for expected in expected_tools:
        assert expected in tool_names, f"Tool '{expected}' is missing from MCPServer"

    # Verify no duplicate tool names
    assert len(tool_names) == len(set(tool_names)), "Duplicate tool names detected"


def test_mcp_server_resources_registered() -> None:
    """Verifies that all domain resources are registered with valid URIs."""
    server = create_mcp_server()
    resources = asyncio.run(server.list_resources())
    uris = [str(resource.uri) for resource in resources]

    expected_uris = [
        "croc://portfolio/active",
        "croc://portfolio/summary",
        "croc://system/health",
        "croc://strategies",
    ]

    for expected in expected_uris:
        assert expected in uris, f"Resource '{expected}' is missing from MCPServer"


def test_mcp_server_prompts_registered() -> None:
    """Verifies that preconfigured prompts are registered."""
    server = create_mcp_server()
    prompts = asyncio.run(server.list_prompts())
    prompt_names = [prompt.name for prompt in prompts]

    expected_prompts = [
        "daily-briefing",
        "trade-post-mortem",
        "strategy-review",
    ]

    for expected in expected_prompts:
        assert expected in prompt_names, (
            f"Prompt '{expected}' is missing from MCPServer"
        )
