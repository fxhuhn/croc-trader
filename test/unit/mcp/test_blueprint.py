"""Unit tests for the Flask MCP Blueprint JSON-RPC 2.0 endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.mcp.server import create_mcp_server
from app.routes.mcp import mcp_bp


@pytest.fixture
def mcp_app() -> Flask:
    """Configures an isolated Flask test application fixture with MCP server."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.get_path.side_effect = lambda name: f"/mock/path/{name}.db"
    mock_config.get_db_path.side_effect = lambda name: f"/mock/path/{name}.db"
    mock_config.app.security.mode = "warning"
    mock_config.app.security.whitelist = ("127.0.0.1",)
    mock_config.app.portfolio.get_budget.return_value = 5000.0
    mock_config.app.portfolio.get_risk_amount.return_value = 100.0
    app.config["APP_CONFIG"] = mock_config

    server = create_mcp_server()
    app.extensions["mcp_server"] = server

    app.register_blueprint(mcp_bp)
    return app


def test_mcp_disabled_returns_503() -> None:
    """Verifies that accessing /mcp returns 503 if mcp_server is not registered."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.app.security.mode = "warning"
    mock_config.app.security.whitelist = ("127.0.0.1",)
    app.config["APP_CONFIG"] = mock_config

    app.register_blueprint(mcp_bp)

    client = app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
    )
    assert response.status_code == 503
    data = response.get_json()
    assert "error" in data


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_initialize(mcp_app: Flask) -> None:
    """Verifies that initialize returns protocol version and server capabilities."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert "result" in data
    assert "protocolVersion" in data["result"]
    assert data["result"]["serverInfo"]["name"] == "croc-trader"
    assert "capabilities" in data["result"]


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_ping(mcp_app: Flask) -> None:
    """Verifies ping method returns empty result dictionary."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 2
    assert data["result"] == {}


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_tools_list(mcp_app: Flask) -> None:
    """Verifies tools/list returns list of available tools."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 3
    assert "tools" in data["result"]
    tool_names = [t["name"] for t in data["result"]["tools"]]
    assert "get_active_positions" in tool_names
    assert "get_strategy_list" in tool_names


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_tools_call(mcp_app: Flask) -> None:
    """Verifies tools/call executes tool function and returns result."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "get_strategy_list", "arguments": {}},
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["id"] == 4
    assert "result" in data
    assert data["result"]["isError"] is False


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_resources_list(mcp_app: Flask) -> None:
    """Verifies resources/list returns registered resource definitions."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 5, "method": "resources/list"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "resources" in data["result"]
    uris = [r["uri"] for r in data["result"]["resources"]]
    assert "croc://strategies" in uris


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_resources_read(mcp_app: Flask) -> None:
    """Verifies resources/read returns resource content."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {"uri": "croc://strategies"},
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "contents" in data["result"]
    assert len(data["result"]["contents"]) > 0
    assert data["result"]["contents"][0]["uri"] == "croc://strategies"


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_prompts_list(mcp_app: Flask) -> None:
    """Verifies prompts/list returns registered prompt templates."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 7, "method": "prompts/list"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prompts" in data["result"]
    prompt_names = [p["name"] for p in data["result"]["prompts"]]
    assert "daily-briefing" in prompt_names


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_prompts_get(mcp_app: Flask) -> None:
    """Verifies prompts/get returns rendered prompt messages."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 8,
            "method": "prompts/get",
            "params": {"name": "daily-briefing", "arguments": {}},
        },
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "messages" in data["result"]
    assert len(data["result"]["messages"]) > 0


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_notifications_return_204(mcp_app: Flask) -> None:
    """Verifies that notifications return HTTP 204 without JSON-RPC response."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
    )
    assert response.status_code == 204


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_invalid_json_returns_parse_error(mcp_app: Flask) -> None:
    """Verifies that non-JSON payloads return JSON-RPC error -32700."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        data="not a json",
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]["code"] == -32700


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_missing_method_returns_invalid_request(mcp_app: Flask) -> None:
    """Verifies missing method returns JSON-RPC error -32600."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 10},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]["code"] == -32600


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_unknown_method_returns_method_not_found(mcp_app: Flask) -> None:
    """Verifies unknown method returns JSON-RPC error -32601."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 11, "method": "non_existent_method"},
    )
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"]["code"] == -32601


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_tools_call_missing_name_returns_invalid_params(mcp_app: Flask) -> None:
    """Verifies tools/call missing name parameter returns error -32602."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 12, "method": "tools/call", "params": {}},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]["code"] == -32602


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_get_request_returns_health_info(mcp_app: Flask) -> None:
    """Verifies GET /mcp returns health and transport info."""
    client = mcp_app.test_client()
    response = client.get("/mcp")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["protocol"] == "MCP JSON-RPC 2.0"


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_mcp_server_discover_returns_init_info(mcp_app: Flask) -> None:
    """Verifies server/discover method returns server metadata."""
    client = mcp_app.test_client()
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 13, "method": "server/discover"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["result"]["serverInfo"]["name"] == "croc-trader"


@patch("app.routes.mcp.require_ip_whitelist", lambda f: f)
def test_oauth_metadata_discovery(mcp_app: Flask) -> None:
    """Verifies OAuth metadata discovery endpoints return 200."""
    client = mcp_app.test_client()
    for endpoint in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
    ):
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.get_json()
        assert data["auth_methods"] == ["ip_whitelist"]
