"""Model Context Protocol (MCP) Streamable HTTP / JSON-RPC 2.0 Blueprint."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, cast

from flask import Blueprint, Response, current_app, jsonify, request

from .security import require_ip_whitelist

logger = logging.getLogger(__name__)

mcp_bp = Blueprint("mcp", __name__)

type ApiResponse = Response | tuple[Response, int]


def _jsonrpc_error(
    request_id: Any, code: int, message: str, data: Any = None
) -> dict[str, Any]:
    """Formats a standard JSON-RPC 2.0 error object."""
    error_payload: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error_payload["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error_payload}


def _jsonrpc_success(request_id: Any, result: Any) -> dict[str, Any]:
    """Formats a standard JSON-RPC 2.0 success object."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _handle_initialize(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP initialize and discovery methods."""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {
            "name": getattr(server, "name", "croc-trader"),
            "version": getattr(server, "version", "1.0.0"),
        },
        "instructions": getattr(server, "instructions", ""),
    }


def _handle_ping(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP ping method."""
    return {}


def _handle_tools_list(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP tools/list method."""
    tools = asyncio.run(server.list_tools())
    return {
        "tools": [
            cast(dict[str, Any], tool.model_dump(mode="json", by_alias=True))
            for tool in tools
        ]
    }


def _handle_tools_call(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP tools/call method."""
    name = params.get("name")
    if not name:
        raise ValueError("Missing 'name' in tool call parameters")
    arguments = params.get("arguments", {})
    logger.debug("Executing MCP tool: %s | arguments: %s", name, arguments)
    result = asyncio.run(server.call_tool(name, arguments))
    logger.debug("MCP tool '%s' execution completed.", name)
    return cast(dict[str, Any], result.model_dump(mode="json", by_alias=True))


def _handle_resources_list(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP resources/list method."""
    resources = asyncio.run(server.list_resources())
    return {
        "resources": [
            cast(dict[str, Any], r.model_dump(mode="json", by_alias=True))
            for r in resources
        ]
    }


def _handle_resources_read(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP resources/read method."""
    uri = params.get("uri")
    if not uri:
        raise ValueError("Missing 'uri' in resource read parameters")
    logger.debug("Reading MCP resource: %s", uri)
    contents = asyncio.run(server.read_resource(uri))
    serialized_contents = []
    for item in contents:
        serialized_contents.append(
            {
                "uri": str(uri),
                "mimeType": (
                    getattr(item, "mime_type", "application/json") or "application/json"
                ),
                "text": getattr(item, "content", ""),
            }
        )
    return {"contents": serialized_contents}


def _handle_prompts_list(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP prompts/list method."""
    prompts = asyncio.run(server.list_prompts())
    return {
        "prompts": [
            cast(dict[str, Any], p.model_dump(mode="json", by_alias=True))
            for p in prompts
        ]
    }


def _handle_prompts_get(server: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Handles MCP prompts/get method."""
    name = params.get("name")
    if not name:
        raise ValueError("Missing 'name' in prompt parameters")
    arguments = params.get("arguments", {})
    logger.debug("Retrieving MCP prompt: %s | arguments: %s", name, arguments)
    result = asyncio.run(server.get_prompt(name, arguments))
    return cast(dict[str, Any], result.model_dump(mode="json", by_alias=True))


_MCP_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "initialize": _handle_initialize,
    "server/discover": _handle_initialize,
    "ping": _handle_ping,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "prompts/list": _handle_prompts_list,
    "prompts/get": _handle_prompts_get,
}


def _dispatch_method(
    server: Any, method: str, params: dict[str, Any], request_id: Any
) -> tuple[dict[str, Any], int]:
    """Dispatches a validated JSON-RPC method to its corresponding MCP handler."""
    handler = _MCP_HANDLERS.get(method)
    if not handler:
        logger.warning("Unknown MCP method requested: %s", method)
        return _jsonrpc_error(request_id, -32601, f"Method not found: '{method}'"), 404

    try:
        result = handler(server, params)
        return _jsonrpc_success(request_id, result), 200
    except ValueError as val_err:
        logger.warning("Invalid parameters for MCP method '%s': %s", method, val_err)
        return _jsonrpc_error(request_id, -32602, str(val_err)), 400
    except Exception as error:
        logger.exception("Internal error processing MCP method '%s': %s", method, error)
        return _jsonrpc_error(request_id, -32603, f"Internal error: {error}"), 500


@mcp_bp.route("/mcp", methods=["GET", "POST"])
@require_ip_whitelist
def handle_mcp_request() -> ApiResponse:
    """Streamable HTTP / JSON-RPC 2.0 entrypoint for the MCP Server."""
    mcp_server = current_app.extensions.get("mcp_server")
    if not mcp_server:
        return (
            jsonify(
                {
                    "error": "MCP Domain Server not initialized or disabled in settings.yaml"
                }
            ),
            503,
        )

    if request.method == "GET":
        return (
            jsonify(
                {
                    "status": "healthy",
                    "server": getattr(mcp_server, "name", "croc-trader"),
                    "version": getattr(mcp_server, "version", "1.0.0"),
                    "transport": "streamable-http",
                    "protocol": "MCP JSON-RPC 2.0",
                }
            ),
            200,
        )

    try:
        payload = request.get_json(force=True, silent=True)
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return jsonify(_jsonrpc_error(None, -32700, "Parse error: Invalid JSON")), 400

    method = payload.get("method")
    if method and method.startswith("notifications/"):
        logger.debug("Received MCP notification: %s", method)
        return Response("", status=204, mimetype="application/json")

    request_id = payload.get("id")
    if not method:
        return (
            jsonify(
                _jsonrpc_error(request_id, -32600, "Invalid Request: Missing method")
            ),
            400,
        )

    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}

    response_payload, status_code = _dispatch_method(
        mcp_server, method, params, request_id
    )
    return jsonify(response_payload), status_code


@mcp_bp.route("/.well-known/oauth-protected-resource", methods=["GET"])
@mcp_bp.route("/.well-known/oauth-protected-resource/mcp", methods=["GET"])
@require_ip_whitelist
def handle_oauth_metadata_discovery() -> ApiResponse:
    """Handles OAuth 2.0 protected resource metadata discovery probes gracefully."""
    return (
        jsonify(
            {
                "resource": request.base_url,
                "auth_methods": ["ip_whitelist"],
                "mcp_endpoint": "/mcp",
            }
        ),
        200,
    )
