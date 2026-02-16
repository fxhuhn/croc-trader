import logging
from typing import Any

from flask import Blueprint, jsonify, render_template, request, Response
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)
errors_bp = Blueprint("errors", __name__)

BLOCKED_EXTENSIONS = (".php", ".aspx", ".jsp", ".cgi", ".env", ".git", ".htaccess", ".ico")
API_PREFIXES = ("/webhook", "/screener", "/orders", "/api")


@errors_bp.app_errorhandler(404)
def page_not_found(e: Exception) -> tuple[Response | str, int]:
    """
    Global 404 Handler.
    
    Features:
    - Suppresses logs for known scanner patterns (script kiddies).
    - Returns JSON for API clients.
    - Returns HTML for Browser clients.

    Args:
        e: The exception object.

    Returns:
        tuple: (Response or HTML string, Status Code 404)
    """
    path = request.path.lower()

    # Check: Ist es ein Script-Kiddie-Scan?
    is_blocked = (
        path.endswith(BLOCKED_EXTENSIONS) or "wp-admin" in path or "wp-login" in path
    )

    if not is_blocked:
        # NUR loggen, wenn es KEIN geblockter Pfad ist
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        # Security: Allow only safe characters in log
        safe_path = request.path.replace("\n", "").replace("\r", "")
        safe_method = request.method.replace("\n", "").replace("\r", "")
        logger.warning(
            f"404 Not Found: {safe_method} {safe_path} - IP: {client_ip}"
        )

    # Normale 404 Seite
    # API Clients erhalten JSON
    if request.path.startswith(API_PREFIXES) or request.is_json:
        return jsonify(
            {"status": "error", "message": "Endpoint not found", "path": request.path}
        ), 404

    return render_template("404.html"), 404


@errors_bp.app_errorhandler(500)
def internal_server_error(e: Exception) -> tuple[Response | str, int]:
    """
    Global 500 Handler.
    
    Catches all unhandled exceptions.
    - Logs the full stacktrace (critical for debugging).
    - Returns JSON for API clients to avoid leaking HTML stacktraces.
    - Returns generic Error Page for Browser clients.

    Args:
        e: The exception object.

    Returns:
        tuple: (Response or HTML string, Status Code 500)
    """
    logger.error(f"500 Internal Server Error: {e}", exc_info=True)

    if request.path.startswith(API_PREFIXES) or request.is_json:
        # Security: Do NOT return str(e) to client, as it may contain sensitive info
        # (SQL queries, file paths, etc.)
        return jsonify(
            {"status": "error", "message": "Internal Server Error"}
        ), 500

    return render_template("500.html"), 500
