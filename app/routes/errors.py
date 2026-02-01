import logging
from typing import Any

from flask import Blueprint, jsonify, render_template, request

logger = logging.getLogger(__name__)
errors_bp = Blueprint("errors", __name__)

BLOCKED_EXTENSIONS = (".php", ".aspx", ".jsp", ".cgi", ".env", ".git", ".htaccess")


@errors_bp.app_errorhandler(404)
def page_not_found(e: Any) -> Any:
    path = request.path.lower()

    # Check: Ist es ein Script-Kiddie-Scan?
    is_blocked = (
        path.endswith(BLOCKED_EXTENSIONS) or "wp-admin" in path or "wp-login" in path
    )

    if not is_blocked:
        # NUR loggen, wenn es KEIN geblockter Pfad ist
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        logger.warning(
            f"404 Not Found: {request.method} {request.path} - IP: {client_ip}"
        )

    # Normale 404 Seite
    # API Clients erhalten JSON
    if (
        request.path.startswith(("/webhook", "/screener", "/orders", "/api"))
        or request.is_json
    ):
        return jsonify(
            {"status": "error", "message": "Endpoint not found", "path": request.path}
        ), 404

    return render_template("404.html"), 404


@errors_bp.app_errorhandler(500)
def internal_server_error(e: Any) -> Any:
    logger.error(f"500 Internal Server Error: {e}", exc_info=True)

    if request.path.startswith(("/webhook", "/screener", "/orders")) or request.is_json:
        return jsonify(
            {"status": "error", "message": "Internal Server Error", "detail": str(e)}
        ), 500

    return render_template("500.html"), 500
