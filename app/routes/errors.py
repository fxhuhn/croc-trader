import logging
import time
from collections import defaultdict
from threading import Lock

from flask import Blueprint, jsonify, render_template, request, Response

logger = logging.getLogger(__name__)
errors_bp = Blueprint("errors", __name__)

BLOCKED_EXTENSIONS = (
    ".php",
    ".php7",
    ".aspx",
    ".jsp",
    ".cgi",
    ".env",
    ".git",
    ".htaccess",
    ".xml",
    ".yaml",
    ".yml",
    ".bak",
    ".old",
)
API_PREFIXES = ("/screener", "/orders", "/api")


class MissingRouteRateLimiter:
    """Tracks failed requests (404s) per IP to block scanners."""

    max_failures: int
    time_window: int
    _ip_records: dict[str, list[float]]
    _lock: Lock

    def __init__(self, max_failures: int = 3, time_window: int = 5) -> None:
        self.max_failures = max_failures
        self.time_window = time_window
        self._ip_records = defaultdict(list)
        self._lock = Lock()

    def should_block(self, ip_address: str) -> bool:
        """
        Records a failure for the given IP and checks if it exceeds the limit.
        """
        current_time = time.time()
        with self._lock:
            # Drop failures older than time_window
            active_failures = [
                timestamp
                for timestamp in self._ip_records[ip_address]
                if current_time - timestamp <= self.time_window
            ]
            active_failures.append(current_time)
            self._ip_records[ip_address] = active_failures

            return len(active_failures) > self.max_failures


_rate_limiter = MissingRouteRateLimiter()


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
    client_ip: str | None = request.headers.get("X-Forwarded-For", request.remote_addr)
    resolved_ip = str(client_ip) if client_ip else "0.0.0.0"  # nosec B104

    if _rate_limiter.should_block(resolved_ip):
        # Quickly drop connection to reduce load / bypass templating overhead
        return Response("Too Many Requests", mimetype="text/plain"), 429

    path = request.path.lower()

    # Check: Ist es ein Script-Kiddie-Scan?
    is_blocked = (
        path.endswith(BLOCKED_EXTENSIONS) or "wp-admin" in path or "wp-login" in path
    )

    if not is_blocked:
        # NUR loggen, wenn es KEIN geblockter Pfad ist
        # Security: Allow only safe characters in log
        safe_path = request.path.replace("\n", "").replace("\r", "")
        safe_method = request.method.replace("\n", "").replace("\r", "")
        logger.warning(f"404 Not Found: {safe_method} {safe_path} - IP: {resolved_ip}")

    # Normale 404 Seite
    # API Clients erhalten JSON
    if request.path.startswith(API_PREFIXES) or request.is_json:
        return jsonify(
            {
                "status": "error",
                "message": "Endpoint not found",
                "path": request.path,
            }
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
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500

    return render_template("500.html"), 500
