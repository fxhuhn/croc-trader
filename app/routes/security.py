import logging
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from flask import current_app, jsonify, request, Response

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R", bound=Response | object)


def _is_ip_whitelisted(client_ip: str, whitelist: list[str] | tuple[str, ...]) -> bool:
    """
    Checks if a given IP is whitelisted.

    Supports:
    - Exact IP matches (e.g., "127.0.0.1")
    - Wildcard ranges (e.g., "172.16.x.x" or "10.0.*.*")
    """
    if not whitelist:
        return False

    client_segments = client_ip.split(".")

    for pattern in whitelist:
        if pattern == client_ip:
            return True

        pattern_segments = pattern.split(".")
        if len(pattern_segments) != len(client_segments):
            continue

        matches = True
        for p_seg, c_seg in zip(pattern_segments, client_segments):
            if p_seg.lower() in ("x", "*"):
                continue
            if p_seg != c_seg:
                matches = False
                break

        if matches:
            return True

    return False


def require_ip_whitelist(func: Callable[P, R]) -> Callable[P, Response | R]:
    """
    Decorator to restrict access to whitelisted IP addresses.

    Checks the client IP against the whitelist defined in the application
    configuration. Supports proxy-aware IP detection via X-Forwarded-For.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Response | R:
        configuration = current_app.config["APP_CONFIG"]

        try:
            security_configuration = configuration.app.security
            whitelist = security_configuration.whitelist
            mode = security_configuration.mode
        except AttributeError:
            logger.error("❌ SECURITY: Security configuration missing! Denying access.")
            return jsonify(
                {
                    "status": "error",
                    "message": "Security Configuration Error",
                }
            ), 500

        # Proxy-Aware IP Detection:
        # 1. Use X-Forwarded-For if available (trusted proxy environment)
        # 2. Fallback to remote_addr
        x_forwarded_for = request.headers.getlist("X-Forwarded-For")
        client_ip = (
            x_forwarded_for[0].split(",")[0].strip()
            if x_forwarded_for
            else request.remote_addr
        )

        if not _is_ip_whitelisted(client_ip, whitelist):
            if mode == "block":
                logger.warning(
                    f"🛡️ SECURITY: Unauthorized IP blocked: {client_ip} "
                    f"(Remote: {request.remote_addr})"
                )
                return jsonify(
                    {
                        "status": "error",
                        "message": "Unauthorized Access",
                    }
                ), 403

            logger.warning(
                f"⚠️ SECURITY: Unauthorized IP warning: {client_ip} "
                f"(Remote: {request.remote_addr}) (Allowing in non-blocking mode)"
            )

        return func(*args, **kwargs)

    return wrapper
