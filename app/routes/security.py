import ipaddress
import logging
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from flask import Response, current_app, jsonify, request

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R", bound=Response | object)


def _is_ip_whitelisted(client_ip: str, whitelist: list[str] | tuple[str, ...]) -> bool:
    """Checks if a given IP address matches any whitelist entry or wildcard pattern.

    Supports:
    - Exact IP matches (e.g. "127.0.0.1")
    - Wildcards in any segment (e.g. "172.16.x.x", "10.0.*.*", "8.*.8.8", "172.16.X.X")
    """
    if not whitelist:
        return False

    try:
        ipaddress.ip_address(client_ip)
    except ValueError:
        logger.warning("Invalid IP address format: %s", client_ip)
        return False

    client_segments = client_ip.split(".")

    for pattern in whitelist:
        if pattern == client_ip:
            return True

        pattern_segments = pattern.split(".")
        if len(pattern_segments) != len(client_segments):
            continue

        matches = True
        for p_seg, c_seg in zip(pattern_segments, client_segments, strict=False):
            if p_seg.lower() in ("x", "*"):
                continue
            if p_seg != c_seg:
                matches = False
                break

        if matches:
            return True

    return False


def require_ip_whitelist[**P, R: Response | object](
    func: Callable[P, R],
) -> Callable[P, Response | tuple[Response, int] | R]:
    """Decorator to restrict access to whitelisted IP addresses.

    Checks the client IP against the whitelist defined in the application
    configuration. Relies on WSGI environment remote address.
    """

    @wraps(func)
    def wrapper(
        *args: P.args, **kwargs: P.kwargs
    ) -> Response | tuple[Response, int] | R:
        configuration = current_app.config.get("APP_CONFIG")

        if not configuration:
            logger.error("❌ SECURITY: APP_CONFIG missing! Denying access.")
            return jsonify(
                {
                    "status": "error",
                    "message": "Security Configuration Error",
                }
            ), 500

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

        client_ip = request.remote_addr or "0.0.0.0"  # nosec B104

        if not _is_ip_whitelisted(client_ip, whitelist):
            if mode == "block":
                logger.warning(
                    "🛡️ SECURITY: Unauthorized IP blocked: %s (Remote: %s)",
                    client_ip,
                    request.remote_addr,
                )
                return jsonify(
                    {
                        "status": "error",
                        "message": "Unauthorized Access",
                    }
                ), 403

            logger.warning(
                "⚠️ SECURITY: Unauthorized IP warning: %s (Remote: %s)",
                client_ip,
                request.remote_addr,
            )

        return func(*args, **kwargs)

    return wrapper
