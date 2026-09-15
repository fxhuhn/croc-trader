import ipaddress
import logging
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, overload

from flask import Response, current_app, jsonify, request

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R", bound=Response | object)


def _matches_cidr_pattern(
    client_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    pattern: str,
) -> bool:
    """Matches an IP address against a CIDR network pattern."""
    if "/" not in pattern:
        return False
    try:
        return client_ip in ipaddress.ip_network(pattern, strict=False)
    except ValueError:
        return False


def _matches_wildcard_pattern(
    client_segments: list[str],
    pattern: str,
) -> bool:
    """Matches IP segments against a wildcard pattern with 'x' or '*'."""
    pattern_segments = pattern.split(".")
    if len(pattern_segments) != len(client_segments):
        return False

    for pattern_segment, client_segment in zip(
        pattern_segments, client_segments, strict=False
    ):
        if pattern_segment.lower() in ("x", "*"):
            continue
        if pattern_segment != client_segment:
            return False

    return True


def _is_ip_whitelisted(client_ip: str, whitelist: list[str] | tuple[str, ...]) -> bool:
    """Checks if a given IP address matches any whitelist entry or wildcard pattern.

    Supports:
    - Exact IP matches (e.g. "127.0.0.1")
    - CIDR subnet notation (e.g. "192.168.1.0/24", "172.16.0.0/16")
    - Wildcards in any segment (e.g. "172.16.x.x", "10.0.*.*", "8.*.8.8", "172.16.X.X")
    """
    if not whitelist:
        return False

    try:
        parsed_client_ip = ipaddress.ip_address(client_ip)
    except ValueError:
        logger.warning("Invalid IP address format: %s", client_ip)
        return False

    client_segments = client_ip.split(".")

    for pattern in whitelist:
        if pattern == client_ip:
            return True
        if _matches_cidr_pattern(parsed_client_ip, pattern):
            return True
        if _matches_wildcard_pattern(client_segments, pattern):
            return True

    return False


@overload
def require_ip_whitelist[**P, R: Response | object](
    func: Callable[P, R],
) -> Callable[P, Response | tuple[Response, int] | R]: ...


@overload
def require_ip_whitelist[**P, R: Response | object](
    func: None = None,
    *,
    strict: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, Response | tuple[Response, int] | R]]: ...


def require_ip_whitelist[**P, R: Response | object](
    func: Callable[P, R] | None = None,
    *,
    strict: bool = False,
) -> (
    Callable[P, Response | tuple[Response, int] | R]
    | Callable[[Callable[P, R]], Callable[P, Response | tuple[Response, int] | R]]
):
    """Decorator to restrict access to whitelisted IP addresses.

    Checks the client IP against the whitelist defined in the application
    configuration. Relies on WSGI environment remote address.

    When strict=True, access from non-whitelisted IPs is immediately rejected
    with HTTP 403 Forbidden even if security.mode is set to 'warning'.
    """

    def decorator(
        target_func: Callable[P, R],
    ) -> Callable[P, Response | tuple[Response, int] | R]:
        @wraps(target_func)
        def wrapper(
            *args: P.args, **kwargs: P.kwargs
        ) -> Response | tuple[Response, int] | R:
            configuration = current_app.config.get("APP_CONFIG")

            if not configuration:
                logger.error("❌ SECURITY: APP_CONFIG missing! Denying access.")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Security Configuration Error",
                        }
                    ),
                    500,
                )

            try:
                security_configuration = configuration.app.security
                whitelist = security_configuration.whitelist
                mode = security_configuration.mode
            except AttributeError:
                logger.error(
                    "❌ SECURITY: Security configuration missing! Denying access."
                )
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Security Configuration Error",
                        }
                    ),
                    500,
                )

            client_ip = request.remote_addr or "0.0.0.0"  # nosec B104

            if not _is_ip_whitelisted(client_ip, whitelist):
                if strict or mode == "block":
                    logger.warning(
                        "🛡️ SECURITY: Unauthorized IP blocked: %s (Remote: %s, Strict: %s)",
                        client_ip,
                        request.remote_addr,
                        strict,
                    )
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Unauthorized Access",
                            }
                        ),
                        403,
                    )

                logger.warning(
                    "⚠️ SECURITY: Unauthorized IP warning: %s (Remote: %s)",
                    client_ip,
                    request.remote_addr,
                )

            return target_func(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
