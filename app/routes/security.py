import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import current_app, jsonify, request

logger = logging.getLogger(__name__)


def require_ip_whitelist(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        conf = current_app.config["APP_CONFIG"]

        try:
            security_conf = conf.app.security
            whitelist = security_conf.whitelist
            mode = security_conf.mode
        except AttributeError:
            logger.error("❌ SECURITY: Security config missing! Denying access (Fail-Secure).")
            return jsonify({"status": "error", "message": "Security Configuration Error"}), 500

        # Proxy-Aware IP Detection:
        # 1. Use X-Forwarded-For if available (trusted proxy environment)
        # 2. Fallback to remote_addr
        xff = request.headers.getlist("X-Forwarded-For")
        client_ip = xff[0].split(",")[0].strip() if xff else request.remote_addr

        if client_ip not in whitelist:
            if mode == "block":
                logger.warning(f"🛡️ SECURITY: Unauthorized IP blocked: {client_ip} (Remote: {request.remote_addr})")
                return jsonify({"status": "error", "message": "Unauthorized Access"}), 403

            logger.warning(
                f"⚠️ SECURITY: Unauthorized IP warning: {client_ip} (Remote: {request.remote_addr}) "
                f"(Allowing in non-blocking mode)"
            )

        return func(*args, **kwargs)

    return wrapper
