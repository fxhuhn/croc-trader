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

        # Security: Default to remote_addr to prevent spoofing.
        # Only trust X-Forwarded-For if explicitly configured or in a trusted proxy environment.
        client_ip = request.remote_addr

        # Note: We keep X-Forwarded-For support but log it as potentially spoofed if not from a proxy
        if xff := request.headers.getlist("X-Forwarded-For"):
            actual_ip = xff[0].split(",")[0].strip()
            if actual_ip != client_ip:
                 logger.debug(f"IP Mismatch: Remote={client_ip}, XFF={actual_ip}")
                 # In a hardened setup, we would only trust client_ip unless a proxy is known.
        
        if client_ip not in whitelist:
            if mode == "block":
                logger.warning(f"🛡️ SECURITY: Unauthorized IP blocked: {client_ip}")
                return jsonify({"status": "error", "message": "Unauthorized Access"}), 403

            logger.warning(f"⚠️ SECURITY: Unauthorized IP warning: {client_ip} (Allowing in non-blocking mode)")

        return func(*args, **kwargs)

    return wrapper
