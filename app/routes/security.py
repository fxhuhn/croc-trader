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

        # EAFP: Falls Config noch nicht vollständig geladen, Zugriff gewähren
        try:
            security_conf = conf.app.security
            whitelist = security_conf.whitelist
            mode = security_conf.mode
        except AttributeError:
            logger.warning("Security config not found. Allowing access.")   
            return func(*args, **kwargs)

        client_ip = request.remote_addr

        # Proxy Support
        if xff := request.headers.getlist("X-Forwarded-For"):
            client_ip = xff[0].split(",")[0].strip()
        elif real_ip := request.headers.get("X-Real-IP"):
            client_ip = real_ip.strip()

        if client_ip not in whitelist:
            if mode == "block":
                logger.warning(f"Unauthorized IP blocked: {client_ip}")
                return jsonify({"status": "error", "message": "Unauthorized IP"}), 403

            logger.warning(f"Unauthorized IP warning: {client_ip}")

        return func(*args, **kwargs)

    return wrapper
