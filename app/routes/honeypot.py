import logging

from flask import Blueprint, render_template, request

logger = logging.getLogger(__name__)
honeypot_bp = Blueprint("honeypot", __name__)


def _process_honeypot_request(route_path: str) -> tuple[str, int]:
    """Core handler for honeypot endpoints.

    Extracts and logs connection details. If a POST request is made,
    it logs any submitted form data before returning a generic failure.
    """
    client_ip = request.remote_addr or "Unknown IP"

    if request.method == "POST":
        attempted_user = request.form.get("username", "")
        attempted_pass = request.form.get("password", "")

        logger.warning(
            "🍯 HONEYPOT TRIGGERED [%s] | IP: %s | Action: POST | Auth Attempt -> User: '%s', Pass: '%s'",
            route_path,
            client_ip,
            attempted_user,
            attempted_pass,
        )

        return "Authentication failed or account locked.", 401

    logger.info(
        "🍯 HONEYPOT SCANNED [%s] | IP: %s | Action: GET", route_path, client_ip
    )
    return render_template("honeypot_login.html"), 200


@honeypot_bp.route("/login", methods=["GET", "POST"])
def route_honeypot_login() -> tuple[str, int]:
    """Catches generic /login brute-force and scanner attempts."""
    return _process_honeypot_request("/login")


@honeypot_bp.route("/admin", methods=["GET", "POST"])
def route_honeypot_admin() -> tuple[str, int]:
    """Catches generic /admin brute-force and scanner attempts."""
    return _process_honeypot_request("/admin")
