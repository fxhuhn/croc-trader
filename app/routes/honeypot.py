import logging

from flask import Blueprint, request

logger = logging.getLogger(__name__)
honeypot_bp = Blueprint("honeypot", __name__)


# A minimal, deceptively real-looking login form to entice automated
# scanners into submitting credentials. Kept simple to minimize footprint.
_HONEYPOT_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Login Area</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #f4f4f5; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .login-container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); width: 100%; max-width: 320px; }
        .form-group { margin-bottom: 1rem; }
        label { display: block; margin-bottom: 0.5rem; font-size: 0.875rem; color: #374151; }
        input { width: 100%; padding: 0.5rem; border: 1px solid #d1d5db; border-radius: 4px; box-sizing: border-box; }
        button { width: 100%; padding: 0.75rem; background: #2563eb; color: white; border: none; border-radius: 4px; font-weight: 500; cursor: pointer; }
        button:hover { background: #1d4ed8; }
    </style>
</head>
<body>
    <div class="login-container">
        <h2 style="margin-top: 0; color: #111827; text-align: center;">System Access</h2>
        <form method="POST">
            <div class="form-group">
                <label for="username">Username</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>"""


def _process_honeypot_request(route_path: str) -> tuple[str, int]:
    """
    Core handler for honeypot endpoints.

    Extracts and logs connection details. If a POST request is made,
    it aggressively logs any submitted form data (credentials) before
    returning a generic failure to slow down automated traversal.
    """
    client_ip = (
        request.headers.get("X-Forwarded-For", request.remote_addr) or "Unknown IP"
    )

    if request.method == "POST":
        # Extract potential credentials from the body
        attempted_user = request.form.get("username", "")
        attempted_pass = request.form.get("password", "")

        # Log this as a high-priority warning
        logger.warning(
            f"🍯 HONEYPOT TRIGGERED [{route_path}] | IP: {client_ip} | "
            f"Action: POST | Auth Attempt -> User: '{attempted_user}', Pass: '{attempted_pass}'"
        )

        # Return generic error to simulate a failed login
        return "Authentication failed or account locked.", 401

    # Log the GET scan and serve the bait
    logger.info(f"🍯 HONEYPOT SCANNED [{route_path}] | IP: {client_ip} | Action: GET")
    return _HONEYPOT_HTML_TEMPLATE, 200


@honeypot_bp.route("/login", methods=["GET", "POST"])
def route_honeypot_login() -> tuple[str, int]:
    """Catches generic /login brute-force and scanner attempts."""
    return _process_honeypot_request("/login")


@honeypot_bp.route("/admin", methods=["GET", "POST"])
def route_honeypot_admin() -> tuple[str, int]:
    """Catches generic /admin brute-force and scanner attempts."""
    return _process_honeypot_request("/admin")
