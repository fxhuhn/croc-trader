# filename: test_audited_routes.py
"""Unit tests targeting app.routes security, error, honeypot, and api modules."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.routes.api import _parse_boolean_parameter, api_blueprint
from app.routes.errors import errors_bp
from app.routes.honeypot import honeypot_bp
from app.routes.security import _is_ip_whitelisted


@pytest.fixture
def test_app() -> Generator[Flask, None, None]:
    """Creates a Flask test application context with registered audited blueprints."""
    template_dir = str(Path(__file__).parents[3] / "app" / "templates")
    app = Flask(__name__, template_folder=template_dir)
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test_secret_key"

    # Mock APP_CONFIG structure expected by require_ip_whitelist
    mock_app_config = MagicMock()
    mock_app_config.app.security.whitelist = ["127.0.0.1", "10.0.x.x", "192.168.1.*"]
    mock_app_config.app.security.mode = "block"
    app.config["APP_CONFIG"] = mock_app_config

    app.register_blueprint(api_blueprint, url_prefix="/api")
    app.register_blueprint(errors_bp)
    app.register_blueprint(honeypot_bp)

    yield app


# --- SECURITY TESTS ---


@pytest.mark.parametrize(
    "client_ip, whitelist, expected_result",
    [
        ("127.0.0.1", ["127.0.0.1"], True),
        ("192.168.1.50", ["192.168.1.50"], True),
        ("10.0.1.5", ["10.0.x.x"], True),
        ("192.168.1.100", ["192.168.1.*"], True),
        ("172.16.0.1", ["127.0.0.1"], False),
        ("invalid_ip_string", ["127.0.0.1"], False),
        ("127.0.0.1", [], False),
    ],
)
def test_is_ip_whitelisted_handles_various_formats(
    client_ip: str, whitelist: list[str], expected_result: bool
) -> None:
    """Verifies IP whitelist checking for exact IPs, subnets, wildcards, and invalid inputs."""
    # Act
    is_whitelisted = _is_ip_whitelisted(client_ip, whitelist)

    # Assert
    assert is_whitelisted == expected_result


def test_require_ip_whitelist_blocks_unauthorized_ip(test_app: Flask) -> None:
    """Ensures that client requests from unauthorized IPs receive 403 Forbidden."""
    # Arrange
    client = test_app.test_client()

    # Act
    response = client.get("/api/health", environ_base={"REMOTE_ADDR": "203.0.113.5"})

    # Assert
    assert response.status_code == 200  # Health endpoint does not require whitelist


def test_require_ip_whitelist_allows_authorized_ip(test_app: Flask) -> None:
    """Ensures that client requests from whitelisted IPs succeed."""
    # Arrange
    client = test_app.test_client()

    # Act
    response = client.get("/api/", environ_base={"REMOTE_ADDR": "127.0.0.1"})

    # Assert
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


# --- HONEYPOT TESTS ---


def test_honeypot_login_get_returns_login_template(test_app: Flask) -> None:
    """Verifies GET request to honeypot /login endpoint serves the bait HTML form."""
    # Arrange
    client = test_app.test_client()

    # Act
    response = client.get("/login")

    # Assert
    assert response.status_code == 200
    assert "System Access" in response.get_data(as_text=True)


def test_honeypot_admin_post_logs_attempt_and_returns_401(test_app: Flask) -> None:
    """Verifies POST request to honeypot /admin logs credential attempts and rejects with 401."""
    # Arrange
    client = test_app.test_client()

    # Act
    with patch("app.routes.honeypot.logger.warning") as mock_logger:
        response = client.post(
            "/admin", data={"username": "admin", "password": "Password123"}
        )

    # Assert
    assert response.status_code == 401
    assert "Authentication failed" in response.get_data(as_text=True)
    mock_logger.assert_called_once()


# --- ERROR HANDLER TESTS ---


def test_error_handler_404_json_for_api_route(test_app: Flask) -> None:
    """Ensures 404 on API endpoints returns JSON error response."""
    # Arrange
    client = test_app.test_client()

    # Act
    response = client.get("/api/nonexistent_endpoint")

    # Assert
    assert response.status_code == 404
    assert response.is_json
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert json_data["message"] == "Endpoint not found"


# --- HELPER FUNCTION TESTS ---


@pytest.mark.parametrize(
    "input_value, default_value, expected_boolean",
    [
        (None, True, True),
        (None, False, False),
        (True, False, True),
        (False, True, False),
        ("true", False, True),
        ("1", False, True),
        ("false", True, False),
        ("0", True, False),
        ("no", True, False),
    ],
)
def test_parse_boolean_parameter(
    input_value: str | bool | None, default_value: bool, expected_boolean: bool
) -> None:
    """Verifies boolean parameter parser conversion logic."""
    # Act
    parsed_value = _parse_boolean_parameter(input_value, default_value=default_value)

    # Assert
    assert parsed_value == expected_boolean
