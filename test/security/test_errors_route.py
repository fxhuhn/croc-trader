import logging
from unittest.mock import patch

import pytest
from flask import Flask

from app.routes.errors import errors_bp


@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(errors_bp)

    # Mock render_template to avoid TemplateNotFound in tests (path issues)
    # We patch it at the module level where it's imported in errors.py
    with patch("app.routes.errors.render_template", return_value="<html>Mocked</html>"):
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_404_log_injection_prevention(client, caplog):
    """
    SECURITY: Verifies that newline characters in the path do not cause log injection.
    """
    caplog.set_level(logging.WARNING)

    # Attack payload: Path with newline and fake log entry
    malicious_path = "/nonexistent\nINFO: User admin logged in"

    client.get(malicious_path)

    # Assert
    found_log = False
    for record in caplog.records:
        if "404 Not Found" in record.message:
            found_log = True
            # Check for raw newline injection
            assert "\nINFO:" not in record.message
            assert "\r" not in record.message

            # Additional check: verify the path was actually sanitized/stripped
            # The code does replace("\n", "") so "nonexistent\nINFO" becomes "nonexistentINFO"
            assert "nonexistentINFO" in record.message

    assert found_log, "404 Warning log not found"


def test_500_info_leak_prevention(app):
    """
    SECURITY: Verifies that 500 error responses do not leak exception details to API clients.
    """
    # CRITICAL: Disable exception propagation to let the error handler run
    app.config["PROPAGATE_EXCEPTIONS"] = False

    # Create a route that raises a sensitive exception
    @app.route("/api/trigger_error")
    def trigger_error():
        raise ValueError(
            "Sensitive DB Connection String: postgres://user:pass@db:5432/secret"
        )

    client = app.test_client()

    # Act: Call the API endpoint
    response = client.get("/api/trigger_error")

    # Assert
    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert data["message"] == "Internal Server Error"

    # CRITICAL: Ensure the sensitive string is NOT in the response
    assert "Sensitive DB Connection String" not in str(data)
    assert "postgres://" not in str(data)
    # Ensure "detail" key is gone
    assert "detail" not in data


def test_html_fallback_works(client):
    """
    Ensure that browser clients still get HTML (mocked).
    """
    # render_template is already mocked in the fixture
    response = client.get("/random-page", headers={"Accept": "text/html"})
    assert response.status_code == 404
    assert b"<html>Mocked</html>" in response.data
