# filename: test_webhook_route.py
from unittest.mock import MagicMock, patch

import pytest

from app import create_app


@pytest.fixture
def mock_repo_class():
    with patch("app.routes.api.SignalRepository") as mock:
        yield mock


@pytest.fixture
def mock_db_session_class():
    with patch("app.routes.api.DatabaseSession") as mock:
        yield mock


@pytest.fixture
def app_instance(mock_repo_class, mock_db_session_class):
    """
    Creates the app with mocked DB dependencies to prevent side effects.
    """
    app = create_app()
    app.config.update({"TESTING": True})
    # Whitelist local IP for tests
    app.config["APP_CONFIG"] = MagicMock()
    app.config["APP_CONFIG"].app.security.whitelist = ["127.0.0.1"]
    app.config["APP_CONFIG"].app.security.mode = "block"
    return app


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


# --- TESTS ---


def test_ingest_webhook_success(client, mock_repo_class, mock_db_session_class):
    """Test standard success scenario."""
    mock_instance = mock_repo_class.return_value
    mock_instance.save_signal.return_value = "msg-12345"

    payload = {"symbol": "AAPL", "signal": "BUY", "price": 150.0}

    response = client.post(
        "/webhook", json=payload, environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )

    assert response.status_code == 201
    assert response.json["status"] == "success"
    assert response.json["id"] == "msg-12345"


def test_ingest_webhook_invalid_json(client):
    """Test validation logic for invalid JSON."""
    response = client.post(
        "/webhook",
        data="not a json",
        content_type="application/json",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert response.status_code == 400
    assert "Invalid JSON" in response.json["message"]


def test_ingest_webhook_internal_error(client, mock_repo_class):
    """Test error shielding: generic message should be returned."""
    mock_instance = mock_repo_class.return_value
    mock_instance.save_signal.side_effect = Exception("DB Connection Failed")

    payload = {"symbol": "BTC"}

    response = client.post(
        "/webhook", json=payload, environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )

    # Internal Error Shielding: Generic message "Internal Server Error"
    assert response.status_code == 500
    assert response.json["status"] == "error"
    assert "Internal Server Error" in response.json["message"]
    assert "error_id" in response.json
