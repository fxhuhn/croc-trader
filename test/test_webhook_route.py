import pytest
from unittest.mock import MagicMock, patch
from flask import Flask

# Import the Blueprint directly to register it on a minimal app for isolated testing
# OR import the full factory. Using the full factory checks dependency wiring,
# but might be heavy. Let's use specific patching on the full app for Integration-like Unit tests.
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
    # Create app using the factory
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    return app

@pytest.fixture
def client(app_instance):
    return app_instance.test_client()

# --- TESTS ---

def test_ingest_webhook_success(client, mock_repo_class, mock_db_session_class):
    """
    Test standard success scenario:
    - Valid JSON payload
    - Repo saves successfully
    - Returns 201 + ID
    """
    # Setup Mock
    mock_instance = mock_repo_class.return_value
    mock_instance.save_signal.return_value = "msg-12345"
    
    payload = {
        "symbol": "AAPL",
        "signal": "BUY",
        "price": 150.0
    }
    
    # Execute
    response = client.post("/webhook", json=payload)
    
    # Verify Response
    assert response.status_code == 201
    assert response.json["status"] == "success"
    assert response.json["id"] == "msg-12345"
    
    # Verify Logic
    mock_db_session_class.assert_called()  # Session created
    mock_instance.save_signal.assert_called_once_with(payload)

def test_ingest_webhook_empty_payload(client):
    """
    Test validation logic:
    - Empty body or non-JSON content
    - Returns 400 Bad Request
    """
    # 1. No Data
    response = client.post("/webhook")
    assert response.status_code == 400
    assert "Empty Payload" in response.json["message"]
    
    # 2. Empty JSON
    response = client.post("/webhook", json={})
    assert response.status_code == 400
    assert "Empty Payload" in response.json["message"]

def test_ingest_webhook_internal_error(client, mock_repo_class):
    """
    Test error handling:
    - Repo raises Exception
    - Returns 500 Internal Error
    """
    # Setup Mock to crash
    mock_instance = mock_repo_class.return_value
    mock_instance.save_signal.side_effect = Exception("DB Connection Failed")
    
    payload = {"symbol": "BTC"}
    
    # Execute
    response = client.post("/webhook", json=payload)
    
    # Verify
    assert response.status_code == 500
    assert response.json["status"] == "error"
    assert "DB Connection Failed" in response.json["message"]
