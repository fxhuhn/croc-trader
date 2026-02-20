# filename: test_api.py
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def app():
    """Provides the Flask application instance."""
    from app import create_app

    app_instance = create_app()
    app_instance.config["TESTING"] = True
    app_instance.config["APP_CONFIG"] = MagicMock()
    app_instance.config["APP_CONFIG"].app.security.whitelist = ["127.0.0.1"]
    app_instance.config["APP_CONFIG"].app.security.mode = "block"
    app_instance.config["APP_CONFIG"].get_db_path.return_value = ":memory:"

    # Mock Extensions
    app_instance.extensions["screener_engine"] = MagicMock()
    app_instance.extensions["trade_manager"] = MagicMock()
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_webhook_ingest(client):
    payload = {"symbol": "AAPL", "signal": "LONG"}
    # Patching both Repo and Session to avoid DB hits
    with (
        patch("app.routes.api.SignalRepository") as mock_repo,
        patch("app.routes.api.DatabaseSession"),
    ):
        mock_instance = mock_repo.return_value
        mock_instance.save_signal.return_value = 1

        response = client.post(
            "/webhook", json=payload, environ_base={"REMOTE_ADDR": "127.0.0.1"}
        )
    assert response.status_code == 201


def test_screener_run_all(client, app):
    mock_engine = app.extensions["screener_engine"]
    mock_engine.run_all.return_value = {"trades_count": 0}

    response = client.post(
        "/screener/run?days=0", environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )
    assert response.status_code == 200
    mock_engine.run_all.assert_called_once()


def test_trades_backfill(client, app):
    mock_tm = app.extensions["trade_manager"]

    response = client.post(
        "/trades/backfill", environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )
    assert response.status_code == 200
    mock_tm.run_daily_process.assert_called_once()
