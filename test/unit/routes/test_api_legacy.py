# filename: test_api_legacy.py
import threading
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.database.session import DatabaseSession


@pytest.fixture
def app() -> Flask:
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
def client(app: Flask) -> FlaskClient:
    return app.test_client()


def test_health_check(client: FlaskClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_webhook_ingest(client: FlaskClient) -> None:
    payload = {"symbol": "AAPL", "signal": "LONG"}
    with (
        patch("app.routes.api.SignalRepository") as mock_repo,
        patch("app.routes.api.DatabaseSession", spec=DatabaseSession),
    ):
        mock_instance = mock_repo.return_value
        mock_instance.save_signal.return_value = 1

        response = client.post(
            "/webhook", json=payload, environ_base={"REMOTE_ADDR": "127.0.0.1"}
        )
    assert response.status_code == 201


def test_screener_run_all(client: FlaskClient, app: Flask) -> None:
    mock_engine = app.extensions["screener_engine"]
    mock_engine.run_all.return_value = {"trades_count": 0}

    response = client.post(
        "/screener/run?days=0", environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )
    assert response.status_code == 200
    mock_engine.run_all.assert_called_once()


def test_trades_backfill(client: FlaskClient, app: Flask) -> None:
    mock_tm = app.extensions["trade_manager"]

    response = client.post(
        "/trades/backfill", environ_base={"REMOTE_ADDR": "127.0.0.1"}
    )
    assert response.status_code == 200
    mock_tm.run_daily_process.assert_called_once()


def test_market_sync(client: FlaskClient) -> None:
    with (
        patch("app.routes.api.DatabaseSession", spec=DatabaseSession),
        patch("app.routes.api.MarketDataUpdater") as mock_updater_class,
        patch("app.routes.api.MarketQualityService") as mock_quality_class,
        patch("app.routes.api.Thread") as mock_thread_class,
    ):
        response = client.post(
            "/market/sync", environ_base={"REMOTE_ADDR": "127.0.0.1"}
        )
        assert response.status_code == 202
        assert response.json["status"] == "accepted"

        target_fn = cast(Any, mock_thread_class.call_args.kwargs.get("target"))
        thread_error: Exception | None = None

        def run_in_thread() -> None:
            nonlocal thread_error
            try:
                if target_fn:
                    target_fn()
            except Exception as exc:
                thread_error = exc

        worker = threading.Thread(target=run_in_thread)
        worker.start()
        worker.join(timeout=5)

        assert thread_error is None
        mock_updater_class.return_value.run_update.assert_called_once_with(
            full_reload=False, provider_mode="auto", ignore_today=False
        )
        mock_quality_class.return_value.perform_gap_check.assert_called_once()
        mock_quality_class.return_value.check_last_trading_day_completeness.assert_called_once()


def test_market_reload(client: FlaskClient) -> None:
    with (
        patch("app.routes.api.DatabaseSession", spec=DatabaseSession),
        patch("app.routes.api.MarketDataUpdater") as mock_updater_class,
        patch("app.routes.api.MarketQualityService") as mock_quality_class,
        patch("app.routes.api.Thread") as mock_thread_class,
    ):
        response = client.post(
            "/market/reload", environ_base={"REMOTE_ADDR": "127.0.0.1"}
        )
        assert response.status_code == 200
        assert response.json["status"] == "queued"

        target_fn = cast(Any, mock_thread_class.call_args.kwargs.get("target"))
        thread_error: Exception | None = None

        def run_in_thread() -> None:
            nonlocal thread_error
            try:
                if target_fn:
                    target_fn()
            except Exception as exc:
                thread_error = exc

        worker = threading.Thread(target=run_in_thread)
        worker.start()
        worker.join(timeout=5)

        assert thread_error is None
        mock_updater_class.return_value.run_update.assert_called_once_with(
            full_reload=True, provider_mode="auto", ignore_today=False
        )
        mock_quality_class.return_value.perform_gap_check.assert_called_once()
        mock_quality_class.return_value.check_last_trading_day_completeness.assert_called_once()


def test_market_reload_with_custom_provider_and_ignore_today(
    client: FlaskClient,
) -> None:
    with (
        patch("app.routes.api.DatabaseSession", spec=DatabaseSession),
        patch("app.routes.api.MarketDataUpdater") as mock_updater_class,
        patch("app.routes.api.MarketQualityService"),
        patch("app.routes.api.Thread") as mock_thread_class,
    ):
        response = client.post(
            "/market/reload?provider=tradingview&ignore_today=true",
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200

        target_fn = cast(Any, mock_thread_class.call_args.kwargs.get("target"))
        if target_fn:
            target_fn()

        mock_updater_class.return_value.run_update.assert_called_once_with(
            full_reload=True, provider_mode="tradingview", ignore_today=True
        )
