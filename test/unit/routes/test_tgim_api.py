"""Unit tests for TGIM API backfill endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.routes.api import api_blueprint


@pytest.fixture
def app() -> Flask:
    """Fixture providing a configured Flask application."""
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.get_path.side_effect = lambda name: f"mock_{name}.db"
    test_app.config["APP_CONFIG"] = mock_config

    test_app.register_blueprint(api_blueprint, url_prefix="/api")
    return test_app


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.services.tgim_backfill.run_tgim_backfill")
def test_tgim_backfill_api_success(mock_run_backfill: MagicMock, app: Flask) -> None:
    """Tests POST /api/trades/backfill/tgim executes successfully with query parameters."""
    mock_run_backfill.return_value = {
        "start_date": "2026-01-01",
        "end_date": "2026-07-24",
        "signals_generated": 6,
        "trades_filled": 6,
        "trades_closed": 6,
        "total_pnl": 567.37,
        "win_rate": 100.0,
        "closed_trades": [],
    }

    client = app.test_client()
    response = client.post(
        "/api/trades/backfill/tgim?start_date=2026-01-01&budget=10000.0"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["result"]["trades_closed"] == 6
    assert data["result"]["win_rate"] == 100.0


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.services.tgim_backfill.run_tgim_backfill")
def test_tgim_backfill_api_url_params(mock_run_backfill: MagicMock, app: Flask) -> None:
    """Tests POST /api/trades/backfill/tgim with short URL query parameters."""
    mock_run_backfill.return_value = {
        "start_date": "2026-01-01",
        "end_date": "2026-07-24",
        "signals_generated": 6,
        "trades_filled": 6,
        "trades_closed": 6,
        "total_pnl": 567.37,
        "win_rate": 100.0,
        "closed_trades": [],
    }

    client = app.test_client()
    response = client.post("/api/trades/backfill/tgim?start=2026-01-01&budget=10000.0")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    mock_run_backfill.assert_called_once()
    call_kwargs = mock_run_backfill.call_args.kwargs
    assert call_kwargs["start_date"] == "2026-01-01"
    assert call_kwargs["budget"] == 10000.0


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.services.tgim_backfill.run_tgim_backfill")
def test_tgim_backfill_api_generic_route(
    mock_run_backfill: MagicMock, app: Flask
) -> None:
    """Tests POST /api/trades/backfill?strategy=tgim dispatches correctly."""
    mock_run_backfill.return_value = {
        "start_date": "2026-01-01",
        "end_date": "2026-07-24",
        "signals_generated": 6,
        "trades_filled": 6,
        "trades_closed": 6,
        "total_pnl": 567.37,
        "win_rate": 100.0,
        "closed_trades": [],
    }

    client = app.test_client()
    response = client.post(
        "/api/trades/backfill?strategy=tgim&start_date=2026-01-01&budget=5000"
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
