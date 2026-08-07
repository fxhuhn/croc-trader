"""Comprehensive unit tests for Prio 3 REST API routes in app/routes/api.py."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.const import Strategies
from app.routes.api import _parse_boolean_parameter, api_blueprint


@pytest.fixture
def api_app() -> Flask:
    """Configured Flask test app fixture for API testing."""
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.get_path.side_effect = lambda name: f"mock_{name}.db"
    mock_config.get_db_path.side_effect = lambda name: f"mock_{name}.db"
    test_app.config["APP_CONFIG"] = mock_config

    test_app.register_blueprint(api_blueprint, url_prefix="/api")
    return test_app


def test_parse_boolean_parameter() -> None:
    assert _parse_boolean_parameter(None, True) is True
    assert _parse_boolean_parameter(None, False) is False
    assert _parse_boolean_parameter(True) is True
    assert _parse_boolean_parameter(False) is False
    assert _parse_boolean_parameter("true") is True
    assert _parse_boolean_parameter("false") is False
    assert _parse_boolean_parameter("0") is False


def test_health_check(api_app: Flask) -> None:
    client = api_app.test_client()
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_root_check(api_app: Flask) -> None:
    client = api_app.test_client()
    res = client.get("/api/")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_webhook_ingestion_validation(api_app: Flask) -> None:
    client = api_app.test_client()

    # Missing payload
    res1 = client.post("/api/webhook", data="invalid raw json")
    assert res1.status_code == 400
    assert res1.get_json()["message"] == "Invalid JSON"

    # Missing symbol
    res2 = client.post("/api/webhook", json={"timeframe": "1d"})
    assert res2.status_code == 400
    assert "Missing mandatory field: symbol" in res2.get_json()["message"]


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.SignalRepository")
@patch("app.routes.api.DatabaseSession")
def test_webhook_ingestion_success(
    mock_session_cls: MagicMock, mock_repo_cls: MagicMock, api_app: Flask
) -> None:
    mock_repo = mock_repo_cls.return_value
    mock_repo.save_signal.return_value = 42

    client = api_app.test_client()
    res = client.post("/api/webhook", json={"symbol": "AAPL", "signal": "DipBuyer"})
    assert res.status_code == 201
    data = res.get_json()
    assert data["status"] == "success"
    assert data["id"] == 42


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_trigger_screener_all(api_app: Flask) -> None:
    client = api_app.test_client()

    # Engine missing
    res1 = client.post("/api/screener/run")
    assert res1.status_code == 503

    # Engine present
    mock_engine = MagicMock()
    mock_engine.run_all.return_value = {"signals": 5}
    api_app.extensions["screener_engine"] = mock_engine

    res2 = client.post("/api/screener/run?days=5&strategy=dip_buyer")
    assert res2.status_code == 200
    assert res2.get_json()["stats"] == {"signals": 5}


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_run_strategy_screener_endpoint(api_app: Flask) -> None:
    client = api_app.test_client()

    # Unknown strategy
    api_app.extensions["screener_engine"] = MagicMock()
    res_unknown = client.post("/api/screener/run/unknown_strat")
    assert res_unknown.status_code == 404

    # Known strategy, missing in engine
    mock_engine = MagicMock()
    mock_engine.get_strategy.return_value = None
    api_app.extensions["screener_engine"] = mock_engine

    res_missing = client.post("/api/screener/run/tgim")
    assert res_missing.status_code == 404

    # Success
    mock_strat = MagicMock()
    mock_strat.run.return_value = 3
    mock_engine.get_strategy.return_value = mock_strat

    res_ok = client.post("/api/screener/run/tgim?date=2026-08-01")
    assert res_ok.status_code == 200
    assert res_ok.get_json()["signals_found"] == 3


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_single_symbol_analyzers(api_app: Flask) -> None:
    client = api_app.test_client()

    # Dip buyer missing symbol
    res_no_sym = client.post("/api/screener/dip-buyer")
    assert res_no_sym.status_code == 400

    # Dip buyer success
    mock_engine = MagicMock()
    mock_strat = MagicMock()
    mock_strat.analyze_single_symbol.return_value = {"symbol": "AAPL", "valid": True}
    mock_engine.get_strategy.return_value = mock_strat
    api_app.extensions["screener_engine"] = mock_engine

    res_dip = client.post("/api/screener/dip-buyer?symbol=AAPL")
    assert res_dip.status_code == 200
    assert res_dip.get_json()["symbol"] == "AAPL"

    # Turnover timing success
    res_turn = client.post("/api/screener/turnover?symbol=MSFT")
    assert res_turn.status_code == 200


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_analyze_croc_and_ndx_momentum(api_app: Flask) -> None:
    client = api_app.test_client()

    mock_engine = MagicMock()
    mock_croc_strat = MagicMock()
    mock_croc_strat.get_all_recommendations.return_value = [{"symbol": "NVDA"}]

    mock_ndx_strat = MagicMock()
    mock_ndx_strat.calculate_analysis.return_value = {
        "date": "2026-08-01",
        "top_symbols": ["QQQ"],
    }

    def get_strat_side_effect(enum_val: Any) -> Any:
        if enum_val == Strategies.CrocSetup:
            return mock_croc_strat
        if enum_val == Strategies.NDXMomentum:
            return mock_ndx_strat
        return None

    mock_engine.get_strategy.side_effect = get_strat_side_effect
    api_app.extensions["screener_engine"] = mock_engine

    res_croc = client.post("/api/screener/croc?days=10")
    assert res_croc.status_code == 200
    assert len(res_croc.get_json()["signals"]) == 1

    res_ndx = client.post("/api/screener/ndx-momentum")
    assert res_ndx.status_code == 200
    assert res_ndx.get_json()["top_leaders"] == ["QQQ"]


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_trigger_orders(api_app: Flask) -> None:
    client = api_app.test_client()

    # Missing trade manager
    res1 = client.post("/api/orders/generate")
    assert res1.status_code == 500

    # Trade manager present
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.return_value = "/data/orders/orders.csv"
    api_app.extensions["trade_manager"] = mock_tm

    res2 = client.post("/api/orders/generate")
    assert res2.status_code == 201
    assert res2.get_json()["file"] == "/data/orders/orders.csv"


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_trades_backfill_daily_process(api_app: Flask) -> None:
    client = api_app.test_client()

    mock_tm = MagicMock()
    api_app.extensions["trade_manager"] = mock_tm

    res = client.post("/api/trades/backfill")
    assert res.status_code == 200
    mock_tm.run_daily_process.assert_called_once()


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.Thread")
def test_market_sync_and_reload(mock_thread_cls: MagicMock, api_app: Flask) -> None:
    client = api_app.test_client()

    res_sync = client.post("/api/market/sync?full=true&provider=yahoo")
    assert res_sync.status_code == 202
    assert res_sync.get_json()["status"] == "accepted"

    res_reload = client.post("/api/market/reload?ignore_today=true")
    assert res_reload.status_code == 200
    assert res_reload.get_json()["status"] == "queued"
