"""Prio 0 Baseline & Regression Test Suite for API routes in app/routes/api.py.

This test suite freezes and asserts the existing behavior (IST-Zustand), parameter
parsing, dependency injection, and error boundaries before applying refactoring.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.const import Strategies
from app.routes.api import api_blueprint


@pytest.fixture
def baseline_app() -> Flask:
    """Configures an isolated Flask test application fixture with mock dependencies."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    mock_config = MagicMock()
    mock_config.get_path.side_effect = lambda name: f"/mock/path/{name}.db"
    mock_config.get_db_path.side_effect = lambda name: f"/mock/path/{name}.db"
    app.config["APP_CONFIG"] = mock_config

    app.register_blueprint(api_blueprint, url_prefix="/api")
    return app


# ==============================================================================
# 1. SCREENER DISPATCH & PARAMETER PARSING BASELINE
# ==============================================================================


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_screener_run_all_baseline_success(baseline_app: Flask) -> None:
    """Verifies /api/screener/run delegates to screener_engine.run_all with correct arguments."""
    client = baseline_app.test_client()

    mock_engine = MagicMock()
    mock_engine.run_all.return_value = {"dip_buyer": 2, "tgim": 1}
    baseline_app.extensions["screener_engine"] = mock_engine

    response = client.post("/api/screener/run?days=3&strategy=tgim")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["stats"] == {"dip_buyer": 2, "tgim": 1}
    mock_engine.run_all.assert_called_once_with(days=3, strategy_filter="tgim")


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_screener_run_all_missing_engine_returns_503(baseline_app: Flask) -> None:
    """Verifies /api/screener/run returns 503 when screener_engine is not initialized."""
    client = baseline_app.test_client()
    response = client.post("/api/screener/run")
    assert response.status_code == 503
    assert response.get_json()["error"] == "Screener Engine not initialized"


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@pytest.mark.parametrize(
    "strategy_param,expected_enum",
    [
        ("dip_buyer", Strategies.DipBuyer),
        ("dip-buyer", Strategies.DipBuyer),
        ("dipbuyer", Strategies.DipBuyer),
        ("two_percent", Strategies.TwoPercent),
        ("twopercentstrategy", Strategies.TwoPercent),
        ("croc_setup", Strategies.CrocSetup),
        ("croc_hold", Strategies.HoldTarget),
        ("croc_split", Strategies.SplitTarget),
        ("turnover_timing", Strategies.TurnOverTiming),
        ("turnover timing", Strategies.TurnOverTiming),
        ("ndx_momentum", Strategies.NDXMomentum),
        ("tgim", Strategies.TGIM),
        ("thank_god_its_monday", Strategies.TGIM),
        ("bridge_scout", Strategies.BridgeScout),
        ("bridge-scout", Strategies.BridgeScout),
        ("qqq_eom", Strategies.BridgeScout),
        ("bounce_bandit", Strategies.BounceBandit),
        ("qqq_meanrev", Strategies.BounceBandit),
    ],
)
def test_screener_run_single_strategy_resolution_baseline(
    baseline_app: Flask, strategy_param: str, expected_enum: Strategies
) -> None:
    """Verifies that standard strategy identifiers resolve and invoke strategy.run()."""
    client = baseline_app.test_client()

    mock_engine = MagicMock()
    mock_strategy = MagicMock()
    mock_strategy.run.return_value = 4
    mock_engine.get_strategy.return_value = mock_strategy
    baseline_app.extensions["screener_engine"] = mock_engine

    response = client.post(f"/api/screener/run/{strategy_param}?days=2&date=2026-08-15")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["signals_found"] == 4
    mock_engine.get_strategy.assert_called_once_with(expected_enum)
    mock_strategy.run.assert_called_once_with(days=2, analysis_date="2026-08-15")


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_screener_run_unregistered_strategy_returns_404(baseline_app: Flask) -> None:
    """Verifies unknown strategy names return HTTP 404."""
    client = baseline_app.test_client()
    mock_engine = MagicMock()
    baseline_app.extensions["screener_engine"] = mock_engine

    response = client.post("/api/screener/run/completely_unknown_strategy")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"
    assert "not found" in data["message"]


# ==============================================================================
# 2. BACKFILL DISPATCH & PARAMETER PARSING BASELINE
# ==============================================================================


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_trades_backfill_without_strategy_triggers_daily_process(
    baseline_app: Flask,
) -> None:
    """Verifies POST /api/trades/backfill without strategy parameter triggers daily process."""
    client = baseline_app.test_client()

    mock_tm = MagicMock()
    baseline_app.extensions["trade_manager"] = mock_tm

    response = client.post("/api/trades/backfill")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    mock_tm.run_daily_process.assert_called_once()


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.run_strategy_backfill")
@patch("app.routes.api.DatabaseSession")
def test_trades_backfill_with_strategy_dispatches_simulation(
    mock_session_cls: MagicMock,
    mock_run_backfill: MagicMock,
    baseline_app: Flask,
) -> None:
    """Verifies POST /api/trades/backfill?strategy=tgim dispatches to backfill runner."""
    client = baseline_app.test_client()
    mock_run_backfill.return_value = {"signals_generated": 10, "trades_closed": 8}

    response = client.post(
        "/api/trades/backfill?strategy=tgim&start=2026-01-01&end=2026-06-30&budget=15000&clear=false"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["result"]["signals_generated"] == 10

    mock_run_backfill.assert_called_once()
    call_kwargs = mock_run_backfill.call_args.kwargs
    assert call_kwargs["strategy_name"] == "tgim"
    assert call_kwargs["start_date"] == "2026-01-01"
    assert call_kwargs["end_date"] == "2026-06-30"
    assert call_kwargs["budget"] == 15000.0
    assert call_kwargs["clear_existing"] is False


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.run_strategy_backfill")
@patch("app.routes.api.DatabaseSession")
def test_trades_backfill_strategy_route_parameter_aliases(
    mock_session_cls: MagicMock,
    mock_run_backfill: MagicMock,
    baseline_app: Flask,
) -> None:
    """Verifies alias parameters start_date/start, end_date/end, clear_existing/clear are supported."""
    client = baseline_app.test_client()
    mock_run_backfill.return_value = {"trades_closed": 5}

    response = client.post(
        "/api/trades/backfill/bridge-scout?start_date=2025-06-01&end_date=2025-12-31&clear_existing=true"
    )
    assert response.status_code == 200
    call_kwargs = mock_run_backfill.call_args.kwargs
    assert call_kwargs["strategy_name"] == "bridge_scout"
    assert call_kwargs["start_date"] == "2025-06-01"
    assert call_kwargs["end_date"] == "2025-12-31"
    assert call_kwargs["clear_existing"] is True


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.run_strategy_backfill")
@patch("app.routes.api.DatabaseSession")
def test_trades_backfill_validation_error_returns_400(
    mock_session_cls: MagicMock,
    mock_run_backfill: MagicMock,
    baseline_app: Flask,
) -> None:
    """Verifies ValueError during backfill returns HTTP 400 Bad Request."""
    client = baseline_app.test_client()
    mock_run_backfill.side_effect = ValueError(
        "Unknown strategy for backfill: 'unknown_strat'"
    )

    response = client.post("/api/trades/backfill/unknown_strat")
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Unknown strategy for backfill" in data["message"]


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.run_strategy_backfill")
@patch("app.routes.api.DatabaseSession")
def test_trades_backfill_server_error_returns_500(
    mock_session_cls: MagicMock,
    mock_run_backfill: MagicMock,
    baseline_app: Flask,
) -> None:
    """Verifies unexpected exceptions during backfill return HTTP 500 Internal Server Error."""
    client = baseline_app.test_client()
    mock_run_backfill.side_effect = RuntimeError("Database disk failure")

    response = client.post("/api/trades/backfill/tgim")
    assert response.status_code == 500
    data = response.get_json()
    assert data["status"] == "error"
    assert "Database disk failure" in data["message"]


# ==============================================================================
# 3. PIPELINE & ORDER GENERATION BASELINE
# ==============================================================================


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.run_daily_eod_pipeline")
def test_eod_pipeline_run_baseline(
    mock_eod_pipeline: MagicMock, baseline_app: Flask
) -> None:
    """Verifies /api/pipeline/run calls synchronous EOD pipeline orchestrator."""
    client = baseline_app.test_client()
    mock_eod_pipeline.return_value = {
        "status": "success",
        "steps_completed": ["trade_manager", "screener", "orders"],
    }

    response = client.post("/api/pipeline/run")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["steps_completed"]) == 3
    mock_eod_pipeline.assert_called_once()


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_order_generation_baseline_success_and_empty(baseline_app: Flask) -> None:
    """Verifies /api/orders/generate returns 201 on file output and 200 when empty."""
    client = baseline_app.test_client()

    mock_tm = MagicMock()
    baseline_app.extensions["trade_manager"] = mock_tm

    # Case 1: File created -> HTTP 201
    mock_tm.generate_daily_orders.return_value = "/data/orders/orders_2026_08_28.csv"
    res1 = client.post("/api/orders/generate")
    assert res1.status_code == 201
    assert res1.get_json()["file"] == "/data/orders/orders_2026_08_28.csv"

    # Case 2: No orders -> HTTP 200
    mock_tm.generate_daily_orders.return_value = None
    res2 = client.post("/api/orders/generate")
    assert res2.status_code == 200
    assert res2.get_json()["message"] == "No orders generated"


# ==============================================================================
# 4. SINGLE-SYMBOL DEBUG ENDPOINTS BASELINE
# ==============================================================================


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_debug_single_symbol_endpoints_baseline(baseline_app: Flask) -> None:
    """Verifies single symbol analyzer debug routes for DipBuyer and Turnover."""
    client = baseline_app.test_client()

    mock_engine = MagicMock()
    mock_dip = MagicMock()
    mock_dip.analyze_single_symbol.return_value = {
        "symbol": "AAPL",
        "conditions_met": True,
    }
    mock_turn = MagicMock()
    mock_turn.analyze_single_symbol.return_value = {
        "symbol": "MSFT",
        "conditions_met": False,
    }

    def get_strat(strat_enum: Any) -> Any:
        if strat_enum == Strategies.DipBuyer:
            return mock_dip
        if strat_enum == Strategies.TurnOverTiming:
            return mock_turn
        return None

    mock_engine.get_strategy.side_effect = get_strat
    baseline_app.extensions["screener_engine"] = mock_engine

    # Missing symbol returns 400
    assert client.post("/api/screener/dip-buyer").status_code == 400
    assert client.post("/api/screener/turnover").status_code == 400

    # Query param symbol
    res_dip = client.post("/api/screener/dip-buyer?symbol=AAPL")
    assert res_dip.status_code == 200
    assert res_dip.get_json()["symbol"] == "AAPL"

    # JSON payload symbol
    res_turn = client.post("/api/screener/turnover", json={"symbol": "MSFT"})
    assert res_turn.status_code == 200
    assert res_turn.get_json()["symbol"] == "MSFT"


# ==============================================================================
# 5. MARKET SYNC & RELOAD PARAMETER INGESTION BASELINE
# ==============================================================================


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
@patch("app.routes.api.Thread")
def test_market_sync_parameter_variations(
    mock_thread_cls: MagicMock, baseline_app: Flask
) -> None:
    """Verifies market sync route accepts parameters via query string and JSON body."""
    client = baseline_app.test_client()

    # Query string parameters
    res1 = client.post("/api/market/sync?full=true&provider=tv&ignore_today=true")
    assert res1.status_code == 202
    assert res1.get_json()["status"] == "accepted"

    # JSON body parameters
    res2 = client.post(
        "/api/market/sync",
        json={"provider": "yahoo", "ignore_today": False},
    )
    assert res2.status_code == 202
    assert res2.get_json()["status"] == "accepted"


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_market_sync_and_reload_concurrency_guard_returns_409(
    baseline_app: Flask,
) -> None:
    """Verifies that market sync/reload returns 409 when synchronization is already active."""
    from app.routes.api import _market_sync_lock

    client = baseline_app.test_client()
    _market_sync_lock.acquire()
    try:
        res_sync = client.post("/api/market/sync")
        assert res_sync.status_code == 409
        data_sync = res_sync.get_json()
        assert data_sync["status"] == "error"
        assert "already in progress" in data_sync["message"]

        res_reload = client.post("/api/market/reload")
        assert res_reload.status_code == 409
        data_reload = res_reload.get_json()
        assert data_reload["status"] == "error"
        assert "already in progress" in data_reload["message"]
    finally:
        _market_sync_lock.release()


@patch("app.routes.api.require_ip_whitelist", lambda f: f)
def test_unified_debug_strategy_endpoint(baseline_app: Flask) -> None:
    """Verifies generic /api/screener/<strategy_name>/debug dispatches dynamically."""
    client = baseline_app.test_client()

    mock_engine = MagicMock()
    mock_strat = MagicMock()
    mock_strat.analyze_single_symbol.return_value = {
        "symbol": "AAPL",
        "score": 95,
    }
    mock_engine.get_strategy.return_value = mock_strat
    baseline_app.extensions["screener_engine"] = mock_engine

    # 1. Symbol-level debug for strategy with alias
    res1 = client.post("/api/screener/dipbuyer/debug", json={"symbol": "AAPL"})
    assert res1.status_code == 200
    assert res1.get_json()["symbol"] == "AAPL"
    mock_strat.analyze_single_symbol.assert_called_once_with("AAPL")

    # 2. Recommendations-level debug
    mock_croc = MagicMock(spec=["get_all_recommendations"])
    mock_croc.get_all_recommendations.return_value = [{"symbol": "NVDA"}]
    mock_engine.get_strategy.return_value = mock_croc

    res2 = client.post("/api/screener/croc_setup/debug?days=5&date=2026-08-01")
    assert res2.status_code == 200
    data2 = res2.get_json()
    assert data2["status"] == "success"
    assert len(data2["signals"]) == 1

    # 3. Unknown strategy -> 404
    mock_engine.get_strategy.return_value = None
    res_unknown = client.post("/api/screener/completely_unknown/debug")
    assert res_unknown.status_code == 404
