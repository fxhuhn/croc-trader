import logging
import uuid
from threading import Thread
from typing import TypedDict

from flask import Blueprint, Response, current_app, jsonify, request

from ..const import Strategies
from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..services import bounce_bandit_backfill, bridge_scout_backfill, tgim_backfill
from ..services.market.quality import MarketQualityService
from ..services.market.updater import MarketDataUpdater
from .security import require_ip_whitelist
from .views.dependencies import cache

logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)


class WebhookPayload(TypedDict, total=False):
    """Schema for incoming signal webhooks."""

    symbol: str
    ticker: str
    timeframe: str
    signal: str
    strategy: str
    exchange: str
    timestamp: str
    date: str
    price: float


def _parse_boolean_parameter(
    value: str | bool | None, default_value: bool = True
) -> bool:
    """Parses incoming query parameters into booleans using guard clauses."""
    if value is None:
        return default_value
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("false", "0", "no")


def _extract_symbol_from_request() -> str | None:
    """Extracts target symbol from query parameters or JSON body."""
    symbol = request.args.get("symbol")
    if symbol:
        return str(symbol).strip().upper()

    data = request.get_json(force=True, silent=True)
    if data and isinstance(data, dict) and data.get("symbol"):
        return str(data["symbol"]).strip().upper()

    return None


# --- STANDARD ROUTES ---


@api_blueprint.route("/health", methods=["GET"])
def health_check() -> Response:
    """Simple health check endpoint.

    Returns:
        Response: JSON status OK.
    """
    return jsonify({"status": "ok"})


@api_blueprint.route("/", methods=["GET"])
@require_ip_whitelist
def root_check() -> Response:
    """Authenticated root check endpoint.

    Returns:
        Response: JSON status OK.
    """
    return jsonify({"status": "ok"})


# --- SIGNAL INGESTION ---


@api_blueprint.route("/webhook", methods=["POST"])
@require_ip_whitelist
def ingest_webhook() -> Response:
    """Ingests signal webhooks and persists them to the signal database.

    Returns:
        Response: JSON success with signal ID or error message.
    """
    try:
        payload: WebhookPayload | None = request.get_json(silent=True, force=True)

        if not payload:
            raw_data = request.get_data(as_text=True)
            logger.warning("⚠️ Malformed Webhook Data: %s", raw_data)
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        symbol = payload.get("symbol") or payload.get("ticker")
        if not symbol:
            logger.warning(
                "⚠️ Webhook rejected: Missing 'symbol' in payload %s", payload
            )
            return jsonify(
                {
                    "status": "error",
                    "message": "Missing mandatory field: symbol",
                }
            ), 400

        configuration = current_app.config.get("APP_CONFIG")
        database_path = (
            configuration.get_db_path("signals")
            if configuration
            else "instance/signals.db"
        )

        session = DatabaseSession(str(database_path))
        repository = SignalRepository(session)

        signal_id = repository.save_signal(dict(payload))

        logger.info("✅ Webhook saved: %s -> ID %s", symbol, signal_id)

        return jsonify({"status": "success", "id": signal_id}), 201

    except Exception as error:
        error_identifier = str(uuid.uuid4())[:8]
        logger.error(
            "Webhook processing error [%s]: %s",
            error_identifier,
            error,
            exc_info=True,
        )
        return jsonify(
            {
                "status": "error",
                "message": "Internal Server Error",
                "error_id": error_identifier,
            }
        ), 500


# --- SCREENER & TRADING ---


@api_blueprint.route("/screener/run", methods=["POST"])
@require_ip_whitelist
def trigger_screener() -> Response:
    """Triggers a manual run of all active screeners.

    Returns:
        Response: JSON status success with statistics.
    """
    try:
        days_lookback = request.args.get("days", default=0, type=int)
        target_strategy = request.args.get("strategy", type=str)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"error": "Screener Engine not initialized"}), 503

        logger.info(
            "API Trigger: Screener run (days=%d, strategy=%s)",
            days_lookback,
            target_strategy,
        )
        statistics = screener_engine.run_all(
            days=days_lookback, strategy_filter=target_strategy
        )

        return jsonify({"status": "success", "stats": statistics}), 200
    except Exception as error:
        logger.exception("Error during screener run: %s", error)
        return jsonify({"error": str(error)}), 500


@api_blueprint.route("/screener/dip-buyer", methods=["POST"])
@require_ip_whitelist
def analyze_dip_buyer() -> Response:
    """Detailed debugging for DipBuyer strategy on a single symbol.

    Returns:
        Response: JSON analysis result or error.
    """
    try:
        symbol = _extract_symbol_from_request()
        if not symbol:
            return jsonify(
                {
                    "status": "error",
                    "message": "Symbol required (query param or JSON)",
                }
            ), 400

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {
                    "status": "error",
                    "message": "Engine not initialized",
                }
            ), 503

        strategy = screener_engine.get_strategy(Strategies.DipBuyer)
        if not strategy:
            return jsonify(
                {
                    "status": "error",
                    "message": "DipBuyer strategy not found",
                }
            ), 404

        analysis_result = strategy.analyze_single_symbol(symbol)
        return jsonify(analysis_result), 200

    except Exception as error:
        logger.exception("Error analyzing symbol: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/turnover", methods=["POST"])
@require_ip_whitelist
def analyze_turnover() -> Response:
    """Detailed debugging for TurnoverTiming strategy on a single symbol.

    Returns:
        Response: JSON analysis result or error.
    """
    try:
        symbol = _extract_symbol_from_request()
        if not symbol:
            return jsonify({"status": "error", "message": "Symbol required"}), 400

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {
                    "status": "error",
                    "message": "Engine not initialized",
                }
            ), 503

        strategy = screener_engine.get_strategy(Strategies.TurnOverTiming)
        if not strategy:
            return jsonify(
                {
                    "status": "error",
                    "message": "TurnoverTiming strategy not found",
                }
            ), 404

        analysis_result = strategy.analyze_single_symbol(symbol)
        return jsonify(analysis_result), 200

    except Exception as error:
        logger.exception("Error analyzing turnover symbol: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/croc", methods=["POST"])
@require_ip_whitelist
def analyze_croc() -> Response:
    """Returns the full list of recommended signals for CrocSetup.

    Returns:
        Response: JSON analysis result or error.
    """
    try:
        days_lookback = request.args.get("days", default=0, type=int)
        analysis_date = request.args.get("date")

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {
                    "status": "error",
                    "message": "Engine not initialized",
                }
            ), 503

        strategy = screener_engine.get_strategy(Strategies.CrocSetup)
        if not strategy:
            return jsonify(
                {
                    "status": "error",
                    "message": "CrocSetup strategy not found",
                }
            ), 404

        signals = strategy.get_all_recommendations(
            days=days_lookback, analysis_date=analysis_date
        )

        return jsonify(
            {
                "status": "success",
                "date": analysis_date,
                "days_lookback": days_lookback,
                "signals": signals,
            }
        ), 200

    except Exception as error:
        logger.exception("Error analyzing Croc setup: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/ndx-momentum", methods=["POST"])
@require_ip_whitelist
def analyze_ndx_momentum() -> Response:
    """Returns the current market regime and top momentum leaders for NDX.

    Returns:
        Response: JSON with regime status and top symbols.
    """
    try:
        analysis_date = request.args.get("date")

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {
                    "status": "error",
                    "message": "Engine not initialized",
                }
            ), 503

        strategy = screener_engine.get_strategy(Strategies.NDXMomentum)

        if not strategy:
            return jsonify(
                {
                    "status": "error",
                    "message": "NDXMomentum strategy not found",
                }
            ), 404

        analysis = strategy.calculate_analysis(
            analysis_date=analysis_date, force_run=True
        )

        return jsonify(
            {
                "status": "success",
                "date": analysis.get("date"),
                "requested_date": analysis.get("requested_date"),
                "is_rebalance_day": analysis.get("is_rebalance_day", False),
                "regime": analysis.get("regime_indicators"),
                "top_leaders": analysis.get("top_symbols", []),
                "error": analysis.get("error"),
            }
        ), 200

    except Exception as error:
        logger.exception("Error analyzing NDX Momentum: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/tgim", methods=["POST"])
@api_blueprint.route("/api/screener/tgim", methods=["POST"])
@require_ip_whitelist
def analyze_tgim() -> Response:
    """Returns analysis and screening candidates for TGIM strategy.

    Returns:
        Response: JSON with TGIM signal status.
    """
    try:
        analysis_date = request.args.get("date")
        days = request.args.get("days", default=0, type=int)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {"status": "error", "message": "Engine not initialized"}
            ), 503

        strategy = screener_engine.get_strategy(Strategies.TGIM)
        if not strategy:
            return jsonify(
                {"status": "error", "message": "TGIM strategy not found"}
            ), 404

        candidates_count = strategy.run(days=days, analysis_date=analysis_date)
        return jsonify(
            {
                "status": "success",
                "strategy": "tgim",
                "symbol": "SPY",
                "signals_found": candidates_count,
            }
        ), 200
    except Exception as error:
        logger.exception("Error analyzing TGIM: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/bridge-scout", methods=["POST"])
@api_blueprint.route("/api/screener/bridge-scout", methods=["POST"])
@require_ip_whitelist
def analyze_bridge_scout() -> Response:
    """Returns analysis and screening candidates for Bridge Scout strategy.

    Returns:
        Response: JSON with Bridge Scout signal status.
    """
    try:
        analysis_date = request.args.get("date")
        days = request.args.get("days", default=0, type=int)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {"status": "error", "message": "Engine not initialized"}
            ), 503

        strategy = screener_engine.get_strategy(Strategies.BridgeScout)
        if not strategy:
            return jsonify(
                {"status": "error", "message": "Bridge Scout strategy not found"}
            ), 404

        candidates_count = strategy.run(days=days, analysis_date=analysis_date)
        return jsonify(
            {
                "status": "success",
                "strategy": "bridge_scout",
                "symbol": "QQQ",
                "signals_found": candidates_count,
            }
        ), 200
    except Exception as error:
        logger.exception("Error analyzing Bridge Scout: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/bounce-bandit", methods=["POST"])
@api_blueprint.route("/api/screener/bounce-bandit", methods=["POST"])
@api_blueprint.route("/screener/bounce_bandit", methods=["POST"])
@api_blueprint.route("/api/screener/bounce_bandit", methods=["POST"])
@require_ip_whitelist
def analyze_bounce_bandit() -> Response:
    """Returns analysis and screening candidates for Bounce Bandit strategy.

    Returns:
        Response: JSON with Bounce Bandit signal status.
    """
    try:
        analysis_date = request.args.get("date")
        days = request.args.get("days", default=0, type=int)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify(
                {"status": "error", "message": "Engine not initialized"}
            ), 503

        strategy = screener_engine.get_strategy(Strategies.BounceBandit)
        if not strategy:
            return jsonify(
                {"status": "error", "message": "Bounce Bandit strategy not found"}
            ), 404

        candidates_count = strategy.run(days=days, analysis_date=analysis_date)
        return jsonify(
            {
                "status": "success",
                "strategy": "bounce_bandit",
                "symbol": "QQQ",
                "signals_found": candidates_count,
            }
        ), 200
    except Exception as error:
        logger.exception("Error analyzing Bounce Bandit: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> Response:
    """Triggers the daily order generation process.

    Returns:
        Response: JSON status with generated file path.
    """
    trade_manager = current_app.extensions.get("trade_manager")
    if not trade_manager:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        order_file_path = trade_manager.generate_daily_orders()
        if order_file_path:
            return jsonify({"status": "success", "file": order_file_path}), 201

        return jsonify({"status": "success", "message": "No orders generated"}), 200
    except Exception as error:
        logger.error("Order generation failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill", methods=["POST"])
@api_blueprint.route("/api/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> Response:
    """Triggers a backfill or retry of trade processing.

    If strategy=tgim or strategy=bridge_scout parameter is specified, runs corresponding backfill.

    Returns:
        Response: JSON confirmation.
    """
    strategy = (request.args.get("strategy") or "").lower()

    if strategy in ("tgim", "thank_god_its_monday"):
        return backfill_tgim_trades()
    if strategy in ("bridge_scout", "bridgescout", "bridge scout", "qqq_eom"):
        return backfill_bridge_scout_trades()
    if strategy in ("bounce_bandit", "bouncebandit", "bounce bandit", "qqq_meanrev"):
        return backfill_bounce_bandit_trades()

    trade_manager = current_app.extensions.get("trade_manager")
    if not trade_manager:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500

    try:
        trade_manager.run_daily_process()
        return jsonify(
            {
                "status": "success",
                "info": "Backfill executed via daily process",
            }
        )
    except Exception as error:
        logger.error("Trades backfill failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill/tgim", methods=["POST"])
@api_blueprint.route("/backfill/tgim", methods=["POST"])
@api_blueprint.route("/api/trades/backfill/tgim", methods=["POST"])
@api_blueprint.route("/api/backfill/tgim", methods=["POST"])
@require_ip_whitelist
def backfill_tgim_trades() -> Response:
    """Executes historical backfill simulation for TGIM strategy trades.

    Returns:
        Response: JSON containing backfill summary metrics and trades list.
    """
    start_date = (
        request.args.get("start_date") or request.args.get("start") or "2026-01-01"
    )
    end_date = request.args.get("end_date") or request.args.get("end")
    budget = float(request.args.get("budget") or 10000.0)
    raw_clear = request.args.get("clear_existing") or request.args.get("clear")
    clear_existing = _parse_boolean_parameter(raw_clear, default_value=True)

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    stocks_session = DatabaseSession(str(configuration.get_path("stocks")))
    signals_session = DatabaseSession(str(configuration.get_path("signals")))

    try:
        result = tgim_backfill.run_tgim_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            clear_existing=clear_existing,
        )
        try:
            cache.clear()
        except Exception as cache_err:
            logger.debug("Cache clear skipped: %s", cache_err)
        return jsonify({"status": "success", "result": result})
    except Exception as error:
        logger.error("TGIM trades backfill failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill/bridge-scout", methods=["POST"])
@api_blueprint.route("/backfill/bridge-scout", methods=["POST"])
@api_blueprint.route("/api/trades/backfill/bridge-scout", methods=["POST"])
@api_blueprint.route("/api/backfill/bridge-scout", methods=["POST"])
@require_ip_whitelist
def backfill_bridge_scout_trades() -> Response:
    """Executes historical backfill simulation for Bridge Scout strategy trades.

    Returns:
        Response: JSON containing backfill summary metrics and trades list.
    """
    start_date = (
        request.args.get("start_date") or request.args.get("start") or "2025-01-01"
    )
    end_date = request.args.get("end_date") or request.args.get("end")
    budget = float(request.args.get("budget") or 10000.0)
    raw_clear = request.args.get("clear_existing") or request.args.get("clear")
    clear_existing = _parse_boolean_parameter(raw_clear, default_value=True)

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    stocks_session = DatabaseSession(str(configuration.get_path("stocks")))
    signals_session = DatabaseSession(str(configuration.get_path("signals")))

    try:
        result = bridge_scout_backfill.run_bridge_scout_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            clear_existing=clear_existing,
        )
        try:
            cache.clear()
        except Exception as cache_err:
            logger.debug("Cache clear skipped: %s", cache_err)
        return jsonify({"status": "success", "result": result})
    except Exception as error:
        logger.error("Bridge Scout trades backfill failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill/bounce-bandit", methods=["POST"])
@api_blueprint.route("/backfill/bounce-bandit", methods=["POST"])
@api_blueprint.route("/api/trades/backfill/bounce-bandit", methods=["POST"])
@api_blueprint.route("/api/backfill/bounce-bandit", methods=["POST"])
@api_blueprint.route("/trades/backfill/bounce_bandit", methods=["POST"])
@api_blueprint.route("/backfill/bounce_bandit", methods=["POST"])
@api_blueprint.route("/api/trades/backfill/bounce_bandit", methods=["POST"])
@api_blueprint.route("/api/backfill/bounce_bandit", methods=["POST"])
@require_ip_whitelist
def backfill_bounce_bandit_trades() -> Response:
    """Executes historical backfill simulation for Bounce Bandit strategy trades.

    Returns:
        Response: JSON containing backfill summary metrics and trades list.
    """
    start_date = (
        request.args.get("start_date") or request.args.get("start") or "2025-01-01"
    )
    end_date = request.args.get("end_date") or request.args.get("end")
    budget = float(request.args.get("budget") or 10000.0)
    raw_clear = request.args.get("clear_existing") or request.args.get("clear")
    clear_existing = _parse_boolean_parameter(raw_clear, default_value=True)

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    stocks_session = DatabaseSession(str(configuration.get_path("stocks")))
    signals_session = DatabaseSession(str(configuration.get_path("signals")))

    try:
        result = bounce_bandit_backfill.run_bounce_bandit_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            clear_existing=clear_existing,
        )
        try:
            cache.clear()
        except Exception as cache_err:
            logger.debug("Cache clear skipped: %s", cache_err)
        return jsonify({"status": "success", "result": result})
    except Exception as error:
        logger.error("Bounce Bandit trades backfill failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


# --- MARKET DATA ---


@api_blueprint.route("/market/sync", methods=["POST"])
@require_ip_whitelist
def sync_market_data() -> Response:
    """Triggers background market data synchronization.

    Returns:
        Response: JSON status accepted.
    """
    should_full_sync = request.args.get("full", "false").lower() == "true"
    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    database_path = configuration.get_db_path("stocks")
    signals_database_path = configuration.get_db_path("signals")

    def _execute_sync_task() -> None:
        """Background task for market synchronization."""
        try:
            market_session = DatabaseSession(str(database_path))
            signals_session = DatabaseSession(str(signals_database_path))

            updater = MarketDataUpdater(market_session, signals_session)
            updater.run_update(full_reload=should_full_sync)

            quality_service = MarketQualityService(updater)
            quality_service.perform_gap_check()
        except (RuntimeError, ValueError) as sync_error:
            logger.error("Market Sync Task Error: %s", sync_error)

    Thread(target=_execute_sync_task, daemon=True).start()
    return jsonify({"status": "accepted", "message": "Sync started"}), 202


@api_blueprint.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> Response:
    """Triggers a full manual reload of market data in the background.

    Returns:
        Response: JSON status queued.
    """
    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    database_path = configuration.get_db_path("stocks")
    signals_database_path = configuration.get_db_path("signals")

    def _execute_reload_task() -> None:
        """Background task for full market reload."""
        try:
            logger.info("Manual full reload via API started...")
            market_session = DatabaseSession(str(database_path))
            signals_session = DatabaseSession(str(signals_database_path))

            updater = MarketDataUpdater(market_session, signals_session)
            updater.run_update(full_reload=True)
        except (RuntimeError, ValueError) as reload_error:
            logger.error("Market Reload Task Error: %s", reload_error)

    Thread(target=_execute_reload_task, daemon=True).start()
    return jsonify({"status": "queued", "message": "Full reload triggered"}), 200
