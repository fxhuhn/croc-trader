import logging
import uuid
from threading import Thread
from typing import TypedDict

from flask import Blueprint, Response, current_app, jsonify, request

from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..services.market.updater import MarketDataUpdater
from ..services.market.quality import MarketQualityService
from ..const import Strategies

# Import from the separate security.py
from .security import require_ip_whitelist

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


# --- STANDARD ROUTES ---

@api_blueprint.route("/health", methods=["GET"])
def health_check() -> Response:
    """
    Simple health check endpoint.
    
    Returns:
        Response: JSON status OK.
    """
    return jsonify({"status": "ok"})


@api_blueprint.route("/", methods=["GET"])
@require_ip_whitelist
def root_check() -> Response:
    """
    Authenticated root check endpoint.
    
    Returns:
        Response: JSON status OK.
    """
    return jsonify({"status": "ok"})


# --- SIGNAL INGESTION ---

@api_blueprint.route("/webhook", methods=["POST"])
@require_ip_whitelist
def ingest_webhook() -> Response:
    """
    Ingests signal webhooks and persists them to the signal database.
    
    Returns:
        Response: JSON success with signal ID or error message.
    """
    try:
        # Use silent=True to avoid crash on empty body, force=True if content-type missing
        payload: WebhookPayload | None = request.get_json(silent=True, force=True)

        if not payload:
            raw_data = request.get_data(as_text=True)
            logger.warning(f"⚠️ Malformed Webhook Data: {raw_data}")
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        # Mandatory Field Validation
        symbol = payload.get("symbol") or payload.get("ticker")
        if not symbol:
            logger.warning(f"⚠️ Webhook rejected: Missing 'symbol' in payload {payload}")
            return jsonify(
                {"status": "error", "message": "Missing mandatory field: symbol"}
            ), 400

        configuration = current_app.config.get("APP_CONFIG")
        database_path = (
            configuration.get_db_path("signals") 
            if configuration 
            else "instance/signals.db"
        )
        
        session = DatabaseSession(str(database_path))
        repository = SignalRepository(session)

        # Mapping dict to satisfy SignalRepository.save_signal which expects dict[str, Any]
        # but we use WebhookPayload for internal type safety.
        signal_id = repository.save_signal(dict(payload))
        
        logger.info(f"✅ Webhook saved: {symbol} -> ID {signal_id}")
        
        return jsonify({"status": "success", "id": signal_id}), 201

    except Exception as error:
        error_identifier = str(uuid.uuid4())[:8]
        logger.error(
            f"Webhook processing error [{error_identifier}]: {error}", 
            exc_info=True
        )
        return jsonify({
            "status": "error", 
            "message": "Internal Server Error", 
            "error_id": error_identifier
        }), 500


# --- SCREENER & TRADING ---

@api_blueprint.route("/screener/run", methods=["POST"])
@require_ip_whitelist
def trigger_screener() -> Response:
    """
    Triggers a manual run of all active screeners.
    
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
            f"API Trigger: Screener run (days={days_lookback}, strategy={target_strategy})"
        )
        statistics = screener_engine.run_all(
            days=days_lookback, 
            strategy_filter=target_strategy
        )

        return jsonify({"status": "success", "stats": statistics}), 200
    except Exception as error:
        logger.exception(f"Error during screener run: {error}")
        return jsonify({"error": str(error)}), 500


@api_blueprint.route("/screener/dip-buyer", methods=["POST"])
@require_ip_whitelist
def analyze_dip_buyer() -> Response:
    """
    Detailed debugging for DipBuyer strategy on a single symbol.
    
    Returns:
        Response: JSON analysis result or error.
    """
    try:
        symbol = request.args.get("symbol")
        
        if not symbol:
            data = request.get_json(force=True, silent=True)
            if data and isinstance(data, dict):
                symbol = data.get("symbol")

        if not symbol:
            return jsonify(
                {"status": "error", "message": "Symbol required (query param or JSON)"}
            ), 400

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"status": "error", "message": "Engine not initialized"}), 503
            
        strategy = screener_engine.get_strategy(Strategies.DipBuyer)
        if not strategy:
            return jsonify(
                {"status": "error", "message": "DipBuyer strategy not found"}
            ), 404
             
        analysis_result = strategy.analyze_single_symbol(symbol)
        return jsonify(analysis_result), 200

    except Exception as error:
        logger.exception(f"Error analyzing symbol: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/turnover", methods=["POST"])
@require_ip_whitelist
def analyze_turnover() -> Response:
    """
    Detailed debugging for TurnoverTiming strategy on a single symbol.
    
    Returns:
        Response: JSON analysis result or error.
    """
    try:
        symbol = request.args.get("symbol")
        
        if not symbol:
            data = request.get_json(force=True, silent=True)
            if data and isinstance(data, dict):
                symbol = data.get("symbol")

        if not symbol:
            return jsonify({"status": "error", "message": "Symbol required"}), 400

        # FORCE UPPERCASE
        clean_symbol = str(symbol).upper().strip()

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"status": "error", "message": "Engine not initialized"}), 503
            
        strategy = screener_engine.get_strategy(Strategies.TurnOverTiming)
        if not strategy:
            return jsonify(
                {"status": "error", "message": "TurnoverTiming strategy not found"}
            ), 404
             
        analysis_result = strategy.analyze_single_symbol(clean_symbol)
        return jsonify(analysis_result), 200

    except Exception as error:
        logger.exception(f"Error analyzing turnover symbol: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/ndx-momentum", methods=["POST"])
@require_ip_whitelist
def analyze_ndx_momentum() -> Response:
    """
    Returns the current market regime and top momentum leaders for NDX.
    
    Returns:
        Response: JSON with regime status and top symbols.
    """
    try:
        analysis_date = request.args.get("date")
        
        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"status": "error", "message": "Engine not initialized"}), 503
            
        strategy = screener_engine.get_strategy(Strategies.NDXMomentum)
        
        if not strategy:
            return jsonify(
                {"status": "error", "message": "NDXMomentum strategy not found"}
            ), 404
            
        # Call the new calculation method (Forced for API status check)
        analysis = strategy.calculate_analysis(analysis_date=analysis_date, force_run=True)
        
        return jsonify({
            "status": "success",
            "date": analysis.get("date"),
            "requested_date": analysis.get("requested_date"),
            "is_rebalance_day": analysis.get("is_rebalance_day", False),
            "regime": analysis.get("regime_indicators"),
            "top_leaders": analysis.get("top_symbols", []),
            "error": analysis.get("error")
        }), 200

    except Exception as error:
        logger.exception(f"Error analyzing NDX Momentum: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> Response:
    """
    Triggers the daily order generation process.
    
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
        logger.error(f"Order generation failed: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> Response:
    """
    Triggers a backfill or retry of trade processing.
    
    Returns:
        Response: JSON confirmation.
    """
    trade_manager = current_app.extensions.get("trade_manager")
    if not trade_manager:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        trade_manager.run_daily_process()
        return jsonify({
            "status": "success", 
            "info": "Backfill executed via daily process"
        })
    except Exception as error:
        logger.error(f"Trades backfill failed: {error}")
        return jsonify({"status": "error", "message": str(error)}), 500


# --- MARKET DATA ---

@api_blueprint.route("/market/sync", methods=["POST"])
@require_ip_whitelist
def sync_market_data() -> Response:
    """
    Triggers background market data synchronization.
    
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
            logger.error(f"Market Sync Task Error: {sync_error}")

    Thread(target=_execute_sync_task, daemon=True).start()
    return jsonify({"status": "accepted", "message": "Sync started"}), 202


@api_blueprint.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> Response:
    """
    Triggers a full manual reload of market data in the background.
    
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
            logger.error(f"Market Reload Task Error: {reload_error}")

    Thread(target=_execute_reload_task, daemon=True).start()
    return jsonify({"status": "queued", "message": "Full reload triggered"}), 200