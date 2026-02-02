import logging
from threading import Thread
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..services.market.updater import MarketDataUpdater
from ..services.market.quality import MarketQualityService

# Import aus der separaten security.py (im gleichen Ordner)
from .security import require_ip_whitelist

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)

# --- STANDARD ROUTES ---

@api_bp.route("/health", methods=["GET"])
def health_check() -> Response:
    return jsonify({"status": "ok"})

@api_bp.route("/", methods=["GET"])
@require_ip_whitelist
def root_check() -> Response:
    return jsonify({"status": "ok"})

# --- SIGNAL INGESTION ---

@api_bp.route("/webhook", methods=["POST"])
@require_ip_whitelist
def ingest_webhook() -> Response:
    try:
        # Use silent=True to avoid crash on empty body, force=True if content-type is missing
        payload: dict[str, Any] | None = request.get_json(silent=True, force=True)

        if not payload:
            raw_data = request.get_data(as_text=True)
            logger.warning(f"⚠️ Malformed Webhook Data: {raw_data}")
            return jsonify({"status": "error", "message": "Empty Payload or Invalid JSON"}), 400

        configuration = current_app.config.get("APP_CONFIG")
        db_path = configuration.get_db_path("signals") if configuration else "instance/signals.db"
        
        session = DatabaseSession(str(db_path))
        repository = SignalRepository(session)

        signal_id = repository.save_signal(payload)
        
        symbol = payload.get("symbol", "UNKNOWN")
        logger.info(f"✅ Webhook saved: {symbol} -> ID {signal_id}")
        
        return jsonify({"status": "success", "id": signal_id}), 201

    except Exception as error:
        logger.error(f"Internal Webhook Error: {error}", exc_info=True)
        return jsonify({"status": "error", "message": str(error)}), 500

# --- SCREENER & TRADING ---

@api_bp.route("/screener/run", methods=["POST"])
@require_ip_whitelist
def trigger_screener() -> Response:
    try:
        days = request.args.get("days", default=0, type=int)
        target_strategy = request.args.get("strategy", type=str)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"error": "Screener Engine not initialized"}), 503

        logger.info(f"API Trigger: Screener run (days={days}, strategy={target_strategy})")
        stats = screener_engine.run_all(days=days, strategy_filter=target_strategy)

        return jsonify({"status": "success", "stats": stats}), 200
    except Exception as e:
        logger.exception("Error during screener run")
        return jsonify({"error": str(e)}), 500

@api_bp.route("/screener/dip-buyer", methods=["POST"])
@require_ip_whitelist
def analyze_dip_buyer() -> Response:
    """Detailed debugging for DipBuyer strategy on a single symbol."""
    try:
        # User requested support for "?symbol=AAPL" via POST
        symbol = request.args.get("symbol")
        
        # If not in args, check JSON body (standard API)
        if not symbol:
            try:
                data = request.get_json(force=True, silent=True)
                if data:
                    symbol = data.get("symbol")
            except Exception as e:
                logger.debug(f"Failed to parse JSON body for symbol: {e}")

        if not symbol:
             return jsonify({"status": "error", "message": "Symbol required (via query param or JSON)"}), 400

        # Change: Use ScreenerEngine instead of TradeManager
        # because the debuggable strategy instance lives in ScreenerEngine.
        engine = current_app.extensions.get("screener_engine")
        if not engine:
            return jsonify({"status": "error", "message": "ScreenerEngine not initialized"}), 503
            
        strategy = engine.get_strategy("DipBuyer")
        if not strategy:
             return jsonify({"status": "error", "message": "DipBuyer strategy not found in Screener"}), 404
             
        # Run analysis
        result = strategy.analyze_single_symbol(symbol)
        
        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Error analyzing symbol {symbol if 'symbol' in locals() else 'unknown'}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route("/screener/turnover", methods=["POST"])
@require_ip_whitelist
def analyze_turnover() -> Response:
    """Detailed debugging for TurnoverTiming strategy on a single symbol."""
    try:
        # Support Query Param and JSON
        symbol = request.args.get("symbol")
        
        if not symbol:
            try:
                data = request.get_json(force=True, silent=True)
                if data:
                    symbol = data.get("symbol")
            except Exception as e:
                logger.debug(f"Failed to parse JSON body for symbol: {e}")

        if not symbol:
             return jsonify({"status": "error", "message": "Symbol required"}), 400

        # FORCE UPPERCASE as requested by user
        symbol = str(symbol).upper().strip()

        engine = current_app.extensions.get("screener_engine")
        if not engine:
            return jsonify({"status": "error", "message": "ScreenerEngine not initialized"}), 503
            
        strategy = engine.get_strategy("TurnoverTiming")
        if not strategy:
             return jsonify({"status": "error", "message": "TurnoverTiming strategy not found in Screener"}), 404
             
        # Run analysis
        result = strategy.analyze_single_symbol(symbol)
        
        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Error analyzing turnover symbol {symbol if 'symbol' in locals() else 'unknown'}")
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> Response:
    tm = current_app.extensions.get("trade_manager")
    if not tm:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        file_path = tm.generate_daily_orders()
        if file_path:
            return jsonify({"status": "success", "file": file_path}), 201
        else:
            return jsonify({"status": "success", "message": "No orders generated"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route("/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> Response:
    """Trigger für Backfill/Retry von Trades."""
    tm = current_app.extensions.get("trade_manager")
    if not tm:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        # Wir nutzen run_daily_process, da dies auch CREATED trades verarbeitet
        tm.run_daily_process()
        return jsonify({"status": "success", "info": "Backfill executed via daily process"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- MARKET DATA ---

@api_bp.route("/market/sync", methods=["POST"])
@require_ip_whitelist
def sync_market_data() -> Response:
    full = request.args.get("full", "false").lower() == "true"
    conf = current_app.config.get("APP_CONFIG")
    if not conf:
        return jsonify({"status": "error"}), 500
    
    db_path = conf.get_db_path("stocks")
    signals_db_path = conf.get_db_path("signals")

    def _task():
        try:
            session = DatabaseSession(str(db_path))
            signals_session = DatabaseSession(str(signals_db_path))
            
            updater = MarketDataUpdater(session, signals_session)
            
            updater.run_update(full_reload=full)
            
            quality = MarketQualityService(updater)
            quality.perform_gap_check()
        except Exception as e:
            logger.error(f"Sync Error: {e}")

    Thread(target=_task, daemon=True).start()
    return jsonify({"status": "accepted", "message": "Sync started"}), 202

@api_bp.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> Response:
    """Manueller Reload Trigger."""
    conf = current_app.config.get("APP_CONFIG")
    if not conf:
        return jsonify({"status": "error"}), 500
    
    db_path = conf.get_db_path("stocks")
    signals_db_path = conf.get_db_path("signals")
    
    def _task():
        try:
            logger.info("Manueller Full-Reload via API gestartet...")
            session = DatabaseSession(str(db_path))
            signals_session = DatabaseSession(str(signals_db_path))
            
            updater = MarketDataUpdater(session, signals_session)
            updater.run_update(full_reload=True)
        except Exception as e:
            logger.error(f"Reload Error: {e}")

    Thread(target=_task, daemon=True).start()
    return jsonify({"status": "queued", "message": "Full reload triggered"}), 200