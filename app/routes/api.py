import json
import logging
from pathlib import Path
from threading import Thread
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..services.market_data import DataValidator, MarketDataService

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

@api_bp.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> Response:
    tm = current_app.extensions.get("trade_manager")
    if not tm: return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        tm.run_daily_process()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route("/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> Response:
    """Trigger für Backfill/Retry von Trades."""
    tm = current_app.extensions.get("trade_manager")
    if not tm: return jsonify({"status": "error", "message": "TradeManager missing"}), 500
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
    if not conf: return jsonify({"status": "error"}), 500
    
    db_path = conf.get_db_path("stocks")

    def _task():
        try:
            svc = MarketDataService(Path(db_path))
            svc.update_market_data(full_reload=full)
            svc.perform_gap_check()
        except Exception as e:
            logger.error(f"Sync Error: {e}")

    Thread(target=_task, daemon=True).start()
    return jsonify({"status": "accepted", "message": "Sync started"}), 202

@api_bp.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> Response:
    """Manueller Reload Trigger."""
    conf = current_app.config.get("APP_CONFIG")
    if not conf: return jsonify({"status": "error"}), 500
    
    db_path = conf.get_db_path("stocks")
    
    def _task():
        try:
            svc = MarketDataService(Path(db_path))
            logger.info("Manueller Full-Reload via API gestartet...")
            svc.update_market_data(full_reload=True)
        except Exception as e:
            logger.error(f"Reload Error: {e}")

    Thread(target=_task, daemon=True).start()
    return jsonify({"status": "queued", "message": "Full reload triggered"}), 200