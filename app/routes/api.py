import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any

# 1. HIER 'abort' HINZUFÜGEN
from flask import Blueprint, Response, current_app, jsonify, request

from app.models import CrocSignal

from ..services.market_data import DataValidator, MarketDataService
from .security import require_ip_whitelist

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check() -> Response:
    return jsonify({"status": "ok"})


@api_bp.route("/", methods=["GET"])
@require_ip_whitelist
def root_check() -> Response:
    return jsonify({"status": "ok"})


@api_bp.route("/webhook", methods=["POST"])
@require_ip_whitelist
def ingest_webhook() -> Response:
    try:
        # EAFP: Versuch JSON direkt zu laden, Fallback auf Raw-Decode
        data: dict[str, Any] | None = request.get_json(silent=True)

        if data is None:
            raw_data = request.data
            decoded_str = raw_data.decode("utf-8")
            data = json.loads(decoded_str)

        if not data:
            raise ValueError("Empty Payload")

        signal = CrocSignal(**data)
        worker = current_app.extensions["worker"]
        worker.enqueue(signal)

        logger.info(f"Webhook received: {signal.symbol} {signal.signal}")
        return jsonify({"status": "queued", "ref": signal.reference})  # type: ignore

    except (TypeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Invalid Webhook Payload: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Internal Webhook Error: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500


@api_bp.route("/screener/run", methods=["POST"])
@require_ip_whitelist
def trigger_screener() -> Response:
    try:
        days = request.args.get("days", default=0, type=int)
        clean = request.args.get("clean", default="false").lower() == "true"

        screener = current_app.extensions.get("screener_engine")
        strategy_engine = current_app.extensions.get("strategy_engine")

        if not screener or not strategy_engine:
            raise RuntimeError("Engines not initialized")

        if clean:
            screener.signals_db.clear_screener_webhook()
            logger.info("Screener tables cleaned.")

        screener_results = screener.run_all(days=days)
        strategy_engine.run_daily_analysis(lookback_days=days if days > 0 else 1)

        return jsonify(
            {
                "status": "success",
                "message": "Run completed",
                "screener_hits": screener_results,
            }
        )

    except Exception as e:
        logger.error(f"Screener Run Failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> Response:
    tm = current_app.extensions.get("trade_manager")
    if not tm:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500

    try:
        tm.run_daily_process()
        return jsonify({"status": "success", "message": "TradeManager executed"})
    except Exception as e:
        logger.error(f"Order Generation Failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> Response:
    """
    Startet den Backfill-Prozess für hängengebliebene CREATED Trades.
    """
    tm = current_app.extensions.get("trade_manager")
    if not tm:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500

    try:
        # Führe den Backfill im TradeManager aus
        stats = tm.run_backfill()

        return jsonify(
            {"status": "success", "message": "Backfill completed", "stats": stats}
        )
    except Exception as e:
        logger.error(f"Backfill Failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/market/update", methods=["POST"])
@require_ip_whitelist
def trigger_market_update() -> Response:
    """Stößt das reguläre Markt-Update (letzte 30 Tage) im Hintergrund an."""
    worker = current_app.extensions.get("market_worker")
    if not worker:
        return jsonify({"status": "error", "message": "MarketDataWorker missing"}), 500

    # Job asynchron in den Scheduler werfen
    worker.scheduler.add_job(
        worker.run_update_job,
        trigger="date",
        run_date=datetime.now(),
        id="manual_market_update",
        replace_existing=True,
    )

    return jsonify(
        {"status": "queued", "message": "Market data update started in background"}
    )


@api_bp.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> Response:
    """
    Erzwingt einen Full-Reload für eine Liste von Symbolen.
    Body: {"symbols": ["AAPL", "TSLA"]}
    """
    data = request.get_json(silent=True)
    if not data or "symbols" not in data:
        return jsonify(
            {"status": "error", "message": "Missing 'symbols' list in body"}
        ), 400

    worker = current_app.extensions.get("market_worker")
    if not worker:
        return jsonify({"status": "error", "message": "MarketDataWorker missing"}), 500

    # Full Reload (lädt ab 2020) im Hintergrund starten
    worker.scheduler.add_job(
        worker._perform_full_reload,
        args=[data["symbols"]],
        trigger="date",
        run_date=datetime.now(),
        id="manual_full_reload",
        replace_existing=True,
    )

    return jsonify(
        {
            "status": "queued",
            "message": f"Full reload for {len(data['symbols'])} symbols queued",
        }
    )


@api_bp.route("/portfolio", methods=["GET"])
def get_portfolio() -> Response:
    return jsonify({"status": "ok"})


@api_bp.route("/market/sync", methods=["POST"])
def sync_market_data():
    """
    Manueller Trigger für Market Data Update.
    Query Params:
      - full=true (erzwingt Full Reload)
      - check=true (führt nur Validierung aus)
    """
    full_reload = request.args.get("full", "false").lower() == "true"
    only_check = request.args.get("check", "false").lower() == "true"

    # Pfad aus Config holen
    conf = current_app.config["APP_CONFIG"]
    db_path_str = conf.get_db_path("stocks")

    def _background_task():
        try:
            # Path Objekt erstellen
            db_path = Path(db_path_str)

            service = MarketDataService(db_path)
            validator = DataValidator(service)

            if only_check:
                logger.info("Manuelle Validierung gestartet...")
                validator.run_logical_checks()
                validator.run_spot_check(sample_size=50, lookback_days=10)
                # Auch beim reinen Check prüfen wir auf Lücken
                service.perform_gap_check()
            else:
                logger.info(f"Manueller Sync gestartet (Full={full_reload})...")

                # 1. Update
                service.update_market_data(full_reload=full_reload)

                # 2. Validierung (Logik)
                validator.run_logical_checks()

                # 3. Gap Check als letzte Instanz (Lücken füllen)
                logger.info("Führe abschließenden Gap-Check durch...")
                service.perform_gap_check()

                logger.info("Manueller Prozess vollständig beendet.")

        except Exception as e:
            logger.error(f"Fehler im manuellen Sync-Task: {e}", exc_info=True)

    # Thread starten (Non-Blocking Response)
    Thread(target=_background_task, daemon=True).start()

    mode = (
        "Validierung"
        if only_check
        else ("Full Reload" if full_reload else "Incremental Update")
    )
    return jsonify(
        {
            "status": "accepted",
            "message": f"Task '{mode}' inkl. Gap-Check wurde im Hintergrund gestartet.",
        }
    ), 202
