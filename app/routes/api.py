import json
import logging
from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from app.models import CrocSignal

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


@api_bp.route("/portfolio", methods=["GET"])
def get_portfolio() -> Response:
    return jsonify({"status": "ok"})


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
