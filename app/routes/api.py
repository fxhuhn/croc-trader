import json
import logging
from datetime import datetime
from typing import Any

# 1. HIER 'abort' HINZUFÜGEN
from flask import Blueprint, Response, abort, current_app, jsonify, request

from app.models import CrocSignal
from app.tools.symbol_lists import ExchangeSymbol

from .security import require_ip_whitelist

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)

# 2. HIER DIE LISTE DER ZU BLOCKENDEN DATEIEN DEFINIEREN
BLOCKED_EXTENSIONS = (".php", ".aspx", ".jsp", ".cgi", ".env", ".git", ".htaccess")


# 3. HIER DER 'TÜRSTEHER'-CODE
@api_bp.before_request
def block_script_kiddies():
    """
    Prüft vor jeder Anfrage in diesem Blueprint, ob nach Skripten
    gesucht wird, und bricht sofort mit 404 ab.
    """
    path = request.path.lower()

    # Check auf Dateiendungen
    if path.endswith(BLOCKED_EXTENSIONS):
        logger.warning(f"Blocked script scan: {path} from {request.remote_addr}")
        abort(404)

    # Check auf typische Wordpress/Admin Pfade
    if "wp-admin" in path or "wp-login" in path or "php" in path:
        logger.warning(f"Blocked WP scan: {path} from {request.remote_addr}")
        abort(404)


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
@require_ip_whitelist
def sync_market_gaps() -> Response:
    """
    Checks for missing stock data by comparing each symbol's last date
    against the global maximum date in the database.
    Triggers a full reload for any outdated or missing symbols.
    """
    worker = current_app.extensions.get("market_worker")
    if not worker:
        return jsonify({"status": "error", "message": "MarketDataWorker missing"}), 500

    # 1. Get the current state of the database (Symbol -> (Date, Close))
    last_entries = worker.db.get_all_last_entries_map(worker.PROVIDER, worker.TIMEFRAME)

    if not last_entries:
        return jsonify(
            {"status": "error", "message": "No market data found in DB."}
        ), 404

    # 2. Determine the 'Target Date' (The most recent date found in the DB)
    all_dates = [meta[0] for meta in last_entries.values()]
    target_date = max(all_dates)

    # 3. Compare all configured symbols against the target date
    all_symbols = ExchangeSymbol().all
    outdated_symbols = []
    stock_status_map = {}

    for symbol in all_symbols:
        entry = last_entries.get(symbol)

        if not entry:
            # Case: Symbol is in config but completely missing from DB
            stock_status_map[symbol] = "MISSING"
            outdated_symbols.append(symbol)
        else:
            # Case: Symbol exists, check if date is behind
            last_date = entry[0]
            stock_status_map[symbol] = last_date

            if last_date < target_date:
                outdated_symbols.append(symbol)

    # 4. Trigger background update for the outdated list
    if outdated_symbols:
        logger.info(f"Triggering gap-fill for {len(outdated_symbols)} symbols.")
        worker.scheduler.add_job(
            worker._perform_full_reload,
            args=[outdated_symbols],
            trigger="date",
            run_date=datetime.now(),
            id="manual_gap_fill",
            replace_existing=True,
        )

    return jsonify(
        {
            "status": "queued" if outdated_symbols else "synced",
            "target_max_date": target_date,
            "outdated_count": len(outdated_symbols),
            "triggered_symbols": outdated_symbols,
            "all_stocks_status": stock_status_map,
        }
    )
