import json
import logging

from flask import Blueprint, current_app, jsonify, request

from .models import CrocSignal
from .services.database import SignalDatabase

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


def check_ip_auth():
    conf = current_app.config["APP_CONFIG"]
    whitelist = conf.app.security.whitelist
    mode = conf.app.security.mode

    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0].split(",")[0].strip()
    elif request.headers.get("X-Real-IP"):
        client_ip = request.headers.get("X-Real-IP").strip()
    else:
        client_ip = request.remote_addr

    if client_ip not in whitelist:
        if mode == "block":
            logger.warning(f"Unauthorized IP: {client_ip} -> BLOCKED")
            return False
        logger.warning(f"Unauthorized IP: {client_ip} -> ALLOWED (Warning)")
    return True


@main_bp.route("/")
def index():
    conf = current_app.config["APP_CONFIG"]
    return jsonify(
        status="running",
        env=conf.env.APP_ENV,
        db_folder=str(conf.db_root_path),
    )


@main_bp.route("/webhook", methods=["POST"])
def webhook():
    try:
        raw_data = request.data
        decoded_str = raw_data.decode("utf-8")
        data = json.loads(decoded_str)
    except Exception:
        data = request.get_json()

    if not check_ip_auth():
        return jsonify({"status": "error", "message": "Unauthorized IP"}), 403

    if not data:
        return jsonify({"status": "error", "message": "JSON required"}), 400

    try:
        signal = CrocSignal(**data)
        worker = current_app.extensions["worker"]
        worker.enqueue(signal)
        logger.info(f"Webhook empfangen: {signal.symbol} {signal.signal}")
        return jsonify({"status": "queued", "ref": signal.reference}), 202

    except (TypeError, ValueError) as e:
        logger.warning(f"Invalid Payload: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Internal Error: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500


@main_bp.route("/screener/run", methods=["POST"])
def run_screener():
    """
    Führt ALLE Strategien aus (DipBuyer, WebhookFilter, etc.).
    Parameter:
      - days (int): Rückblick in Tagen (Default: 0 = Daily).
      - clean (bool): Wenn true, leere vorher die Screener-Tabellen.
    """
    check_ip_auth()

    try:
        days = request.args.get("days", default=0, type=int)
        clean = request.args.get("clean", default="false").lower() == "true"

        screener = current_app.extensions.get("screener_engine")
        strategy_engine = current_app.extensions.get("strategy_engine")

        if not screener or not strategy_engine:
            return jsonify(
                {"status": "error", "message": "Engines not initialized"}
            ), 500

        # 1. Optional: Aufräumen
        if clean:
            # Wir leeren die Tabellen der bekannten Strategien
            screener.signals_db.clear_screener_webhook()
            # Falls DipBuyer auch geleert werden soll, müsste das in DB ergänzt werden,
            # aber Webhook ist meistens das Wichtigste beim Testen.
            logger.info("Screener Tabellen (Webhook) geleert.")

        # 2. SCREENER LAUF (Alle Strategien)
        # Hier nutzen wir jetzt die generische Methode!
        screener_results = screener.run_all(days=days)

        # 3. STRATEGY ENGINE LAUF (Trades erstellen)
        # Die Engine schaut in die Tabellen, die der Screener gerade gefüllt hat.
        strategy_engine.run_daily_analysis(lookback_days=days if days > 0 else 1)

        return jsonify(
            {
                "status": "success",
                "message": "Screener & Strategy Run completed.",
                "screener_hits": screener_results,
            }
        )

    except Exception as e:
        logger.error(f"Fehler im Screener Run: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@main_bp.route("/portfolio/active", methods=["GET"])
def get_active_trades():
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))
    trades = db.get_open_trades()
    return jsonify({"count": len(trades), "trades": trades})


@main_bp.route("/strategies/trades", methods=["GET"])
def get_strategy_trades():
    """Zeigt die generierten Strategie-Vorschläge an."""
    limit = request.args.get("limit", 50, type=int)
    strat_db = current_app.extensions.get("strategy_engine").strat_db
    df = strat_db.get_latest_trades(limit=limit)

    if df.empty:
        return jsonify({"count": 0, "data": []})

    # Pandas DF zu JSON
    data = df.to_dict(orient="records")
    return jsonify({"count": len(data), "data": data})


@main_bp.route("/health")
def health():
    return "OK", 200
