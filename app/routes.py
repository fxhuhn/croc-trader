import logging

from flask import Blueprint, current_app, jsonify, request

from .models import CrocSignal
from .services.database import SignalDatabase

logger = logging.getLogger(__name__)

# Blueprint definieren
main_bp = Blueprint("main", __name__)


def check_ip_auth():
    """
    Prüft, ob die Anfrage von einer erlaubten IP kommt.
    Gibt vorerst nur eine Warnung aus, blockiert aber nicht.
    """
    conf = current_app.config["APP_CONFIG"]
    whitelist = conf.app.security.whitelist
    mode = conf.app.security.mode

    # 1. Versuche die ECHTE IP zu ermitteln (hinter Proxy/Docker)
    if request.headers.getlist("X-Forwarded-For"):
        # Wir nehmen das erste Element und splitten am Komma, falls es als String kommt
        header_val = request.headers.getlist("X-Forwarded-For")[0]
        client_ip = header_val.split(",")[0].strip()

    # 2. X-Real-IP: Alternative (Oft von Nginx genutzt)
    elif request.headers.get("X-Real-IP"):
        client_ip = request.headers.get("X-Real-IP").strip()

    # 3. Fallback: Direkte Verbindung (Docker Gateway oder lokaler Aufruf)
    else:
        client_ip = request.remote_addr

    # 2. Prüfen
    if client_ip not in whitelist:
        msg = f"SECURITY ALERT: Webhook von unbekannter IP empfangen: {client_ip}"

        if mode == "block":
            logger.warning(f"{msg} -> BLOCKED")
            # Hier würden wir abbrechen: abort(403)
            return False
        else:
            # Modus "warning"
            logger.warning(f"{msg} -> ALLOWED (Warning Mode)")
            return True

    # IP ist in der Liste
    return True


@main_bp.route("/")
def index():
    conf = current_app.config["APP_CONFIG"]
    return jsonify(
        status="running",
        env=conf.env.APP_ENV,
        db_folder=str(conf.db_root_path),
        active_signals_db=conf.get_db_path("signals"),
    )


@main_bp.route("/webhook", methods=["POST"])
def webhook():
    if not check_ip_auth():
        # Falls Block-Modus aktiv ist und Check fehlschlug:
        return jsonify({"status": "error", "message": "Unauthorized IP"}), 403

    if not request.is_json:
        return jsonify({"status": "error", "message": "JSON required"}), 400

    try:
        data = request.get_json()
        # Validiere und erzeuge Objekt
        try:
            signal = CrocSignal(**data)
        except TypeError as e:
            return jsonify({"status": "error", "message": str(e)}), 400

        # Zugriff auf den Worker via current_app Extensions
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


@main_bp.route("/croc-signal", methods=["GET"])
def get_croc_signals():
    """
    Liefert eine Liste der letzten Signale angereichert mit Win-Rates.
    Parameter:
      - limit: Anzahl der Einträge (Standard: 50)
      - symbol: Filter nach Aktienkürzel (Optional)
    """
    # 1. Parameter aus URL lesen
    limit = request.args.get("limit", default=50, type=int)
    symbol = request.args.get("symbol", default=None, type=str)

    # 2. Datenbank-Pfad aus Config holen
    conf = current_app.config["APP_CONFIG"]
    db_path = conf.get_db_path("signals")

    # 3. Datenbank abfragen
    try:
        db = SignalDatabase(db_path)
        data = db.get_latest_signals_with_stats(limit=limit, symbol=symbol)

        return jsonify(
            {
                "status": "success",
                "count": len(data),
                "filter": {"symbol": symbol, "limit": limit},
                "data": data,
            }
        )

    except Exception as e:
        logger.error(f"API Error /croc-signal: {e}", exc_info=True)
        bot = current_app.extensions.get("telegram")
        if bot:
            bot.send(f"⚠️ **FEHLER im Webhook:**\n{str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@main_bp.route("/admin/clean-exchanges", methods=["POST"])
def clean_exchanges():
    # Security Check (IP Auth)
    check_ip_auth()

    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))

    count = db.clean_batz_exchanges()

    return jsonify({"status": "success", "updated_rows": count})


@main_bp.route("/screener/dip-buyer", methods=["GET"])
def get_dip_buyer_results():
    """
    Zeigt die Ergebnisse des Dip-Buyer Screeners.
    Query Params: limit (default 50)
    """
    limit = request.args.get("limit", 50, type=int)

    conf = current_app.config["APP_CONFIG"]
    db_path = conf.get_db_path("signals")

    try:
        db = SignalDatabase(db_path)
        # Wir holen explizit die Strategie "dip_buyer"
        results = db.get_screener_results("dip_buyer", limit=limit)

        return jsonify(
            {
                "status": "success",
                "strategy": "dip_buyer",
                "count": len(results),
                "data": results,
            }
        )
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Screener Daten: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# Optional: Manueller Trigger zum Testen (geschützt)
@main_bp.route("/screener/run", methods=["POST"])
def run_screener_manual():
    check_ip_auth()

    # Zugriff auf den Screener Worker (den wir gleich in __init__ einbauen)
    screener = current_app.extensions.get("screener_engine")
    if screener:
        count = screener.run_dip_buyer()
        return jsonify({"status": "success", "hits": count})
    return jsonify({"status": "error", "message": "Screener not initialized"}), 500


@main_bp.route("/health")
def health():
    # Prüfen, ob DB erreichbar ist
    # Prüfen, ob Threads noch leben
    return "OK", 200


@main_bp.route("/portfolio/active", methods=["GET"])
def get_active_trades():
    """Zeigt alle offenen Trades."""
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))

    trades = db.get_open_trades()
    return jsonify({"count": len(trades), "trades": trades})


@main_bp.route("/portfolio/check", methods=["POST"])
def trigger_trade_check():
    """Manuell den TradeManager starten."""
    check_ip_auth()

    manager = current_app.extensions.get("trade_manager")
    if manager:
        manager.check_active_positions()
        return jsonify({"status": "triggered"})
    return jsonify({"status": "error"}), 500


@main_bp.route("/admin/backfill-signals", methods=["POST"])
def backfill_signals():
    """Startet den Screener für die letzten 10 Tage."""
    check_ip_auth()

    # Optional: Tage per Parameter übergeben ?days=20
    days = request.args.get("days", 10, type=int)

    screener = current_app.extensions.get("screener_engine")
    if screener:
        count = screener.run_historical_test(lookback_days=days)
        return jsonify(
            {"status": "success", "signals_generated": count, "days_checked": days}
        )

    return jsonify({"status": "error", "message": "Screener not ready"}), 500
