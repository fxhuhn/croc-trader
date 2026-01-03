import logging

from flask import Blueprint, current_app, jsonify, request

from .models import CrocSignal

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
