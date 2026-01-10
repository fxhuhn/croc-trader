import json
import logging

from flask import Blueprint, current_app, jsonify, render_template_string, request

from .models import CrocSignal
from .services.database import SignalDatabase

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


def check_ip_auth():
    """
    Prüft die IP-Adresse gegen die Whitelist in der Konfiguration.
    Unterstützt X-Forwarded-For für Proxies (z.B. Docker/Nginx).
    """
    conf = current_app.config["APP_CONFIG"]

    # Zugriff auf die Security-Config sicherstellen
    if not hasattr(conf.app, "security"):
        # Fallback, falls Config noch nicht geladen/strukturiert ist
        return True

    whitelist = conf.app.security.whitelist
    mode = conf.app.security.mode

    # IP Ermittlung (hinter Reverse Proxy oder direkt)
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0].split(",")[0].strip()
    elif request.headers.get("X-Real-IP"):
        client_ip = request.headers.get("X-Real-IP").strip()
    else:
        client_ip = request.remote_addr

    # Whitelist Check
    if client_ip not in whitelist:
        if mode == "block":
            logger.warning(f"Unauthorized IP: {client_ip} -> BLOCKED")
            return False
        logger.warning(f"Unauthorized IP: {client_ip} -> ALLOWED (Warning)")

    return True


@main_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@main_bp.route("/", methods=["GET"])
def main():
    if check_ip_auth():
        return jsonify({"status": "error", "message": "Unauthorized IP"}), 403
    return jsonify({"status": "ok"}), 200


@main_bp.route("/webhook", methods=["POST"])
def webhook():
    # 1. Daten Parsing (Robust für verschiedene Content-Types)
    try:
        raw_data = request.data
        decoded_str = raw_data.decode("utf-8")
        data = json.loads(decoded_str)
    except Exception:
        data = request.get_json()

    # 2. Security Check
    if not check_ip_auth():
        return jsonify({"status": "error", "message": "Unauthorized IP"}), 403

    if not data:
        return jsonify({"status": "error", "message": "JSON required"}), 400

    # 3. Signal Verarbeitung
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


# ==============================================================================
# HTML VIEWS
# ==============================================================================


@main_bp.route("/screener/webhook", methods=["GET"])
def view_screener_webhook():
    """Zeigt die Ergebnisse des Webhook-Screeners als HTML-Tabelle."""
    limit = request.args.get("limit", 100, type=int)
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))
    results = db.get_webhook_results(limit=limit)

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Screener Webhook Ergebnisse</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #009879; color: white; }
            tr:nth-child(even) { background-color: #f3f3f3; }
            tr:hover { background-color: #f1f1f1; }
            h1 { color: #333; }
            .details { font-size: 0.85em; color: #555; font-style: italic; }
            .rank-high { font-weight: bold; color: #d35400; }
        </style>
    </head>
    <body>
        <h1>🔎 Webhook Screener Ergebnisse</h1>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Datum</th>
                    <th>Symbol</th>
                    <th>Strategie</th>
                    <th>Signal</th>
                    <th>Kriterien (Filter)</th>
                    <th>Close</th>
                    <th>RSI</th>
                    <th>SMA 200</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td class="rank-high">#{{ row['rank'] }}</td>
                    <td>{{ row['date'] }}</td>
                    <td><b>{{ row['symbol'] }}</b></td>
                    <td>{{ row['strategy'] }}</td>
                    <td>{{ row['signal'] }}</td>
                    <td class="details">{{ row['filter_details'] }}</td>
                    <td>{{ row['close'] }}</td>
                    <td>{{ row['rsi'] }}</td>
                    <td>{{ row['sma_200'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, results=results)


@main_bp.route("/screener/dip-buyer", methods=["GET"])
def view_screener_dip_buyer():
    """Zeigt die Ergebnisse des Dip-Buyer-Screeners als HTML-Tabelle."""
    limit = request.args.get("limit", 100, type=int)
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))
    results = db.get_dip_buyer_results(limit=limit)

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Screener Dip-Buyer Ergebnisse</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #2980b9; color: white; }
            tr:nth-child(even) { background-color: #f3f3f3; }
            tr:hover { background-color: #f1f1f1; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>📉 Dip-Buyer Screener Ergebnisse</h1>
        <table>
            <thead>
                <tr>
                    <th>Datum</th>
                    <th>Symbol</th>
                    <th>Setup Score</th>
                    <th>ATR R3</th>
                    <th>Entry Limit</th>
                    <th>ATR 5</th>
                    <th>Close</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row['date'] }}</td>
                    <td><b>{{ row['symbol'] }}</b></td>
                    <td>{{ row['setup_score'] }}</td>
                    <td>{{ row['atr_r3'] }}</td>
                    <td>{{ row['entry_limit'] }}</td>
                    <td>{{ row['atr5'] }}</td>
                    <td>{{ row['close'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, results=results)


@main_bp.route("/strategy/trades", methods=["GET"])
def view_strategy_trades():
    """Zeigt eine Übersicht aller generierten Trades (Historie)."""
    limit = request.args.get("limit", 100, type=int)
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))
    results = db.get_trades_history(limit=limit)

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Strategy Trades Übersicht</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #8e44ad; color: white; }
            tr:nth-child(even) { background-color: #f3f3f3; }
            tr:hover { background-color: #f1f1f1; }
            h1 { color: #333; }
            .status-open { color: #27ae60; font-weight: bold; }
            .status-created { color: #d35400; font-weight: bold; }
            .status-closed { color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>💼 Strategy Trades (Historie)</h1>
        <table>
            <thead>
                <tr>
                    <th>Entry Date</th>
                    <th>Strategy</th>
                    <th>Symbol</th>
                    <th>Status</th>
                    <th>Entry Price</th>
                    <th>Quantity</th>
                    <th>ATR @ Entry</th>
                    <th>Exit Reason</th>
                    <th>Closed At</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row['entry_date'] }}</td>
                    <td>{{ row['strategy'] }}</td>
                    <td><b>{{ row['symbol'] }}</b></td>
                    <td class="status-{{ row['status']|lower }}">{{ row['status'] }}</td>
                    <td>{{ row['entry_price'] }}</td>
                    <td>{{ row['quantity'] }}</td>
                    <td>{{ row['atr_at_entry'] }}</td>
                    <td>{{ row['exit_reason'] or '-' }}</td>
                    <td>{{ row['closed_at'] or '-' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, results=results)


# ==============================================================================
# ACTIONS (POST)
# ==============================================================================


@main_bp.route("/screener/run", methods=["POST"])
def run_screener():
    """
    Führt ALLE Strategien aus (DipBuyer, WebhookFilter, etc.).
    """
    # Security Check
    if not check_ip_auth():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    try:
        days = request.args.get("days", default=0, type=int)
        clean = request.args.get("clean", default="false").lower() == "true"

        screener = current_app.extensions.get("screener_engine")
        strategy_engine = current_app.extensions.get("strategy_engine")

        if not screener or not strategy_engine:
            return jsonify(
                {"status": "error", "message": "Engines not initialized"}
            ), 500

        # Optional: Aufräumen
        if clean:
            screener.signals_db.clear_screener_webhook()
            logger.info("Screener Tabellen (Webhook) geleert.")

        # 1. SCREENER LAUF (Alle Strategien)
        screener_results = screener.run_all(days=days)

        # 2. STRATEGY ENGINE LAUF (Trades erstellen)
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


@main_bp.route("/orders/generate", methods=["POST"])
def generate_orders():
    """
    Manuelles Triggern der Order-Erstellung (für Tests).
    """
    # Security Check
    if not check_ip_auth():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    tm = current_app.extensions.get("trade_manager")
    if not tm:
        return jsonify({"status": "error", "message": "TradeManager not found"}), 500

    try:
        # Führt die Logik aus: Prüfen -> Update -> YAML erstellen
        tm.check_active_positions()
        return jsonify(
            {
                "status": "success",
                "message": "TradeManager ausgeführt (siehe Logs/Ordner).",
            }
        )
    except Exception as e:
        logger.error(f"Fehler bei Order-Generierung: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@main_bp.route("/portfolio", methods=["GET"])
def portfolio():
    # Keine strenge IP Prüfung für Portfolio View (ReadOnly), optional hinzufügen wenn gewünscht
    return jsonify({"status": "ok"})
