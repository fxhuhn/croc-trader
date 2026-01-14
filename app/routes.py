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
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); font-size: 0.9em; }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #009879; color: white; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:nth-child(even) { background-color: #f3f3f3; }
            tr:hover { background-color: #f1f1f1; }
            .details { font-size: 0.85em; color: #555; font-style: italic; }
            .rank-high { font-weight: bold; color: #d35400; }

            /* TradingView Link Styling */
            a.tv-link {
                text-decoration: none;
                color: #009879; /* Angepasst an Header-Farbe */
                font-weight: bold;
                display: inline-flex;
                align-items: center;
            }
            a.tv-link:hover {
                text-decoration: underline;
                color: #007f65;
            }
            .tv-icon {
                font-size: 0.8em;
                margin-left: 4px;
                color: #7f8c8d;
            }
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
                    <th>Exchange</th>
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
                    {# --- TradingView URL Logic --- #}
                    {% set exchange_prefix = '' %}
                    {% if row['exchange'] and row['exchange'] != 'UNKNOWN' %}
                        {% set exchange_prefix = row['exchange'] ~ ':' %}
                    {% endif %}

                    {% set tv_interval = 'D' %}
                    {% if row['timeframe'] == '1D' %}
                        {% set tv_interval = 'D' %}
                    {% else %}
                        {% set tv_interval = row['timeframe'] %}
                    {% endif %}

                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}

                <tr>
                    <td class="rank-high">#{{ row['rank'] }}</td>
                    <td>{{ row['date'] }}</td>
                    <td>
                        <a href="{{ tv_url }}" class="tv-link" target="_blank" title="Chart auf TradingView öffnen">
                            {{ row['symbol'] }} <span class="tv-icon">↗</span>
                        </a>
                    </td>
                    <td style="font-size: 0.8em; color: #666;">{{ row['exchange'] }}</td>
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


@main_bp.route("/screener/webhook_2", methods=["GET"])
def view_screener_croc():
    """Zeigt die Ergebnisse des Croc-Setup Screeners (Ranking 2026)."""
    check_ip_auth()  # Security

    limit = request.args.get("limit", 200, type=int)
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))

    # Holt Ergebnisse sortiert nach Rank ASC, R-Per-Trade DESC
    results = db.get_croc_results(limit=limit)

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>🐊 Croc Setup Screener</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background: #fdfdfd; }
            h1 { color: #27ae60; border-bottom: 2px solid #27ae60; padding-bottom: 10px; display: inline-block; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.08); font-size: 0.9em; background: white; }
            th, td { border: 1px solid #eee; padding: 12px 15px; text-align: left; }
            th { background-color: #27ae60; color: white; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f0fdf4; }

            .rank-badge {
                background: #333; color: #fff; padding: 3px 8px; border-radius: 10px; font-weight: bold; font-size: 0.85em;
            }
            .r-val { color: #d35400; font-weight: bold; }
            .strat-highlight { color: #2980b9; font-weight: bold; }

            /* TradingView Link */
            a.tv-link { text-decoration: none; color: #27ae60; font-weight: bold; display: inline-flex; align-items: center; }
            a.tv-link:hover { text-decoration: underline; color: #1e8449; }
        </style>
    </head>
    <body>
        <h1>🐊 Croc Setup (Ranking 2026)</h1>
        <p>Top Setups basierend auf EMA, RSI und Extra-Filtern der letzten 10 Tage.</p>

        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Datum</th>
                    <th>Symbol</th>
                    <th>Signal</th>
                    <th>R / Trade</th>
                    <th>Empf. Strategie</th>
                    <th>Close</th>
                    <th>RSI</th>
                    <th>Dist EMA %</th>
                    <th>Auslöser (Filter)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}

                    {# Timeframe Mapping für TradingView: 1D -> D #}
                    {% set tv_interval = 'D' %}
                    {% if row['timeframe'] != '1D' %}
                         {% set tv_interval = row['timeframe'] %}
                    {% endif %}

                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}
                <tr>
                    <td><span class="rank-badge">#{{ row['rank'] }}</span></td>
                    <td>{{ row['date'] }}</td>
                    <td>
                        <a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} ↗</a>
                    </td>
                    <td>{{ row['signal'] }}</td>
                    <td class="r-val">{{ row['r_per_trade'] }}</td>
                    <td class="strat-highlight">{{ row['recommended_strategy'] }}</td>
                    <td>{{ row['close'] }}</td>

                    {# RSI: Clean rendering for None #}
                    <td>{{ row['rsi']|round(1) if row['rsi'] is not none else '-' }}</td>

                    {# EMA: Clean rendering for None (vermeidet 'None%') #}
                    <td>{{ row['dist_ema'] ~ '%' if row['dist_ema'] is not none else '-' }}</td>

                    <td style="font-style: italic; color: #666;">{{ row['match_filter'] }}</td>
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
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); font-size: 0.9em; }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #2980b9; color: white; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            tr:hover { background-color: #e9ecef; }

            /* TradingView Link Styling */
            a.tv-link {
                text-decoration: none;
                color: #2980b9;
                font-weight: bold;
                display: inline-flex;
                align-items: center;
            }
            a.tv-link:hover {
                text-decoration: underline;
                color: #1a5276;
            }
            .tv-icon {
                font-size: 0.8em;
                margin-left: 4px;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <h1>📉 Dip-Buyer Screener Ergebnisse</h1>
        <table>
            <thead>
                <tr>
                    <th>Datum</th>
                    <th>Symbol</th>
                    <th>Exchange</th>
                    <th>Setup Score</th>
                    <th>ATR R3</th>
                    <th>Entry Limit</th>
                    <th>ATR 5</th>
                    <th>Close</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {# --- Logic für TradingView URL Aufbau --- #}

                    {# 1. Exchange Prefix: Nur setzen, wenn Exchange bekannt und nicht UNKNOWN ist #}
                    {% set exchange_prefix = '' %}
                    {% if row['exchange'] and row['exchange'] != 'UNKNOWN' %}
                        {% set exchange_prefix = row['exchange'] ~ ':' %}
                    {% endif %}

                    {# 2. Timeframe Mapping: 1D -> D für TradingView URL #}
                    {% set tv_interval = 'D' %}
                    {% if row['timeframe'] == '1D' %}
                        {% set tv_interval = 'D' %}
                    {% else %}
                        {% set tv_interval = row['timeframe'] %}
                    {% endif %}

                    {# 3. Finale URL #}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}

                <tr>
                    <td>{{ row['date'] }}</td>
                    <td>
                        <a href="{{ tv_url }}" class="tv-link" target="_blank" title="Chart auf TradingView öffnen">
                            {{ row['symbol'] }} <span class="tv-icon">↗</span>
                        </a>
                    </td>
                    <td style="font-size: 0.8em; color: #666;">{{ row['exchange'] }}</td>
                    <td>{{ row['setup_score'] }}</td>
                    <td style="{{ 'color: green;' if row['atr_r3'] < -2 else '' }}">{{ row['atr_r3'] }}</td>
                    <td><b>{{ row['entry_limit'] }}</b></td>
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


# ==============================================================================
# NEU: TURNOVER TIMING VIEW
# ==============================================================================


@main_bp.route("/screener/turnover-timing", methods=["GET"])
def view_screener_turnover():
    """Zeigt die Ergebnisse des Turnover-Timing Screeners."""
    check_ip_auth()

    limit = request.args.get("limit", 100, type=int)
    conf = current_app.config["APP_CONFIG"]
    db = SignalDatabase(conf.get_db_path("signals"))
    results = db.get_turnover_timing_results(limit=limit)

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Turnover Timing Ergebnisse</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background-color: #f9f9f9; }
            h1 { color: #2c3e50; border-bottom: 3px solid #f39c12; display: inline-block; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 4px 8px rgba(0,0,0,0.1); font-size: 0.9em; background: white; }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background-color: #f39c12; color: white; text-transform: uppercase; letter-spacing: 0.5px; }
            tr:nth-child(even) { background-color: #fcfcfc; }
            tr:hover { background-color: #fff8e1; }

            .index-badge {
                background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-weight: bold; color: #555;
            }
            .money { font-family: monospace; color: #27ae60; font-weight: bold; }
            .entry-zone { color: #d35400; font-weight: bold; }

            /* TradingView Link Styling */
            a.tv-link { text-decoration: none; color: #e67e22; font-weight: bold; }
            a.tv-link:hover { text-decoration: underline; color: #d35400; }
        </style>
    </head>
    <body>
        <h1>🔄 Turnover Timing Screener</h1>
        <p>Top Aktien nach Turnover (SMA20) über SMA100 aus NDX, SPX, DOW.</p>

        <table>
            <thead>
                <tr>
                    <th>Datum</th>
                    <th>Symbol</th>
                    <th>Index (Quelle)</th>
                    <th>Turnover SMA20 ($)</th>
                    <th>Close</th>
                    <th>ATR(3)</th>
                    <th>Entry 1 (-0.5 ATR)</th>
                    <th>Entry 2 (-1.0 ATR)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] %}

                <tr>
                    <td>{{ row['date'] }}</td>
                    <td>
                        <a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} ↗</a>
                    </td>
                    <td><span class="index-badge">{{ row['source_index'] }}</span></td>
                    <td class="money">{{ "{:,.0f}".format(row['turnover_sma20']) }}</td>
                    <td>{{ row['close'] }}</td>
                    <td>{{ row['atr3'] }}</td>
                    <td class="entry-zone">{{ row['entry_1'] }}</td>
                    <td class="entry-zone">{{ row['entry_2'] }}</td>
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
        # HIER DIE ÄNDERUNG: Aufruf der neuen Methode
        tm.run_daily_process()

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


# ==============================================================================
# ERROR HANDLERS
# ==============================================================================


@main_bp.app_errorhandler(404)
def page_not_found(e):
    """
    Fängt alle 404 Fehler ab.
    Entscheidet intelligent, ob JSON (für API/Bots) oder HTML (für Browser) zurückgegeben wird.
    """
    # Loggen des Zugriffsversuchs (Wichtig für Security!)
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    logger.warning(f"404 Not Found: {request.method} {request.path} - IP: {client_ip}")

    # Wenn der Request JSON erwartet oder an die API geht -> JSON Antwort
    if (
        request.path.startswith(("/webhook", "/screener/run", "/orders", "/api"))
        or request.is_json
    ):
        return jsonify(
            {"status": "error", "message": "Endpoint not found", "path": request.path}
        ), 404

    # Sonst -> Schöne HTML Seite
    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>404 - Nicht gefunden</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 50px; color: #333; }
            h1 { font-size: 50px; color: #e74c3c; margin-bottom: 10px; }
            p { font-size: 18px; color: #666; }
            a { color: #2980b9; text-decoration: none; font-weight: bold; }
            .croc { font-size: 80px; }
        </style>
    </head>
    <body>
        <div class="croc">🐊❓</div>
        <h1>404</h1>
        <p>Hoppla! Diese Seite existiert nicht im Croc-Trader Universum.</p>
        <p>Vielleicht wolltest du zu den <a href="/screener/webhook">Screener Ergebnissen</a>?</p>
    </body>
    </html>
    """
    return render_template_string(html), 404


@main_bp.app_errorhandler(500)
def internal_server_error(e):
    """
    Fängt Programmabstürze ab, damit der Server nicht "hängt".
    """
    logger.error(f"500 Internal Server Error: {e}", exc_info=True)

    # Immer JSON versuchen bei 500ern, das ist sicherer für debug
    if request.path.startswith(("/webhook", "/screener", "/orders")) or request.is_json:
        return jsonify(
            {
                "status": "error",
                "message": "Internal Server Error",
                "detail": str(e),  # Vorsicht: Im Produktivbetrieb evtl. ausblenden
            }
        ), 500

    return render_template_string(
        "<h1>500 - Server Fehler</h1><p>Der Croc-Trader hat sich verschluckt. Check die Logs!</p>"
    ), 500


@main_bp.route("/backtest/dip-buyer", methods=["GET", "POST"])
def backtest_dip_buyer():
    """
    GET: Zeigt Startseite mit Eingabefeld für Debug-Symbol.
    POST: Startet den Backtest (optional mit Debugging).
    """
    check_ip_auth()  # Security Check

    if request.method == "POST":
        backtester = current_app.extensions.get("backtester")

        # Wir holen das Symbol aus dem Formular.
        # .strip() entfernt Leerzeichen, "or None" macht aus einem leeren String ein echtes None.
        debug_sym = request.form.get("debug_symbol", "").strip().upper() or None

        if debug_sym:
            logger.info(f"Starte Backtest mit Debugging für: {debug_sym}")

        # Backtest starten
        results = backtester.run_backtest(start_year=2023, debug_symbol=debug_sym)

        return render_backtest_view(results)

    # GET Request: Startseite mit Formular
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dip-Buyer Backtest</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; padding: 40px; text-align: center; background: #f4f7f6; color: #333; }
                h1 { color: #2c3e50; }
                .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; }
                input[type="text"] { padding: 12px; font-size: 16px; border: 1px solid #ddd; border-radius: 5px; width: 250px; margin-right: 10px; }
                button { padding: 12px 25px; font-size: 16px; background: #2980b9; color: white; border: none; border-radius: 5px; cursor: pointer; transition: background 0.3s; }
                button:hover { background: #3498db; }
                p { color: #7f8c8d; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div style="font-size: 60px;">📉</div>
                <h1>Dip-Buyer Strategie Analyse</h1>
                <p>Backtest über die Jahre 2023, 2024, 2025 bis heute.</p>

                <form method="POST">
                    <input type="text" name="debug_symbol" placeholder="Debug Symbol (z.B. APP) optional">
                    <button type="submit">Backtest starten 🚀</button>
                </form>
                <br>
                <small style="color: #999;">Lasse das Feld leer für einen kompletten Lauf ohne Detail-Logs.</small>
            </div>
        </body>
        </html>
    """)


def render_backtest_view(data):
    if "metrics" not in data:
        return render_template_string("<h1>Keine Daten</h1>")

    html = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Backtest Report</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f4f7f6; color: #333; max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .section-title { font-size: 1.1em; color: #7f8c8d; margin-top: 30px; text-transform: uppercase; letter-spacing: 1px; font-weight: bold; }

            .card { background: white; padding: 25px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }

            .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }

            .metric-box { text-align: center; padding: 15px; border: 1px solid #eee; border-radius: 8px; background: #fafafa; }
            .metric-box span.val { display: block; font-size: 2em; font-weight: bold; color: #2980b9; margin-bottom: 5px; }
            .metric-box span.lbl { font-size: 0.9em; color: #7f8c8d; }
            .metric-box.bad .val { color: #c0392b; }

            /* Tabellen Styles */
            table { width: 100%; border-collapse: collapse; font-size: 0.95em; }
            th, td { padding: 10px; text-align: center; border-bottom: 1px solid #eee; }
            th { background: #ecf0f1; color: #555; }

            /* Heatmap */
            .pos-high { background-color: #27ae60; color: white; }
            .pos-med { background-color: #2ecc71; color: white; }
            .pos-low { background-color: #a9dfbf; }
            .neg-low { background-color: #f5b7b1; }
            .neg-med { background-color: #e74c3c; color: white; }
            .neg-high { background-color: #c0392b; color: white; }
            .neutral { background-color: #fff; color: #eee; }

            /* Trades Table */
            .trades-table th { text-align: left; }
            .trades-table td { text-align: left; }
            .pos-text { color: #27ae60; font-weight: bold; }
            .neg-text { color: #c0392b; font-weight: bold; }

            .btn { display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px; }
            .btn:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <h1>📉 Dip-Buyer Backtest Report</h1>

        <div class="card">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <b>Zeitraum:</b> {{ data.data_universe.first_record }} bis {{ data.data_universe.last_record }}<br>
                    <b>Symbole:</b> {{ data.data_universe.total_symbols }}
                </div>
                <div style="text-align: right;">
                    <b>Total Signale:</b> {{ data.metrics.total_signals }}<br>
                    <b>Ausgeführt:</b> {{ data.metrics.total_trades }} (Fill-Rate: {{ data.metrics.fill_rate }}%)
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">Performance</div>
            <div class="grid-4">
                <div class="metric-box">
                    <span class="val">{{ data.metrics.profit_factor }}</span>
                    <span class="lbl">Profit Factor</span>
                </div>
                <div class="metric-box">
                    <span class="val">{{ data.metrics.win_rate }}%</span>
                    <span class="lbl">Win Rate</span>
                </div>
                <div class="metric-box">
                    <span class="val">{{ data.metrics.avg_return_pct }}%</span>
                    <span class="lbl">Ø Return / Trade</span>
                </div>
                <div class="metric-box bad">
                    <span class="val">{{ data.metrics.max_drawdown }}%</span>
                    <span class="lbl">Max Drawdown</span>
                </div>
            </div>
        </div>

        <div class="grid-4">
            <div class="card">
                <div class="section-title">Exit Gründe</div>
                <table style="margin-top: 10px;">
                    {% for reason, count in data.metrics.exit_reasons.items() %}
                    <tr>
                        <td style="text-align: left;">{{ reason }}</td>
                        <td style="text-align: right;"><b>{{ count }}</b></td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            <div class="card">
                <div class="section-title">Aktueller Monat ({{ data.comparison.current_month_name }})</div>
                <div style="text-align: center; margin-top: 15px;">
                    <span style="font-size: 2.5em; font-weight: bold; color: #2c3e50;">{{ data.comparison.current_perf }}%</span>
                    <br>
                    <span style="color: #7f8c8d;">Ø Historisch: {{ data.comparison.historical_avg }}%</span><br>
                    <span style="font-weight: bold; color: {{ 'green' if data.comparison.status == 'BETTER' else 'red' }}">{{ data.comparison.status }}</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="section-title">Monatliche Returns (%)</div>
            <table class="heatmap" style="margin-top: 15px;">
                <thead>
                    <tr>
                        <th>Jahr</th>
                        {% for m in range(1, 13) %}<th>{{ m }}</th>{% endfor %}
                    </tr>
                </thead>
                <tbody>
                    {% for year in data.years %}
                    <tr>
                        <td><b>{{ year }}</b></td>
                        {% for m in range(1, 13) %}
                            {% set val = data.monthly_matrix.get(year, {}).get(m, 0) %}
                            {% set count = data.monthly_counts.get(year, {}).get(m, 0) %}

                            {% set cls = 'neutral' %}
                            {% if count > 0 %}
                                {% if val > 5 %}{% set cls = 'pos-high' %}
                                {% elif val > 2 %}{% set cls = 'pos-med' %}
                                {% elif val > 0 %}{% set cls = 'pos-low' %}
                                {% elif val < -5 %}{% set cls = 'neg-high' %}
                                {% elif val < -2 %}{% set cls = 'neg-med' %}
                                {% elif val < 0 %}{% set cls = 'neg-low' %}
                                {% endif %}
                            {% endif %}

                            <td class="{{ cls }}">
                                {% if count > 0 %}
                                    {{ val }}%<br><small>({{ count }})</small>
                                {% else %} - {% endif %}
                            </td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="section-title">Letzte 20 Trades</div>
            <table class="trades-table" style="margin-top: 15px;">
                <thead>
                    <tr>
                        <th>Datum</th>
                        <th>Symbol</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Return</th>
                        <th>Grund</th>
                    </tr>
                </thead>
                <tbody>
                    {% for t in data.recent_trades %}
                    <tr>
                        <td>{{ t.date }}</td>
                        <td><b>{{ t.symbol }}</b></td>
                        <td>{{ t.entry }}</td>
                        <td>{{ t.exit }}</td>
                        <td class="{{ t.class }}">{{ t.pct }}</td>
                        <td>{{ t.reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div style="text-align: center; margin-bottom: 40px;">
            <a href="/backtest/dip-buyer" class="btn">Neuer Backtest</a>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, data=data)
