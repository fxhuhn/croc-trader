# app/routes/templates_raw.py

HTML_TEMPLATES = {
    "webhook": """
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
            a.tv-link { text-decoration: none; color: #009879; font-weight: bold; display: inline-flex; align-items: center; }
            a.tv-link:hover { text-decoration: underline; color: #007f65; }
            .tv-icon { font-size: 0.8em; margin-left: 4px; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>🔎 Webhook Screener Ergebnisse</h1>
        <table>
            <thead>
                <tr>
                    <th>Rank</th><th>Datum</th><th>Symbol</th><th>Exchange</th>
                    <th>Strategie</th><th>Signal</th><th>Kriterien (Filter)</th>
                    <th>Close</th><th>RSI</th><th>SMA 200</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}
                    {% set tv_interval = 'D' if row['timeframe'] == '1D' else row['timeframe'] %}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}
                <tr>
                    <td class="rank-high">#{{ row['rank'] }}</td>
                    <td>{{ row['date'] }}</td>
                    <td><a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} <span class="tv-icon">↗</span></a></td>
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
    """,
    "croc_setup": """
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
            .rank-badge { background: #333; color: #fff; padding: 3px 8px; border-radius: 10px; font-weight: bold; font-size: 0.85em; }
            .r-val { color: #d35400; font-weight: bold; }
            .strat-highlight { color: #2980b9; font-weight: bold; }
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
                    <th>Rank</th><th>Datum</th><th>Symbol</th><th>Signal</th>
                    <th>R / Trade</th><th>Empf. Strategie</th><th>Close</th>
                    <th>RSI</th><th>Dist EMA %</th><th>Auslöser (Filter)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}
                    {% set tv_interval = 'D' if row['timeframe'] == '1D' else row['timeframe'] %}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}
                <tr>
                    <td><span class="rank-badge">#{{ row['rank'] }}</span></td>
                    <td>{{ row['date'] }}</td>
                    <td><a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} ↗</a></td>
                    <td>{{ row['signal'] }}</td>
                    <td class="r-val">{{ row['r_per_trade'] }}</td>
                    <td class="strat-highlight">{{ row['recommended_strategy'] }}</td>
                    <td>{{ row['close'] }}</td>
                    <td>{{ row['rsi']|round(1) if row['rsi'] is not none else '-' }}</td>
                    <td>{{ row['dist_ema'] ~ '%' if row['dist_ema'] is not none else '-' }}</td>
                    <td style="font-style: italic; color: #666;">{{ row['match_filter'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """,
    "dip_buyer": """
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
            a.tv-link { text-decoration: none; color: #2980b9; font-weight: bold; display: inline-flex; align-items: center; }
            a.tv-link:hover { text-decoration: underline; color: #1a5276; }
            .tv-icon { font-size: 0.8em; margin-left: 4px; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>📉 Dip-Buyer Screener Ergebnisse</h1>
        <table>
            <thead>
                <tr>
                    <th>Datum</th><th>Symbol</th><th>Exchange</th><th>Setup Score</th>
                    <th>ATR R3</th><th>Entry Limit</th><th>ATR 5</th><th>Close</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}
                    {% set tv_interval = 'D' if row['timeframe'] == '1D' else row['timeframe'] %}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] ~ "&interval=" ~ tv_interval %}
                <tr>
                    <td>{{ row['date'] }}</td>
                    <td><a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} <span class="tv-icon">↗</span></a></td>
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
    """,
    "turnover": """
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
            .index-badge { background: #eee; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-weight: bold; color: #555; }
            .money { font-family: monospace; color: #27ae60; font-weight: bold; }
            .entry-zone { color: #d35400; font-weight: bold; }
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
                    <th>Datum</th><th>Symbol</th><th>Index (Quelle)</th><th>Turnover SMA20 ($)</th>
                    <th>Close</th><th>ATR(3)</th><th>Entry 1 (-0.5 ATR)</th><th>Entry 2 (-1.0 ATR)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                    {% set exchange_prefix = (row['exchange'] ~ ':') if row['exchange'] and row['exchange'] != 'UNKNOWN' else '' %}
                    {% set tv_url = "https://www.tradingview.com/chart/?symbol=" ~ exchange_prefix ~ row['symbol'] %}
                <tr>
                    <td>{{ row['date'] }}</td>
                    <td><a href="{{ tv_url }}" class="tv-link" target="_blank">{{ row['symbol'] }} ↗</a></td>
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
    """,
    "strategy_trades": """
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
                    <th>Entry Date</th><th>Strategy</th><th>Symbol</th><th>Status</th>
                    <th>Entry Price</th><th>Quantity</th><th>ATR @ Entry</th><th>Exit Reason</th><th>Closed At</th>
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
    """,
    "active_trades_raw": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>DB: Active Trades</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background-color: #f4f4f9; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; display: inline-block; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 10px 12px; text-align: left; font-size: 0.9em; }
            th { background-color: #34495e; color: white; text-transform: uppercase; font-size: 0.85em; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            tr:hover { background-color: #e2e6ea; }
            .status-created { color: #d35400; font-weight: bold; }
            .status-active { color: #27ae60; font-weight: bold; }
            .status-closed { color: #7f8c8d; }
            .status-missed { color: #c0392b; text-decoration: line-through; }
        </style>
    </head>
    <body>
        <h1>🗃️ Tabelle: active_trades</h1>
        <p>Inhalt der Datenbank (Limit: {{ limit }}), sortiert nach <b>Entry Date (DESC)</b>.</p>
        <table>
            <thead>
                <tr>
                    <th>ID</th><th>Symbol</th><th>Entry Date</th><th>Strategy</th><th>Status</th>
                    <th>Entry Price</th><th>ATR</th><th>Qty</th><th>Exit Reason</th>
                    <th>Closed At</th><th>Created At</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row['id'] }}</td>
                    <td><b>{{ row['symbol'] }}</b></td>
                    <td>{{ row['entry_date'] }}</td>
                    <td>{{ row['strategy'] }}</td>
                    <td class="status-{{ row['status']|lower }}">{{ row['status'] }}</td>
                    <td>{{ row['entry_price'] }}</td>
                    <td>{{ row['atr_at_entry'] }}</td>
                    <td>{{ row['quantity'] }}</td>
                    <td>{{ row['exit_reason'] or '' }}</td>
                    <td>{{ row['closed_at'] or '' }}</td>
                    <td style="color: #999; font-size: 0.8em;">{{ row['created_at'] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """,
    "backtest_form": """
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
    """,
    "backtest_report": """
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
            table { width: 100%; border-collapse: collapse; font-size: 0.95em; }
            th, td { padding: 10px; text-align: center; border-bottom: 1px solid #eee; }
            th { background: #ecf0f1; color: #555; }
            .pos-high { background-color: #27ae60; color: white; }
            .pos-med { background-color: #2ecc71; color: white; }
            .pos-low { background-color: #a9dfbf; }
            .neg-low { background-color: #f5b7b1; }
            .neg-med { background-color: #e74c3c; color: white; }
            .neg-high { background-color: #c0392b; color: white; }
            .neutral { background-color: #fff; color: #eee; }
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
                    <tr><th>Jahr</th>{% for m in range(1, 13) %}<th>{{ m }}</th>{% endfor %}</tr>
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
                                {% if val > 5 %}{% set cls = 'pos-high' %}{% elif val > 2 %}{% set cls = 'pos-med' %}{% elif val > 0 %}{% set cls = 'pos-low' %}{% elif val < -5 %}{% set cls = 'neg-high' %}{% elif val < -2 %}{% set cls = 'neg-med' %}{% elif val < 0 %}{% set cls = 'neg-low' %}{% endif %}
                            {% endif %}
                            <td class="{{ cls }}">
                                {% if count > 0 %}{{ val }}%<br><small>({{ count }})</small>{% else %} - {% endif %}
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
                    <tr><th>Datum</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Return</th><th>Grund</th></tr>
                </thead>
                <tbody>
                    {% for t in data.recent_trades %}
                    <tr>
                        <td>{{ t.date }}</td><td><b>{{ t.symbol }}</b></td><td>{{ t.entry }}</td><td>{{ t.exit }}</td>
                        <td class="{{ t.class }}">{{ t.pct }}</td><td>{{ t.reason }}</td>
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
    """,
    "404": """
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
    """,
    "500": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>500 - Server Fehler</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 50px; color: #333; }
            h1 { font-size: 50px; color: #c0392b; margin-bottom: 10px; }
            p { font-size: 18px; color: #666; }
        </style>
    </head>
    <body>
        <h1>500 - Server Fehler</h1>
        <p>Der Croc-Trader hat sich verschluckt. Check die Logs!</p>
    </body>
    </html>
    """,
}
