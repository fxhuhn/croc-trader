# app/web/templates_raw.py

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
        <title>🐊 Croc Trades (Pending)</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background: #f8f9fa; }
            h1 { color: #27ae60; border-bottom: 2px solid #27ae60; padding-bottom: 10px; display: inline-block; }
            .container { max-width: 1400px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
            th { background-color: #27ae60; color: white; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; }
            tr:hover { background-color: #f1f8e9; }
            
            .price { font-family: monospace; font-weight: bold; font-size: 1.1em; }
            .score-high { color: #27ae60; font-weight: bold; }
            .score-med { color: #f39c12; font-weight: bold; }
            .score-low { color: #c0392b; font-weight: bold; }
            
            .badge { padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .bg-idx { background: #e9ecef; color: #495057; }
            
            a.tv-link { text-decoration: none; color: #2c3e50; font-weight: bold; display: inline-flex; align-items: center; }
            a.tv-link:hover { color: #27ae60; }
            
            .meta-info { font-size: 0.85em; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐊 Croc Signale (Pending)</h1>
            <p>Aktuelle Setup-Kandidaten aus der Datenbank (Status: CREATED).</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Datum</th>
                        <th>Symbol</th>
                        <th>Index</th>
                        <th>Strategie</th>
                        <th>Score</th>
                        <th>Phase</th>
                        <th>Entry (Stop Buy)</th>
                        <th>Stop Loss</th>
                        <th>Risiko</th>
                        <th>TP1 / TP3</th>
                        <th>Signal Info</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in results %}
                    <tr>
                        <td>{{ row['display_date'] }}</td>
                        <td>
                            <a href="https://www.tradingview.com/chart/?symbol={{ row['symbol'] }}" class="tv-link" target="_blank">
                                {{ row['symbol'] }} ↗
                            </a>
                        </td>
                        <td><span class="badge bg-idx">{{ row['ctx'].get('indices', '-') }}</span></td>
                        <td style="font-weight:bold; color:#2980b9;">{{ row['strategy'] }}</td>
                        <td>
                            {% set s = row['setup_score']|float %}
                            <span class="{{ 'score-high' if s >= 7 else 'score-med' if s >= 4 else 'score-low' }}">
                                {{ s }}
                            </span>
                        </td>
                        <td class="meta-info">{{ row['market_phase'] }}</td>
                        
                        <td class="price" style="color: #27ae60;">{{ row['entry_price'] }}</td>
                        <td class="price" style="color: #c0392b;">{{ row['current_stop_loss'] }}</td>
                        
                        {% set risk = (row['entry_price'] - row['current_stop_loss']) | round(2) %}
                        <td style="font-size:0.9em;">
                            {{ risk }} $ <br> 
                            <span style="color:#999;">(1R)</span>
                        </td>
                        
                        {% set match = row['ctx'].get('match_rule', {}) %}
                        <td class="price" style="color: #2980b9;">
                            {% if row['ctx'].get('tp1') %}
                                {{ row['ctx']['tp1'] }} / {{ row['ctx']['tp3'] }}
                            {% else %}
                                <span style="font-size:0.8em; color:#999;">Auto (Split)</span>
                            {% endif %}
                        </td>
                        
                        <td class="meta-info">
                            {{ row['ctx'].get('original_signal', '-') }}<br>
                            {% if match.get('R_2026') %}R26: {{ match.get('R_2026') }}{% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """,

    "dip_buyer": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>📉 DipBuyer Signale</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background: #f0f2f5; }
            h1 { color: #2980b9; border-bottom: 2px solid #2980b9; padding-bottom: 10px; display: inline-block; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
            th { background-color: #2980b9; color: white; text-transform: uppercase; font-size: 0.85em; letter-spacing: 0.5px; }
            tr:hover { background-color: #eaf2f8; }
            
            .price { font-family: monospace; font-weight: bold; font-size: 1.1em; }
            .limit-buy { color: #27ae60; font-weight: bold; }
            .limit-loc { color: #d35400; font-weight: bold; }
            
            a.tv-link { text-decoration: none; color: #2c3e50; font-weight: bold; display: inline-flex; align-items: center; }
            a.tv-link:hover { color: #2980b9; }
            
            .atr-badge { background: #eee; color: #555; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
            .badge-idx { background: #d6eaf8; color: #2471a3; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📉 DipBuyer Signale (Pending)</h1>
            <p>Limit-Order Kandidaten für den nächsten Handelstag.</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Datum</th>
                        <th>Symbol</th>
                        <th>Index</th>
                        <th>Entry (Limit)</th>
                        <th>LOC</th>
                        <th>Score</th>
                        <th>Close</th>
                        <th>ATR (5)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in results %}
                    <tr>
                        <td>{{ row['display_date'] }}</td>
                        <td>
                            <a href="https://www.tradingview.com/chart/?symbol={{ row['symbol'] }}" class="tv-link" target="_blank">
                                {{ row['symbol'] }} ↗
                            </a>
                        </td>
                        <td><span class="badge-idx">{{ row['ctx'].get('indices', '-') }}</span></td>
                        <td class="price limit-buy">{{ "%.2f"|format(row['entry_price']) if row['entry_price'] else '-' }} $</td>
                        <td class="price limit-loc">
                            {% if row['ctx'].get('threshold_loc') %}
                                {{ "%.2f"|format(row['ctx']['threshold_loc']) }}
                            {% else %}-{% endif %}
                        </td>
                        <td><b>{{ row['setup_score'] }}</b></td>
                        <td class="price" style="color:#777;">{{ "%.2f"|format(row['ctx'].get('close')) if row['ctx'].get('close') else '-' }}</td>
                        <td><span class="atr-badge">{{ row['ctx'].get('atr5', '-') }}</span></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """,

    # --- CROC TRADES ---
    "trades_croc": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>🐊 Croc Trades (Active/Closed)</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background: #f8f9fa; }
            h1 { color: #27ae60; margin: 0; display: inline-block; border-bottom: 2px solid #27ae60; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 40px; margin-bottom: 15px; font-size: 1.4em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .container-fluid { max-width: 1600px; }
            
            .card { border: none; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; margin-bottom: 20px; }
            .card-header { background-color: #2c3e50; color: white; font-weight: 600; padding: 12px 20px; }
            
            table { margin-bottom: 0 !important; }
            th { background-color: #f8f9fa; color: #7f8c8d; text-transform: uppercase; font-size: 0.85em; font-weight: 600; letter-spacing: 0.5px; border-bottom: 2px solid #eee; vertical-align: middle; }
            td { vertical-align: middle; font-size: 0.95rem; }
            
            .price { font-family: 'Consolas', monospace; font-weight: 600; }
            .pos { color: #27ae60 !important; font-weight: bold; }
            .neg { color: #c0392b !important; font-weight: bold; }
            .pct { font-size: 0.85em; color: #666; font-weight: normal; }
            
            .badge-signal { background-color: #e8f6f3; color: #16a085; border: 1px solid #1abc9c; font-weight: normal; }
            .badge-strat { background-color: #f0f2f5; color: #555; border: 1px solid #dcdcdc; font-weight: normal; font-size: 0.8em; }
            
            a.tv-link { text-decoration: none; color: #2c3e50; font-weight: bold; }
            a.tv-link:hover { color: #27ae60; }
            
            .text-muted-small { font-size: 0.85em; color: #999; }
            
            .blink { animation: blinker 1.5s linear infinite; color: #e74c3c !important; }
            @keyframes blinker { 50% { opacity: 0.5; } }
            
            .progress-container { width: 100px; height: 6px; background: #eee; border-radius: 3px; overflow:hidden; margin-top: 4px; }
            .progress-bar { height: 100%; background: linear-gradient(90deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%); }
            .progress-marker { height: 100%; width: 4px; background: black; position: relative; top: -6px; }
        </style>
    </head>
    <body>
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <div>
                <h1>🐊 Croc Strategy Dashboard</h1>
            </div>
            <div>
                <a href="/screener/croc" class="btn btn-outline-success me-2">Zu den Signalen</a>
                <a href="/" class="btn btn-secondary">Home</a>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-white text-center p-3">
                    <small class="text-muted">Investiertes Kapital</small>
                    <h3 class="mb-0">{{ "{:,.0f}".format(summary['invested']) }} $</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-white text-center p-3">
                    <small class="text-muted">Offener PnL</small>
                    <h3 class="mb-0 {{ 'pos' if summary['open_pnl'] >= 0 else 'neg' }}">
                        {{ "%+.2f"|format(summary['open_pnl']) }} $
                    </h3>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
                <span>🚀 Aktive Positionen ({{ active_trades|length }})</span>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover table-striped mb-0">
                        <thead>
                            <tr>
                                <th>Entry Date</th>
                                <th>Symbol</th>
                                <th>Signal</th>
                                <th>Exit-Strategy</th>
                                <th class="text-end">Size</th>
                                <th class="text-end">Entry $</th>
                                <th class="text-end">Aktuell / Target</th>
                                <th class="text-end">SL $</th>
                                <th class="text-end">Open PnL</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for t in active_trades %}
                            <tr>
                                <td>{{ t['entry_date'].split(' ')[0] if t['entry_date'] else '-' }}</td>
                                <td>
                                    <a href="https://www.tradingview.com/chart/?symbol={{ t['symbol'] }}" class="tv-link" target="_blank">
                                        {{ t['symbol'] }} ↗
                                    </a>
                                </td>
                                <td>
                                    <span class="badge badge-signal">{{ t['ctx'].get('original_signal', '-') }}</span>
                                </td>
                                <td>
                                    <span class="badge badge-strat">{{ t['strategy'] }}</span>
                                </td>
                                <td class="text-end">{{ t['current_size']|int }}</td>
                                <td class="text-end price">{{ "%.2f"|format(t['entry_price'] or 0) }}</td>
                                
                                <td class="text-end price">
                                    {{ "%.2f"|format(t['current_price'] or 0) }}
                                    <div class="progress-container" title="Position relative to SL/TP">
                                        <div class="progress-bar" style="width: {{ t['progress'] }}%;"></div>
                                    </div>
                                    <small class="text-muted">TP: 
                                        {% if t['ctx'].get('tp3') %}{{ "%.1f"|format(t['ctx']['tp3']) }}
                                        {% elif t['ctx'].get('target_price') %}{{ "%.1f"|format(t['ctx']['target_price']) }}
                                        {% else %} - {% endif %}
                                    </small>
                                </td>
                                
                                <td class="text-end price {{ 'text-danger fw-bold blink' if t['is_critical'] else 'text-danger' }}">
                                    {{ "%.2f"|format(t['current_stop_loss'] or 0) }}
                                </td>
                                <td class="text-end">
                                    <span class="{{ 'pos' if t['unrealized_pnl'] >= 0 else 'neg' }}">{{ "%+.2f"|format(t['unrealized_pnl']) }} $</span>
                                    <br><span class="pct">({{ "%+.2f"|format(t['pnl_pct']) }}%)</span>
                                </td>
                                <td>
                                    {% if t['ctx'].get('is_phase_2') %}
                                        <span class="badge bg-success">Risk Free (Phase 2)</span>
                                    {% else %}
                                        <span class="badge bg-primary">Running (Phase 1)</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% else %}
                            <tr><td colspan="10" class="text-center py-4 text-muted">Keine offenen Positionen</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                💰 Abgeschlossene Trades (Letzte {{ closed_trades|length }})
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover table-sm mb-0 align-middle">
                        <thead>
                            <tr>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>Symbol</th>
                                <th>Signal</th>
                                <th>Strategy</th>
                                <th class="text-end">Entry $</th>
                                <th class="text-end">Exit $</th>
                                <th class="text-end">PnL $</th>
                                <th>Grund</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for t in closed_trades %}
                            <tr>
                                <td class="text-muted-small">{{ t['entry_date'].split(' ')[0] if t['entry_date'] else '-' }}</td>
                                <td>{{ t['exit_date'].split(' ')[0] if t['exit_date'] else '-' }}</td>
                                <td class="fw-bold">
                                    <a href="https://www.tradingview.com/chart/?symbol={{ t['symbol'] }}" class="tv-link" target="_blank" style="color:#333;">
                                        {{ t['symbol'] }}
                                    </a>
                                </td>
                                <td><span class="badge badge-signal">{{ t['ctx'].get('original_signal', '-') }}</span></td>
                                <td class="text-muted-small">{{ t['strategy'] }}</td>
                                <td class="text-end price text-muted">{{ "%.2f"|format(t['entry_price'] or 0) }}</td>
                                <td class="text-end price">{{ "%.2f"|format(t['exit_price'] or 0) }}</td>
                                <td class="text-end">
                                    {% set pnl = t['realized_pnl']|float %}
                                    <span class="{{ 'pos' if pnl >= 0 else 'neg' }}">
                                        {{ "%+.2f"|format(pnl) }} $
                                    </span>
                                </td>
                                <td>
                                    {% set reason = t['exit_reason'] or '-' %}
                                    {% if pnl > 0 %}
                                        <span class="badge bg-success">{{ reason }}</span>
                                    {% elif pnl < 0 %}
                                        <span class="badge bg-danger">{{ reason }}</span>
                                    {% else %}
                                        <span class="badge bg-secondary">{{ reason }}</span>
                                    {% endif %}
                                </td>
                            </tr>
                            {% else %}
                            <tr><td colspan="9" class="text-center py-4 text-muted">Keine Historie vorhanden</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

    </div>
    </body>
    </html>
    """,

    "trades_dip_buyer": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>📉 DipBuyer Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; color: #333; background: #f0f2f5; }
            h1 { color: #2980b9; margin: 0; }
            h2 { color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-size: 1.4em; }
            .container { max-width: 1400px; margin: 0 auto; }
            
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-top: 10px; border-radius: 6px; overflow: hidden; }
            th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9em; }
            th { background-color: #2980b9; color: white; text-transform: uppercase; font-size: 0.85em; }
            tr:hover { background-color: #eaf2f8; }
            
            .price { font-family: monospace; font-weight: bold; }
            .pos { color: #27ae60 !important; font-weight: bold; }
            .neg { color: #c0392b !important; font-weight: bold; }
            .pct { font-size: 0.85em; color: #666; font-weight: normal; }
            .badge-loc { background-color: #eee; color: #555; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <div class="container-fluid" style="max-width: 1400px;">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1>📉 DipBuyer Trades</h1>
                <div>
                    <a href="/screener/dip-buyer" class="btn btn-outline-primary me-2">Zu den Signalen</a>
                    <a href="/" class="btn btn-secondary">Home</a>
                </div>
            </div>

            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card bg-white text-center p-3">
                        <small class="text-muted">Investiertes Kapital</small>
                        <h3 class="mb-0">{{ "{:,.0f}".format(summary['invested']) }} $</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-white text-center p-3">
                        <small class="text-muted">Offener PnL</small>
                        <h3 class="mb-0 {{ 'pos' if summary['open_pnl'] >= 0 else 'neg' }}">
                            {{ "%+.2f"|format(summary['open_pnl']) }} $
                        </h3>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">🚀 Aktive Positionen ({{ active_trades|length }})</div>
                <div class="card-body p-0">
                    <table class="table table-hover mb-0 align-middle">
                        <thead>
                            <tr>
                                <th>Entry Datum</th><th>Symbol</th><th>Größe</th><th>Tage</th>
                                <th class="text-end">Entry $</th><th class="text-end">Aktuell $</th>
                                <th class="text-end">LOC</th><th class="text-end">Ziel (TP)</th><th class="text-end">Open PnL</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for t in active_trades %}
                            <tr>
                                <td>{{ t['entry_date'] }}</td>
                                <td>
                                    <a href="https://www.tradingview.com/chart/?symbol={{ t['symbol'] }}" class="tv-link" target="_blank">
                                        {{ t['symbol'] }} ↗
                                    </a>
                                </td>
                                <td>{{ t['current_size']|int }}</td>
                                <td>{{ t['days_held'] }}</td>
                                <td class="text-end price">{{ "%.2f"|format(t['entry_price']) }}</td>
                                <td class="text-end price">{{ "%.2f"|format(t['current_price'] or 0) }}</td>
                                <td class="text-end">
                                    {% if t['ctx'].get('threshold_loc') %}
                                        <span class="badge badge-loc">< {{ "%.2f"|format(t['ctx']['threshold_loc']) }}</span>
                                    {% else %} - {% endif %}
                                </td>
                                <td class="text-end price text-success">{{ "%.2f"|format(t['current_target']) if t['current_target'] else '-' }}</td>
                                <td class="text-end">
                                    <span class="{{ 'pos' if t['unrealized_pnl'] >= 0 else 'neg' }}">
                                        {{ "%+.2f"|format(t['unrealized_pnl']) }} $
                                    </span>
                                    <br><span class="pct">({{ "%+.2f"|format(t['pnl_pct']) }}%)</span>
                                </td>
                            </tr>
                            {% else %}
                            <tr><td colspan="9" class="text-center text-muted py-3">Keine offenen Positionen</td></tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-header bg-secondary text-white">🏁 Abgeschlossene Trades (Letzte {{ closed_trades|length }})</div>
                <div class="card-body p-0">
                    <table class="table table-hover mb-0 align-middle">
                        <thead>
                            <tr>
                                <th>Entry</th><th>Exit</th><th>Symbol</th><th>Tage</th>
                                <th class="text-end">Entry $</th><th class="text-end">Exit $</th><th class="text-end">PnL</th><th>Grund</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for t in closed_trades %}
                            <tr>
                                <td>{{ t['display_entry'] }}</td>
                                <td>{{ t['display_exit'] }}</td>
                                <td><b>{{ t['symbol'] }}</b></td>
                                <td>{{ t['days_held'] }}</td>
                                <td class="text-end price text-muted">{{ "%.2f"|format(t['entry_price']) }}</td>
                                <td class="text-end price">{{ "%.2f"|format(t['exit_price']) }}</td>
                                <td class="text-end">
                                    <span class="{{ 'pos' if t['realized_pnl'] >= 0 else 'neg' }}">
                                        {{ "%+.2f"|format(t['realized_pnl']) }} $
                                    </span>
                                    <br><span class="pct">({{ "%+.2f"|format(t['pnl_pct']) }}%)</span>
                                </td>
                                <td><span class="badge bg-light text-dark">{{ t['exit_reason'] }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """,

    # --- NEW: TURNOVER TRADES DASHBOARD ---
    "trades_turnover": """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Turnover Timing Trades</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f8f9fa; }
            .card-header { font-weight: bold; }
            .pos { color: #27ae60; font-weight: bold; }
            .neg { color: #c0392b; font-weight: bold; }
            .price { font-family: monospace; }
            .pct { font-size: 0.85em; color: #666; font-weight: normal; }
            .badge-strat { background-color: #e9ecef; color: #495057; border: 1px solid #ced4da; }
            
            /* Gruppierung der Trades */
            .group-start td { border-top: 3px solid #dfe6ed !important; }
        </style>
    </head>
    <body>
    <div class="container-fluid" style="max-width: 1600px;">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1>Turnover Timing Dashboard</h1>
            <a href="/" class="btn btn-secondary">Home</a>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card bg-white text-center p-3">
                    <small class="text-muted">Investiertes Kapital</small>
                    <h3 class="mb-0">{{ "{:,.0f}".format(summary['invested']) }} $</h3>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card bg-white text-center p-3">
                    <small class="text-muted">Offener PnL</small>
                    <h3 class="mb-0 {{ 'pos' if summary['open_pnl'] >= 0 else 'neg' }}">
                        {{ "%+.2f"|format(summary['open_pnl']) }} $
                    </h3>
                </div>
            </div>
        </div>

        <div class="card mb-4">
            <div class="card-header bg-primary text-white">🚀 Laufende Trades ({{ active_trades|length }})</div>
            <div class="card-body p-0">
                <table class="table table-hover mb-0">
                    <thead>
                        <tr>
                            <th>Entry</th><th>Symbol</th><th>Strategie</th><th>Size</th>
                            <th>Entry $</th><th>Aktuell $</th><th>Open PnL</th>
                            <th>Setup (Fri)</th><th>Tage</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for t in active_trades %}
                    <tr class="{{ 'group-start' if loop.index > 1 and (t['entry_date'] != active_trades[loop.index0 - 1]['entry_date'] or t['symbol'] != active_trades[loop.index0 - 1]['symbol']) }}">
                        <td>{{ t['display_entry'] }}</td>
                        <td><b>{{ t['symbol'] }}</b></td>
                        <td><span class="badge badge-strat">{{ t['strategy'] }}</span></td>
                        <td>{{ t['current_size']|int }}</td>
                        <td class="price">{{ "%.2f"|format(t['entry_price']) }}</td>
                        <td class="price">{{ "%.2f"|format(t['current_price'] or 0) }}</td>
                        <td>
                            <span class="{{ 'pos' if t['unrealized_pnl'] >= 0 else 'neg' }}">
                                {{ "%+.2f"|format(t['unrealized_pnl']) }} $
                            </span>
                            <br><span class="pct">({{ "%+.2f"|format(t['pnl_pct']) }}%)</span>
                        </td>
                        <td>
                            {% if t['ctx'].get('setup_candle_green') %}
                                <span class="badge bg-success">Grün</span>
                            {% else %}
                                <span class="badge bg-danger">Rot</span>
                            {% endif %}
                        </td>
                        <td>{{ t['days_held'] }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="9" class="text-center text-muted py-3">Keine laufenden Trades</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header bg-secondary text-white">🏁 Abgeschlossene Trades (Letzte {{ closed_trades|length }})</div>
            <div class="card-body p-0">
                <table class="table table-hover mb-0">
                    <thead>
                        <tr>
                            <th>Entry</th><th>Exit</th><th>Symbol</th><th>Strategie</th>
                            <th>Size</th><th>Entry $</th><th>Exit $</th><th>PnL</th>
                            <th>Grund</th><th>Tage</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for t in closed_trades %}
                    <tr class="{{ 'group-start' if loop.index > 1 and (t['exit_date'] != closed_trades[loop.index0 - 1]['exit_date'] or t['symbol'] != closed_trades[loop.index0 - 1]['symbol']) }}">
                        <td>{{ t['display_entry'] }}</td>
                        <td>{{ t['display_exit'] }}</td>
                        <td><b>{{ t['symbol'] }}</b></td>
                        <td><span class="badge badge-strat">{{ t['strategy'] }}</span></td>
                        <td>{{ t['current_size']|int }}</td>
                        <td class="price">{{ "%.2f"|format(t['entry_price']) }}</td>
                        <td class="price">{{ "%.2f"|format(t['exit_price']) }}</td>
                        <td>
                            <span class="{{ 'pos' if t['realized_pnl'] >= 0 else 'neg' }}">
                                {{ "%+.2f"|format(t['realized_pnl']) }} $
                            </span>
                            <br><span class="pct">({{ "%+.2f"|format(t['pnl_pct']) }}%)</span>
                        </td>
                        <td>{{ t['exit_reason'] }}</td>
                        <td>{{ t['days_held'] }}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="10" class="text-center text-muted py-3">Keine Historie</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    </body>
    </html>
    """,
    "404": """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>404 Not Found</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 50px; color: #333; }
            h1 { font-size: 50px; margin-bottom: 10px; }
            p { font-size: 20px; color: #666; }
            a { color: #2980b9; text-decoration: none; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>404</h1>
        <p>Page not found.</p>
        <p><a href="/">Go Home</a></p>
    </body>
    </html>
    """,

    "500": """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>500 Internal Error</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 50px; color: #333; }
            h1 { font-size: 50px; margin-bottom: 10px; color: #c0392b; }
            p { font-size: 20px; color: #666; }
            a { color: #2980b9; text-decoration: none; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>500</h1>
        <p>Internal Server Error.</p>
        <p><a href="/">Go Home</a></p>
    </body>
    </html>
    """
}