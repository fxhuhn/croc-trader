import json
import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from flask import Blueprint, current_app, render_template, request

from ..types import TradeStatus
from ..database.repositories.signal import SignalRepository
from ..database.repositories.trade import TradeRepository
from ..database.repositories.market import MarketRepository
from ..database.session import DatabaseSession

logger = logging.getLogger(__name__)
views_bp = Blueprint("views", __name__)

def _get_db_path(name="signals"):
    conf = current_app.config["APP_CONFIG"]
    return Path(conf.get_db_path(name)).resolve()

def _get_repo() -> SignalRepository:
    session = DatabaseSession(str(_get_db_path("signals")))
    return SignalRepository(session)

def _get_trade_repo() -> TradeRepository:
    session = DatabaseSession(str(_get_db_path("signals")))
    return TradeRepository(session)

def _get_market_repo() -> MarketRepository:
    session = DatabaseSession(str(_get_db_path("stocks")))
    return MarketRepository(session)

def is_strategy_match(trade: dict, keyword: str) -> bool:
    strat = str(trade.get("strategy", "")).lower()
    return keyword.lower() in strat

def prepare_view_model(trades, market_repo: MarketRepository):
    """Bereitet Trades für die Anzeige auf (Haltedauer, Live-Preis-Fix, PnL %, Visuals)."""
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    for t in trades:
        # 1. Context parsen
        try:
            raw = t.get("signal_context")
            t["ctx"] = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except Exception:
            t["ctx"] = {}

        # 2. Datum formatieren
        entry_date = t.get("entry_date")
        exit_date = t.get("exit_date")
        t["display_entry"] = str(entry_date).split(" ")[0] if entry_date else "-"
        t["display_exit"] = str(exit_date).split(" ")[0] if exit_date else "-"

        # 3. Haltedauer
        t["days_held"] = 0
        if entry_date:
            start = str(entry_date).split(" ")[0]
            end = str(exit_date).split(" ")[0] if exit_date else today_str
            t["days_held"] = market_repo.get_trading_days_count(t["symbol"], start, end)

        # 4. Preis & PnL Berechnung
        entry_price = float(t.get("entry_price") or 0)
        current_price = float(t.get("current_price") or 0)
        size = float(t.get("current_size") or 0)
        
        # FIX: Wenn Active Trade keinen aktuellen Preis hat (0), lade den letzten Close aus Market DB
        if t.get("status") == "ACTIVE" and current_price == 0:
            latest_price = market_repo.get_latest_price(t["symbol"])
            if latest_price:
                current_price = latest_price
                t["current_price"] = latest_price

        # --- Metrics Calculation ---
        t["unrealized_pnl"] = 0.0
        t["pnl_pct"] = 0.0
        t["is_critical"] = False
        t["progress"] = 0 # Für Progress Bar (0 = SL, 100 = Target)

        # Active Trades
        if t.get("status") == "ACTIVE" and entry_price > 0 and current_price > 0:
            t["unrealized_pnl"] = (current_price - entry_price) * size
            t["pnl_pct"] = ((current_price - entry_price) / entry_price) * 100
            
            # SL Warnung & Progress Bar Logic (für Croc)
            sl = float(t.get("current_stop_loss") or 0)
            
            # Target ermitteln (TP3 oder TP1 oder kalkuliert)
            target = 0
            if t["ctx"].get("target_price"): target = float(t["ctx"]["target_price"])
            elif t["ctx"].get("tp3"): target = float(t["ctx"]["tp3"])
            elif t["ctx"].get("tp1"): target = float(t["ctx"]["tp1"])
            
            # Calculation für Progress Bar (Position zwischen SL und Target)
            if sl > 0 and target > 0 and sl != target:
                # Range: SL (0%) bis Target (100%)
                total_range = target - sl
                current_dist = current_price - sl
                pct = (current_dist / total_range) * 100
                t["progress"] = max(0, min(100, pct)) # Clamp 0-100
            
            # Kritischer SL Abstand (< 1%)
            if sl > 0:
                dist = abs(current_price - sl)
                if (dist / current_price) < 0.01:
                    t["is_critical"] = True
            
        # Closed Trades (Fallback PnL)
        if t.get("status") == "CLOSED":
             exit_price = float(t.get("exit_price") or 0)
             if exit_price > 0 and entry_price > 0:
                 if not t.get("realized_pnl"):
                     t["realized_pnl"] = (exit_price - entry_price) * size
                 # Prozentualer Gewinn
                 t["pnl_pct"] = ((exit_price - entry_price) / entry_price) * 100

def get_portfolio_summary(active_trades):
    """Berechnet Kennzahlen für das Cockpit."""
    total_invested = sum((float(t.get("entry_price") or 0) * float(t.get("current_size") or 0)) for t in active_trades)
    total_open_pnl = sum(t.get("unrealized_pnl", 0) for t in active_trades)
    return {
        "invested": total_invested,
        "open_pnl": total_open_pnl
    }

# --- ROUTES ---

# 1. Landing Pages (Übersichten)
@views_bp.route("/screener", methods=["GET"])
def view_screener_overview() -> str:
    repo = _get_repo()
    
    # Signale zählen (wir laden die Liste und nehmen die Länge, 
    # für EOD/geringe Datenmengen ist das performant genug)
    count_croc = len(repo.get_trade_candidates("Croc", limit=100))
    count_dip = len(repo.get_trade_candidates("DipBuyer", limit=100))
    count_turnover = len(repo.get_trade_candidates("TurnoverTiming", limit=100))
    
    return render_template(
        "screener.html",
        count_croc=count_croc,
        count_dip=count_dip,
        count_turnover=count_turnover
    )

def generate_sparkline(dates: list, prices: list, is_up: bool) -> str:
    """Generates a minimalistic sparkline chart (Spline, No Axes)."""
    color = '#10b981' if is_up else '#ef4444' # Emerald-500 or Rose-500
    fill_color = 'rgba(16, 185, 129, 0.1)' if is_up else 'rgba(239, 68, 68, 0.1)'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        line=dict(color=color, width=2, shape='spline', smoothing=1.3),
        fill='tozeroy',
        fillcolor=fill_color,
        hoverinfo='skip'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=50,
        width=120
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

def generate_donut_chart(labels: list, values: list, colors: list) -> str:
    """Generates a clean donut chart for strategy allocation."""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.8,
        textinfo='none',
        hoverinfo='label+percent+value',
        marker=dict(colors=colors),
        sort=False
    )])

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=180,
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})

@views_bp.route("/trades", methods=["GET"])
def view_trades_overview() -> str:
    trade_repo = _get_trade_repo()
    market_repo = _get_market_repo()
    
    # 1. Fetch Active Trades
    active = trade_repo.get_by_status([TradeStatus.ACTIVE])
    prepare_view_model(active, market_repo) # Calculates PnL, Price etc.
    
    # 2. Portfolio Metrics
    total_invested = sum((float(t.get("entry_price") or 0) * float(t.get("current_size") or 0)) for t in active)
    total_open_pnl = sum(t.get("unrealized_pnl", 0) for t in active)
    active_count = len(active)
    
    # 3. Strategy Allocation & Performance
    strategy_stats = {}
    
    for t in active:
        strat = t.get("strategy", "Unknown")
        # Normalize strategy names if needed (e.g. remove versions)
        if "Croc" in strat: label = "Croc Setup"
        elif "DipBuyer" in strat: label = "Dip Buyer"
        elif "Turnover" in strat: label = "Turnover"
        else: label = strat
        
        if label not in strategy_stats:
            strategy_stats[label] = {"count": 0, "pnl": 0.0, "invested": 0.0}
            
        strategy_stats[label]["count"] += 1
        strategy_stats[label]["pnl"] += t.get("unrealized_pnl", 0)
        entry = float(t.get("entry_price") or 0)
        size = float(t.get("current_size") or 0)
        strategy_stats[label]["invested"] += (entry * size)

    # Prepare Data for Donut Chart
    alloc_labels = list(strategy_stats.keys())
    alloc_values = [d["invested"] for d in strategy_stats.values()]
    # Custom colors: Blue, Purple, Orange, Slate...
    palette = ['#2563eb', '#8b5cf6', '#f97316', '#64748b'] 
    
    donut_html = generate_donut_chart(alloc_labels, alloc_values, palette)
    
    # 4. Generate Sparklines for Active Trades
    # Get history for all active symbols (last 14 days)
    today = pd.Timestamp.now()
    start_date = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    symbols = [t["symbol"] for t in active]
    
    history_df = market_repo.get_batch_history_raw(symbols, start_date, today.strftime("%Y-%m-%d"))
    
    for t in active:
        sym = t["symbol"]
        # Filter df for symbol
        rows = history_df[history_df["symbol"] == sym].sort_values("date")
        if not rows.empty:
            dates = rows["date"].tolist()
            prices = rows["close"].tolist()
            
            # Trend determination (simple)
            is_up = prices[-1] >= prices[0] if prices else True
            if t.get("unrealized_pnl", 0) < 0: is_up = False # Visual match with PnL
            
            t["sparkline"] = generate_sparkline(dates, prices, is_up)
        else:
            t["sparkline"] = ""

    summary = {
        "invested": total_invested,
        "open_pnl": total_open_pnl,
        "count": active_count
    }
    
    return render_template(
        "trades.html", 
        active_trades=active, 
        summary=summary,
        strategy_stats=strategy_stats,
        donut_html=donut_html
    )


# 2. Screener Details
@views_bp.route("/screener/croc", methods=["GET"])
def view_screener_croc() -> str:
    limit = request.args.get("limit", 200, type=int)
    repo = _get_repo()
    results = repo.get_trade_candidates("Croc", limit=limit)
    
    # --- FIX: Datum aus Context extrahieren für korrekte Anzeige ---
    processed_results = []
    for row in results:
        # Row zu Dict konvertieren, damit wir sie ändern können
        item = dict(row)
        try:
            # Signal Context parsen
            raw_ctx = item.get("signal_context")
            ctx = json.loads(raw_ctx) if raw_ctx and isinstance(raw_ctx, str) else (raw_ctx or {})
            
            # Wenn ein echtes Signal-Datum existiert, überschreiben wir 'created_at' für die Anzeige
            if ctx.get("date"):
                # ISO-String (z.B. "2026-01-16T...") zu Datum ("2026-01-16")
                signal_date = str(ctx["date"]).split("T")[0].split(" ")[0]
                item["created_at"] = signal_date
                
            item["ctx"] = ctx # Context für Template verfügbar machen
        except Exception as e:
            logger.warning(f"Fehler beim Parsen des Context für {item.get('symbol')}: {e}")
            item["ctx"] = {}
            
        processed_results.append(item)
        
    return render_template("screener_croc.html", results=processed_results)

@views_bp.route("/screener/dip-buyer", methods=["GET"])
def view_screener_dip_buyer() -> str:
    limit = request.args.get("limit", 100, type=int)
    repo = _get_repo()
    results = repo.get_trade_candidates("DipBuyer", limit=limit)
    
    # Auch hier sicherheitshalber in Dicts wandeln
    processed_results = []
    for row in results:
        item = dict(row)
        try:
            raw = item.get("signal_context") or item.get("ctx")
            item["ctx"] = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        except Exception:
            item["ctx"] = {}
        processed_results.append(item)
        
    return render_template("screener_dip_buyer.html", results=processed_results)

@views_bp.route("/screener/turnover", methods=["GET"])
def view_screener_turnover() -> str:
    limit = request.args.get("limit", 200, type=int)
    repo = _get_repo()
    # 1. Fetch Raw Candidates (e.g. NVDA_0.5, NVDA_1.0)
    results = repo.get_trade_candidates("TurnoverTiming", limit=limit)
    
    # 2. Aggregation Logic
    aggregated = {}
    
    for row in results:
        symbol = row["symbol"]
        
        # Ensure base structure
        if symbol not in aggregated:
            aggregated[symbol] = {
                "symbol": symbol,
                "display_date": row["display_date"],
                "entry_0_5": None,
                "entry_1_0": None,
                "close": 0.0,
                "atr": 0.0
            }
            
        # Parse Context (where setup data lives)
        try:
            # Already parsed in get_trade_candidates? Check repo code.
            # get_trade_candidates calls "r['ctx'] = ctx", so we can use row['ctx']
            ctx = row.get("ctx", {})
            
            # Extract common metrics (should be same for both variants)
            aggregated[symbol]["close"] = float(ctx.get("setup_close", 0))
            aggregated[symbol]["atr"] = float(ctx.get("setup_atr", 0))
            
            # Extract Index (Bucket)
            raw_bucket = ctx.get("bucket", "UNKNOWN")
            # Nice formatting: NASDAQ_100 -> NASDAQ 100
            aggregated[symbol]["bucket"] = raw_bucket.replace("_", " ") if raw_bucket else "-"
            
            # Identify variant based on strategy name pattern "TurnoverTiming_0.5"
            strat_name = row["strategy"]
            entry_price = float(row.get("entry_price") or 0)
            
            if "_0.5" in strat_name:
                aggregated[symbol]["entry_0_5"] = entry_price
            elif "_1.0" in strat_name:
                aggregated[symbol]["entry_1_0"] = entry_price
                
        except Exception as e:
            logger.warning(f"Error Aggregating Turnover {symbol}: {e}")

    # Convert dict to list for template
    final_list = list(aggregated.values())
    
    return render_template("screener_turnover.html", results=final_list)


# 3. Trades Details
@views_bp.route("/trades/croc", methods=["GET"])
def view_trades_croc() -> str:
    limit = request.args.get("limit", 100, type=int)
    trade_repo = _get_trade_repo()
    market_repo = _get_market_repo()
    
    all_active = trade_repo.get_by_status([TradeStatus.ACTIVE])
    all_closed = trade_repo.get_by_status([TradeStatus.CLOSED])
    
    # Filterung
    active = [t for t in all_active if is_strategy_match(t, "croc")]
    closed = [t for t in all_closed if is_strategy_match(t, "croc")]
    
    # Sortierung
    closed.sort(key=lambda x: x.get("exit_date") or "", reverse=True)
    
    # Datenaufbereitung (Dates, PnL Calculation etc.)
    prepare_view_model(active, market_repo)
    prepare_view_model(closed, market_repo)
    
    # Summary für AKTIVE Trades
    summary = get_portfolio_summary(active)

    # NEU: Berechnung Summary für GESCHLOSSENE Trades
    total_closed_pnl = sum(float(t.get("realized_pnl") or 0) for t in closed)
    closed_count = len(closed)
    avg_pnl = total_closed_pnl / closed_count if closed_count > 0 else 0

    closed_summary = {
        "total_pnl": total_closed_pnl,
        "count": closed_count,
        "avg_pnl": avg_pnl
    }

    return render_template(
        "trades_croc.html", 
        active_trades=active, 
        closed_trades=closed[:limit],
        summary=summary,
        closed_summary=closed_summary  # Hier übergeben wir die berechneten Werte
    )

@views_bp.route("/trades/dip-buyer", methods=["GET"])
def view_trades_dip_buyer() -> str:
    limit = request.args.get("limit", 100, type=int)
    trade_repo = _get_trade_repo()
    market_repo = _get_market_repo()
    
    active = [t for t in trade_repo.get_by_status([TradeStatus.ACTIVE]) if is_strategy_match(t, "dipbuyer")]
    closed = [t for t in trade_repo.get_by_status([TradeStatus.CLOSED]) if is_strategy_match(t, "dipbuyer")]
    
    closed.sort(key=lambda x: x.get("exit_date") or "", reverse=True)
    
    prepare_view_model(active, market_repo)
    prepare_view_model(closed, market_repo)
    
    summary = get_portfolio_summary(active)

    # NEU: Statistik für geschlossene Trades berechnen (analog zu Croc)
    total_closed_pnl = sum(float(t.get("realized_pnl") or 0) for t in closed)
    closed_count = len(closed)
    avg_pnl = total_closed_pnl / closed_count if closed_count > 0 else 0

    closed_summary = {
        "total_pnl": total_closed_pnl,
        "count": closed_count,
        "avg_pnl": avg_pnl
    }

    return render_template(
        "trades_dip_buyer.html", 
        active_trades=active, 
        closed_trades=closed[:limit],
        summary=summary,
        closed_summary=closed_summary # Wichtig für das neue Template
    )


@views_bp.route("/trades/turnover", methods=["GET"])
def view_trades_turnover() -> str:
    limit = request.args.get("limit", 200, type=int)
    trade_repo = _get_trade_repo()
    market_repo = _get_market_repo()
    
    all_active = trade_repo.get_by_status([TradeStatus.ACTIVE])
    all_closed = trade_repo.get_by_status([TradeStatus.CLOSED])
    
    # Filtern nach Strategie-Key
    active = [t for t in all_active if "TurnoverTiming" in t['strategy']]
    closed = [t for t in all_closed if "TurnoverTiming" in t['strategy']]
    
    # Sortierung: Datum (Prio 1) + Symbol (Prio 2) für saubere Gruppierung im Template
    active.sort(key=lambda x: (x.get("entry_date") or "", x.get("symbol")), reverse=True)
    closed.sort(key=lambda x: (x.get("exit_date") or "", x.get("symbol")), reverse=True)
    
    prepare_view_model(active, market_repo)
    prepare_view_model(closed, market_repo)
    
    summary = get_portfolio_summary(active)

    # NEU: Statistik für geschlossene Turnover Trades
    total_closed_pnl = sum(float(t.get("realized_pnl") or 0) for t in closed)
    closed_count = len(closed)
    avg_pnl = total_closed_pnl / closed_count if closed_count > 0 else 0

    closed_summary = {
        "total_pnl": total_closed_pnl,
        "count": closed_count,
        "avg_pnl": avg_pnl
    }

    return render_template(
        "trades_turnover.html", 
        active_trades=active, 
        closed_trades=closed[:limit],
        summary=summary,
        closed_summary=closed_summary
    )

@views_bp.route("/backtest", methods=["GET"])
def view_backtest_dashboard() -> str:
    """Displays the Backtest Dashboard."""
    # Paths
    bt_db = str(_get_db_path("backtest").parent / "backtest.db")
    mkt_db = str(_get_db_path("stocks"))
    
    # 1. Analytics
    from ..services.backtester.analytics import BacktestAnalytics
    from ..services.backtester.charts import generate_backtest_charts, generate_profit_factor_gauge, generate_win_rate_gauge, generate_sqn_gauge
    
    analytics = BacktestAnalytics(bt_db, mkt_db)
    metrics = analytics.run_analysis()
    
    # 2. Charts
    df_equity = analytics.get_equity_curve()
    chart_eq, chart_dd = generate_backtest_charts(df_equity)
    
    # Gauge for Profit Factor
    chart_pf = generate_profit_factor_gauge(metrics.profit_factor)
    
    # Gauge for Win Rate
    chart_wr = generate_win_rate_gauge(metrics.win_rate * 100)
    
    # Gauge for SQN
    chart_sqn = generate_sqn_gauge(metrics.sqn)
    
    # 3. Trade Lists (Recent, Top, Worst)
    trade_lists = analytics.get_trade_lists()
    
    # 4. Strategy Breakdown
    strategy_metrics = analytics.run_strategy_analysis()
    
    return render_template(
        "backtest_dashboard.html",
        metrics=metrics,
        strategy_metrics=strategy_metrics,
        chart_equity=chart_eq,
        chart_drawdown=chart_dd,
        chart_pf=chart_pf,
        chart_wr=chart_wr,
        chart_sqn=chart_sqn,
        recent_trades=trade_lists['recent'],
        top_trades=trade_lists['top'],
        worst_trades=trade_lists['worst']
    )