import json
import logging
from pathlib import Path
import pandas as pd

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
        except: t["ctx"] = {}

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
    return render_template("screener.html")

@views_bp.route("/trades", methods=["GET"])
def view_trades_overview() -> str:
    trade_repo = _get_trade_repo()
    market_repo = _get_market_repo()
    
    # Alle aktiven Trades laden für die Gesamtübersicht
    active = trade_repo.get_by_status([TradeStatus.ACTIVE])
    prepare_view_model(active, market_repo)
    summary = get_portfolio_summary(active)
    
    return render_template("trades.html", active_trades=active, summary=summary)


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
        except: 
            item["ctx"] = {}
        processed_results.append(item)
        
    return render_template("screener_dip_buyer.html", results=processed_results)

@views_bp.route("/screener/turnover", methods=["GET"])
def view_screener_turnover() -> str:
    limit = request.args.get("limit", 200, type=int)
    repo = _get_repo()
    results = repo.get_trade_candidates("TurnoverTiming", limit=limit)
    return render_template("screener_turnover.html", results=results)


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