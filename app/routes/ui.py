from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.backtest_viewer import RESULTS_DIR, BacktestViewer

# --- INITIALIZE VIEWER HERE ---
viewer = BacktestViewer()

bp = Blueprint("ui", __name__)


@bp.get("/")
def index():
    return jsonify({"status": "running", "app": "croc-trader", "version": "0.3.0"})


@bp.get("/croc-signal")
def croc_signal():
    f_symbol = request.args.get("symbol")
    f_tf = request.args.get("timeframe")
    f_sig = request.args.get("signal")
    f_day = request.args.get("day")

    repo = current_app.container.repo
    signals = repo.get_signals(
        symbol=f_symbol, timeframe=f_tf, signal=f_sig, day=f_day, limit=500
    )

    # attach statistics
    signals = repo.enrich_signals_with_stats(signals)

    stats = Counter(s["signal"] for s in signals)
    return render_template(
        "croc_signals.html",
        signals=signals,
        stats=stats,
        unique_symbols=repo.get_distinct("symbol"),
        unique_timeframes=repo.get_distinct("timeframe"),
        unique_signals=repo.get_distinct("signal"),
        current_filters={
            "symbol": f_symbol,
            "timeframe": f_tf,
            "signal": f_sig,
            "day": f_day,
        },
        symbol_markets=current_app.symbol_markets,
    )


@bp.get("/trades")
def trades():
    """Journal-like view of all tracked trades with filters and stats."""

    # 1. Get Filters
    f_symbol = request.args.get("symbol")
    f_signal = request.args.get("signal")
    f_state = request.args.get("state")

    # 2. Fetch All Trades (Repo should support filtering, but we can do it in memory for now)
    all_trades = current_app.container.repo.get_all_trades(limit=2000)

    filtered_trades = []

    # Dropdown lists
    unique_symbols = set()
    unique_signals = set()
    unique_states = set()

    # 3. Filter & Collect Stats
    for t in all_trades:
        # Collect unique values for dropdowns
        if t.get("symbol"):
            unique_symbols.add(t["symbol"])
        if t.get("signal"):
            unique_signals.add(t["signal"])
        if t.get("state"):
            unique_states.add(t["state"])

        # Apply Filters
        if f_symbol and t.get("symbol") != f_symbol:
            continue
        if f_signal and t.get("signal") != f_signal:
            continue
        if f_state and t.get("state") != f_state:
            continue

        filtered_trades.append(t)

    # 4. Calculate KPIs based on FILTERED data
    stats = {
        "total": len(filtered_trades),
        "pending": 0,
        "active": 0,
        "rejected": 0,
        "closed": 0,  # win + loss
        "total_pnl": 0.0,
    }

    closed_count_for_avg = 0

    for t in filtered_trades:
        state = (t.get("state") or "").lower()
        pnl = t.get("profit_loss") or 0.0

        # --- NEW P/L CALCULATION LOGIC ---
        state = (t.get("state") or "").lower()
        quantity = t.get("quantity") or 0
        entry_price = t.get("entry_price")
        exit_price = t.get("exit_price")
        current_price = t.get("current_price")  # Ensure your repo fetches this!

        # Fallback for missing quantity to show *some* PnL
        effective_qty = quantity if quantity and quantity > 0 else 1

        if pnl is None:
            if state in ["win", "loss", "closed"] and entry_price and exit_price:
                # Calculate PnL with effective quantity
                pnl = (exit_price - entry_price) * effective_qty
            elif state == "active" and entry_price and current_price:
                pnl = (current_price - entry_price) * effective_qty
            else:
                pnl = 0.0

        # Store calculated PnL back into the dict for the template
        t["profit_loss"] = pnl

        if state == "pending":
            stats["pending"] += 1
        elif state == "active":
            stats["active"] += 1
        elif state == "rejected":
            stats["rejected"] += 1
        elif state in ["win", "loss", "closed"]:
            stats["closed"] += 1
            stats["total_pnl"] += pnl
            closed_count_for_avg += 1

    stats["avg_pnl"] = (
        (stats["total_pnl"] / closed_count_for_avg) if closed_count_for_avg > 0 else 0.0
    )

    return render_template(
        "trades.html",
        trades=filtered_trades,
        stats=stats,
        unique_symbols=sorted(list(unique_symbols)),
        unique_signals=sorted(list(unique_signals)),
        unique_states=sorted(list(unique_states)),
        current_filters={"symbol": f_symbol, "signal": f_signal, "state": f_state},
    )


# --- NEW ROUTES ---


@bp.route("/strategies")
def strategies_list():
    # 1. Get basic list (ID, name, timestamp)
    basic_list = viewer.list_strategies()

    # 2. Enrich with metrics
    strategies = []
    for s in basic_list:
        try:
            # Re-use get_details logic to load the JSON/YAML
            details = viewer.get_details(s["id"])
            if details:
                # Handle nested performance dict safely
                raw_metrics = details.get("metrics", {})
                if not isinstance(raw_metrics, dict):
                    raw_metrics = {}

                perf = raw_metrics.get("performance", raw_metrics)

                # Attach perf to the strategy object
                s["perf"] = perf
            else:
                s["perf"] = {}
        except Exception:
            s["perf"] = {}

        strategies.append(s)

    return render_template("strategies.html", strategies=strategies)


# app/routes/ui.py


@bp.route("/strategies/<strategy_id>")
def strategy_detail(strategy_id):
    # 1. Load Data
    data = viewer.get_details(strategy_id)
    if not data:
        abort(404)

    raw_metrics = data.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    perf_data = raw_metrics.get("performance", raw_metrics)
    full_name = raw_metrics.get("strategy_name")

    # 2. Process Monthly Returns (Pass Data, Not HTML)
    monthly_data = []
    monthly_cols = []

    monthly_path = RESULTS_DIR / f"{strategy_id}_monthly_returns.csv"
    if monthly_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(monthly_path)
            # Fill NaNs with empty string or 0 for display
            df = df.fillna(0)
            # Convert to list of dictionaries for Jinja
            monthly_data = df.to_dict(orient="records")
            monthly_cols = df.columns.tolist()
        except Exception as e:
            current_app.logger.error(f"Error loading monthly returns: {e}")

    # 3. Get All Trades
    all_trades = viewer.get_trades(full_name)

    # 4. Filter Logic
    f_symbol = request.args.get("symbol")
    f_year = request.args.get("year")
    f_month = request.args.get("month")

    filtered_trades = []
    available_symbols = set()
    available_years = set()

    for t in all_trades:
        entry = t.get("entry_date", "")
        t_year = entry[:4] if entry else ""
        t_month = entry[5:7] if len(entry) >= 7 else ""

        if t.get("symbol"):
            available_symbols.add(t["symbol"])
        if t_year:
            available_years.add(t_year)

        if f_symbol and t.get("symbol") != f_symbol:
            continue
        if f_year and t_year != f_year:
            continue
        if f_month and t_month != f_month:
            continue

        filtered_trades.append(t)

    return render_template(
        "strategy_detail.html",
        strategy=data,
        perf=perf_data,
        trades=filtered_trades,
        strategy_id=strategy_id,
        # Monthly Data
        monthly_data=monthly_data,
        monthly_cols=monthly_cols,
        # Filters
        symbols=sorted(list(available_symbols)),
        years=sorted(list(available_years), reverse=True),
        current_filters={"symbol": f_symbol, "year": f_year, "month": f_month},
    )


@bp.route("/strategies/image/<path:filename>")
def strategy_image(filename):
    # Ensure RESULTS_DIR is absolute for send_from_directory
    return send_from_directory(RESULTS_DIR.absolute(), filename)
