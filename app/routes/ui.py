from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

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
    """Journal-like view of all tracked trades."""
    trades = current_app.container.repo.get_all_trades(limit=1000)
    return render_template("trades.html", trades=trades)


# --- NEW ROUTES ---


@bp.route("/strategies")
def strategies_list():
    # Now 'viewer' is defined
    strategies = viewer.list_strategies()
    return render_template("strategies.html", strategies=strategies)


@bp.route("/strategies/<strategy_id>")
def strategy_detail(strategy_id):
    data = viewer.get_details(strategy_id)
    if not data:
        abort(404)

    full_name = data["metrics"].get("strategy_name")
    trades = viewer.get_trades(full_name)

    return render_template(
        "strategy_detail.html", strategy=data, trades=trades, strategy_id=strategy_id
    )


@bp.route("/strategies/image/<path:filename>")
def strategy_image(filename):
    # Ensure RESULTS_DIR is absolute for send_from_directory
    return send_from_directory(RESULTS_DIR.absolute(), filename)
