from __future__ import annotations

from collections import Counter

from flask import Blueprint, current_app, jsonify, render_template, request

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

    signals = current_app.container.repo.get_signals(
        symbol=f_symbol, timeframe=f_tf, signal=f_sig, day=f_day, limit=500
    )

    stats = Counter(s["signal"] for s in signals)
    return render_template(
        "croc_signals.html",
        signals=signals,
        stats=stats,
        unique_symbols=current_app.container.repo.get_distinct("symbol"),
        unique_timeframes=current_app.container.repo.get_distinct("timeframe"),
        unique_signals=current_app.container.repo.get_distinct("signal"),
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
