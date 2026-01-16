import logging

import pandas as pd
from flask import Blueprint, current_app, render_template_string, request

from app.services.database import SignalDatabase

from .security import require_ip_whitelist
from .templates_raw import HTML_TEMPLATES

logger = logging.getLogger(__name__)
views_bp = Blueprint("views", __name__)


def _get_db() -> SignalDatabase:
    conf = current_app.config["APP_CONFIG"]
    return SignalDatabase(conf.get_db_path("signals"))


@views_bp.route("/screener/webhook", methods=["GET"])
def view_screener_webhook() -> str:
    limit = request.args.get("limit", 100, type=int)
    results = _get_db().get_webhook_results(limit=limit)  # type: ignore
    return render_template_string(HTML_TEMPLATES["webhook"], results=results)


@views_bp.route("/screener/webhook_2", methods=["GET"])
@require_ip_whitelist
def view_screener_croc() -> str:
    limit = request.args.get("limit", 200, type=int)
    results = _get_db().get_croc_results(limit=limit)
    return render_template_string(HTML_TEMPLATES["croc_setup"], results=results)


@views_bp.route("/screener/dip-buyer", methods=["GET"])
def view_screener_dip_buyer() -> str:
    limit = request.args.get("limit", 100, type=int)
    results = _get_db().get_dip_buyer_results(limit=limit)  # type: ignore
    return render_template_string(HTML_TEMPLATES["dip_buyer"], results=results)


@views_bp.route("/screener/turnover-timing", methods=["GET"])
@require_ip_whitelist
def view_screener_turnover() -> str:
    limit = request.args.get("limit", 100, type=int)
    results = _get_db().get_turnover_timing_results(limit=limit)  # type: ignore
    return render_template_string(HTML_TEMPLATES["turnover"], results=results)


@views_bp.route("/strategy/trades", methods=["GET"])
def view_strategy_trades() -> str:
    limit = request.args.get("limit", 100, type=int)
    results = _get_db().get_trades_history(limit=limit)  # type: ignore
    return render_template_string(HTML_TEMPLATES["strategy_trades"], results=results)


@views_bp.route("/active-trades", methods=["GET"])
def view_active_trades_dashboard() -> str:
    """
    Zeigt das Portfolio-Dashboard an.
    Reihenfolge: Stats -> Active -> Created -> History
    """
    limit = request.args.get("limit", 500, type=int)
    all_trades = _get_db().get_trades_history(limit=limit)  # type: ignore

    # 1. Sortierung in 3 Eimer: Active, Created, History
    active_trades = []  # Status ACTIVE
    new_trades = []  # Status CREATED
    history_log = []  # CLOSED, MISSED, etc.

    for trade in all_trades:
        status = str(trade.get("status", "")).upper()
        if status == "ACTIVE":
            active_trades.append(trade)
        elif status == "CREATED":
            new_trades.append(trade)
        else:
            history_log.append(trade)

    # 2. Statistik Berechnung (Basierend auf History)
    strategy_stats = []
    if history_log:
        try:
            valid_trades = [
                t
                for t in history_log
                if t.get("exit_price") is not None and t.get("entry_price")
            ]

            if valid_trades:
                df = pd.DataFrame(valid_trades)
                df["entry_price"] = pd.to_numeric(df["entry_price"])
                df["exit_price"] = pd.to_numeric(df["exit_price"])
                df["pct"] = (
                    (df["exit_price"] - df["entry_price"]) / df["entry_price"] * 100
                )

                if "strategy" in df.columns:
                    for strategy_name, group in df.groupby("strategy"):
                        count = len(group)
                        wins = len(group[group["pct"] > 0])
                        win_rate = (wins / count * 100) if count > 0 else 0
                        avg_ret = group["pct"].mean()

                        strategy_stats.append(
                            {
                                "name": strategy_name,
                                "count": count,
                                "win_rate": round(win_rate, 1),
                                "avg_return": round(avg_ret, 2),
                            }
                        )

                    strategy_stats.sort(key=lambda x: x["count"], reverse=True)

        except Exception as e:
            logger.error(f"Statistik Fehler: {e}")

    # Übergabe der getrennten Listen an das Template
    return render_template_string(
        HTML_TEMPLATES["active_trades_dashboard"],
        active_trades=active_trades,
        new_trades=new_trades,
        history=history_log,
        stats=strategy_stats,
        limit=limit,
    )


@views_bp.route("/backtest/dip-buyer", methods=["GET", "POST"])
@require_ip_whitelist
def backtest_dip_buyer() -> str:
    if request.method == "POST":
        backtester = current_app.extensions.get("backtester")
        debug_sym = request.form.get("debug_symbol", "").strip().upper() or None

        if debug_sym:
            logger.info(f"Starting debug backtest for: {debug_sym}")

        results = backtester.run_backtest(start_year=2023, debug_symbol=debug_sym)

        if not results or "metrics" not in results:
            return render_template_string(
                "<h1>Keine Daten oder Fehler im Backtest</h1>"
            )

        return render_template_string(HTML_TEMPLATES["backtest_report"], data=results)

    return render_template_string(HTML_TEMPLATES["backtest_form"])
