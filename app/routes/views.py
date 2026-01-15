import logging

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
def view_active_trades_raw() -> str:
    limit = request.args.get("limit", 500, type=int)
    results = _get_db().get_trades_history(limit=limit)  # type: ignore
    return render_template_string(
        HTML_TEMPLATES["active_trades_raw"], results=results, limit=limit
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
