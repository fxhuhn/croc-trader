"""Routes and views for strategy setups screeners."""

from typing import TypedDict

from flask import render_template, request

from ...const import Strategies
from .blueprint import views_bp
from .dependencies import (
    ScreenerViewService,
    SignalRepository,
    _get_screener_view_service,
    _get_signal_repository,
)


class StrategyOverview(TypedDict):
    """Represents a strategy summary for the screener dashboard."""

    id: str
    name: str
    # 'desc' is preserved strictly for frontend template compatibility
    desc: str
    icon: str
    count: int
    is_active: bool


def _get_strategy_overview(
    signals_repository: SignalRepository,
    screener_service: ScreenerViewService,
) -> list[StrategyOverview]:
    """Fetches the overview statistics for all trading strategies.

    Args:
        signals_repository: Active repository instance for database access.
        screener_service: Service layer instance for screening.

    Returns:
        list[StrategyOverview]: Populated strategies metadata and signals count.
    """
    count_croc = len(screener_service.get_candidates(Strategies.CrocSetup, limit=100))
    count_dip = len(
        signals_repository.get_trade_candidates(Strategies.DipBuyer, limit=100)
    )
    count_turnover = len(
        signals_repository.get_trade_candidates(Strategies.TurnOverTiming, limit=100)
    )
    count_twopercent = len(
        signals_repository.get_trade_candidates(Strategies.TwoPercent, limit=100)
    )
    count_ndx_momentum = len(
        signals_repository.get_trade_candidates(Strategies.NDXMomentum, limit=100)
    )
    count_tgim = len(
        signals_repository.get_trade_candidates(Strategies.TGIM, limit=100)
    )
    count_bridge_scout = len(
        signals_repository.get_trade_candidates(Strategies.BridgeScout, limit=100)
    )

    return [
        {
            "id": "croc",
            "name": "Croc Setup",
            "desc": (
                "Trendfolge-Signale basierend auf Wochen- und Tageschart-Momentum."
            ),
            "icon": "arrow-up",
            "count": count_croc,
            "is_active": count_croc > 0,
        },
        {
            "id": "dip-buyer",
            "name": "Dip Buyer",
            "desc": "Kurzfristige Mean-Reversion Setups für Indizes.",
            "icon": "trending-up",
            "count": count_dip,
            "is_active": count_dip > 0,
        },
        {
            "id": "turnover",
            "name": "Turnover Timing",
            "desc": "Saisonale Effekte zum Monatsende und Rebalancing.",
            "icon": "refresh-cw",
            "count": count_turnover,
            "is_active": count_turnover > 0,
        },
        {
            "id": "twopercent",
            "name": "Two Percent",
            "desc": "Intraday Mean-Reversion bei extremen Bewegungen.",
            "icon": "percent",
            "count": count_twopercent,
            "is_active": count_twopercent > 0,
        },
        {
            "id": "ndx-momentum",
            "name": "NDX Momentum",
            "desc": "Top 5 Momentum Leaderboard des Nasdaq 100.",
            "icon": "zap",
            "count": count_ndx_momentum,
            "is_active": count_ndx_momentum > 0,
        },
        {
            "id": "tgim",
            "name": "TGIM",
            "desc": "Thank God It's Monday Mean-Reversion Setup für SPY.",
            "icon": "calendar",
            "count": count_tgim,
            "is_active": count_tgim > 0,
        },
        {
            "id": "bridge-scout",
            "name": "Bridge Scout",
            "desc": "End-of-Month Mean-Reversion Setup für QQQ.",
            "icon": "compass",
            "count": count_bridge_scout,
            "is_active": count_bridge_scout > 0,
        },
    ]


@views_bp.route("/screener", methods=["GET"])
def view_screener_overview() -> str:
    """Displays the overview dashboard page for all available screeners.

    Returns:
        str: Rendered HTML dashboard template.
    """
    signals_repository = _get_signal_repository()
    screener_service = _get_screener_view_service()

    strategies = _get_strategy_overview(signals_repository, screener_service)

    return render_template(
        "screener.html",
        strategies=strategies,
    )


@views_bp.route("/screener/croc", methods=["GET"])
def view_screener_croc() -> str:
    """Displays the Croc Setup screener details and current candidates.

    Returns:
        str: Rendered HTML template with Croc setup candidates list.
    """
    limit = request.args.get("limit", 200, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates("Croc_", limit=limit)
    return render_template("screener_croc.html", results=results)


@views_bp.route("/screener/dip-buyer", methods=["GET"])
def view_screener_dip_buyer() -> str:
    """Displays the Dip Buyer screener details and index candidates.

    Returns:
        str: Rendered HTML template with Dip Buyer candidates list.
    """
    limit = request.args.get("limit", 100, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.DipBuyer, limit=limit)
    return render_template("screener_dip_buyer.html", results=results)


@views_bp.route("/screener/turnover", methods=["GET"])
def view_screener_turnover() -> str:
    """Displays the Turnover Timing screener results aggregated by symbol.

    Returns:
        str: Rendered HTML template with Turnover candidates list.
    """
    limit = request.args.get("limit", 200, type=int)
    service = _get_screener_view_service()
    results = service.get_turnover_candidates(limit=limit)
    return render_template("screener_turnover.html", results=results)


@views_bp.route("/screener/twopercent", methods=["GET"])
def view_screener_twopercent() -> str:
    """Displays the Two Percent Screener with current candidates.

    Returns:
        str: Rendered HTML template with Two Percent candidates list.
    """
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.TwoPercent, limit=limit)
    return render_template("screener_twopercent.html", results=results)


@views_bp.route("/screener/ndx-momentum", methods=["GET"])
def view_screener_ndx_momentum() -> str:
    """Displays the Nasdaq 100 Momentum screener leaderboard results.

    Returns:
        str: Rendered HTML template with Momentum candidates list.
    """
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.NDXMomentum, limit=limit)
    return render_template("screener_ndx_momentum.html", results=results)


@views_bp.route("/screener/tgim", methods=["GET"])
def view_screener_tgim() -> str:
    """Displays the TGIM screener with current candidates.

    Returns:
        str: Rendered HTML template with TGIM candidates list.
    """
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.TGIM, limit=limit)
    return render_template("screener_tgim.html", results=results)


@views_bp.route("/screener/bridge-scout", methods=["GET"])
def view_screener_bridge_scout() -> str:
    """Displays the Bridge Scout screener with current candidates.

    Returns:
        str: Rendered HTML template with Bridge Scout candidates list.
    """
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.BridgeScout, limit=limit)
    return render_template("screener_bridge_scout.html", results=results)
