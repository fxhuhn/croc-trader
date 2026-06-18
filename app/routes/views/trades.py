"""Routes and views for active and closed trades across strategies."""

from flask import render_template, request

from ...const import Strategies, ExitReason
from ...types import TradeStatus
from .blueprint import views_bp
from .dependencies import (
    _get_trade_view_service,
    cache,
)


@views_bp.route("/trades", methods=["GET"])
@views_bp.route("/trades/", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_overview() -> str:
    """Displays an overview of all active trades across strategies.

    Returns:
        str: Rendered HTML trades dashboard template.
    """
    service = _get_trade_view_service()

    # 1. Fetch Active Trades
    active_trades = service.get_trades(status=TradeStatus.ACTIVE)
    service.attach_sparklines(active_trades)

    # 2. Portfolio Metrics
    summary_metrics = service.get_portfolio_summary(active_trades)

    # 3. Strategy Allocation & Performance
    strategy_stats = {
        "Croc Setup": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Dip Buyer": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Turnover": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Two Percent": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "NDX Momentum": {"count": 0, "pnl": 0.0, "invested": 0.0},
    }

    croc_group = [
        Strategies.CrocSetup,
        Strategies.HoldTarget,
        Strategies.SplitTarget,
        "croc",
    ]
    turnover_group = [
        Strategies.TurnOverTiming,
        Strategies.TurnOverTiming_05,
        Strategies.TurnOverTiming_10,
    ]

    for trade in active_trades:
        # Resolve strategy via service to get the Enum value string
        strategy_key = service.resolve_strategy(trade)

        # Robust grouping
        if strategy_key in croc_group:
            label = "Croc Setup"
        elif strategy_key == Strategies.DipBuyer:
            label = "Dip Buyer"
        elif strategy_key in turnover_group:
            label = "Turnover"
        elif strategy_key == Strategies.TwoPercent:
            label = "Two Percent"
        elif strategy_key == Strategies.NDXMomentum:
            label = "NDX Momentum"
        else:
            label = str(trade.get("strategy", "Unknown"))

        if label not in strategy_stats:
            strategy_stats[label] = {"count": 0, "pnl": 0.0, "invested": 0.0}

        strategy_stats[label]["count"] += 1
        strategy_stats[label]["pnl"] += float(trade.get("unrealized_pnl", 0.0) or 0.0)

        entry_price = float(trade.get("entry_price") or 0.0)
        initial_size = float(trade.get("initial_size") or 0.0)
        strategy_stats[label]["invested"] += entry_price * initial_size

    # Prepare Data for Donut Chart
    allocation_labels = list(strategy_stats.keys())
    allocation_values = [float(data["invested"]) for data in strategy_stats.values()]
    # Custom colors: Blue, Purple, Orange, Slate, Emerald...
    palette = ["#2563eb", "#8b5cf6", "#f97316", "#64748b", "#10b981"]

    allocation_chart_html = service.generate_donut_chart(
        allocation_labels, allocation_values, palette
    )

    return render_template(
        "trades.html",
        active_trades=active_trades,
        summary=summary_metrics,
        strategy_stats=strategy_stats,
        donut_html=allocation_chart_html,
    )


@views_bp.route("/trades/croc", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_croc() -> str:
    """Displays the Croc Setup trade history and active positions.

    Returns:
        str: Rendered HTML template with Croc setup active/closed trades.
    """
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    croc_group = [
        Strategies.CrocSetup,
        Strategies.HoldTarget,
        Strategies.SplitTarget,
        "croc",
    ]

    active = service.get_trades(strategies=croc_group, status=TradeStatus.ACTIVE)
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed_all = service.get_trades(strategies=croc_group, status=TradeStatus.CLOSED)
    closed = [trade for trade in closed_all if trade.get("exit_reason") != "EXPIRED"]
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    index_stats = service.get_index_stats(closed)
    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)

    # Signal Aggregation (specific to Croc)
    signal_stats = {}
    for trade in closed:
        raw_signal = (
            trade["context"].get("original_signal")
            or trade["context"].get("match_rule", {}).get("Signal")
            or trade["strategy"]
        )
        signal_name = str(raw_signal).replace("Croc_", "") if raw_signal else "Unknown"

        if signal_name not in signal_stats:
            signal_stats[signal_name] = {
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            }

        realized_profit_and_loss = float(trade["realized_pnl"] or 0.0)
        service._update_statistics(signal_stats[signal_name], realized_profit_and_loss)

    # Add Avg PnL
    for value in signal_stats.values():
        value["average_pnl"] = (
            value["pnl"] / value["count"] if value["count"] > 0 else 0.0
        )

    sorted_signals = dict(
        sorted(
            signal_stats.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )
    )

    return render_template(
        "trades_croc.html",
        active_trades=active,
        active_groups=active_groups,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
        index_stats=index_stats,
        signal_stats=sorted_signals,
    )


@views_bp.route("/trades/dip-buyer", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_dip_buyer() -> str:
    """Displays the Dip Buyer trade history and active positions dashboard.

    Returns:
        str: Rendered HTML template with Dip Buyer trades.
    """
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=Strategies.DipBuyer, status=TradeStatus.ACTIVE
    )
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    # Inject Max Days for Time Stop Visualization
    for trade in active:
        trade["max_days"] = 7

    closed = service.get_trades(
        strategies=Strategies.DipBuyer, status=TradeStatus.CLOSED
    )
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    index_stats = service.get_index_stats(closed)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_dip_buyer.html",
        active_trades=active,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
        index_stats=index_stats,
    )


@views_bp.route("/trades/turnover", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_turnover() -> str:
    """Displays the Turnover Timing trade history and active positions.

    Returns:
        str: Rendered HTML template with Turnover trades.
    """
    limit = request.args.get("limit", 200, type=int)
    service = _get_trade_view_service()

    turnover_group = [
        Strategies.TurnOverTiming,
        Strategies.TurnOverTiming_05,
        Strategies.TurnOverTiming_10,
    ]

    active = service.get_trades(strategies=turnover_group, status=TradeStatus.ACTIVE)

    # Inject Max Days for Visualization (Mon-Fri = 5 days)
    for trade in active:
        trade["max_days"] = 5
        trade["green_candle_count"] = trade.get("context", {}).get(
            "green_candle_count", 0
        )

    # Fetch CLOSED Trades EXCLUDING Expired ones
    closed = service.get_trades(
        strategies=turnover_group,
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
    )

    # Sort Closed: Exit Date desc, then Symbol
    closed.sort(key=lambda x: (x["exit_date"] or "", x["symbol"]), reverse=True)
    closed = closed[:limit]

    # Active Groups
    active_groups = service.group_trades_by_symbol(active)

    # Stats
    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    # Aggregations
    index_stats = service.get_index_stats(closed)

    # Variant Stats
    variant_stats = {
        "Turnover 0.5": {
            "name": "Turnover",
            "version": "0.5",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
        },
        "Turnover 1.0": {
            "name": "Turnover",
            "version": "1.0",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
        },
    }

    for trade in closed:
        strategy_name = str(trade.get("strategy") or "")
        variant_key = None
        if "0.5" in strategy_name:
            variant_key = "Turnover 0.5"
        elif "1.0" in strategy_name:
            variant_key = "Turnover 1.0"

        if variant_key:
            realized_profit_and_loss = float(trade["realized_pnl"] or 0.0)
            service._update_statistics(
                variant_stats[variant_key], realized_profit_and_loss
            )

    # Calc Averages for Variants
    for item in variant_stats.values():
        item["average_pnl"] = item["pnl"] / item["count"] if item["count"] > 0 else 0.0

    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_turnover.html",
        summary=summary_metrics,
        active_trades=active_groups,
        history_groups=history_groups,
        closed_trades=closed,
        closed_summary=closed_summary,
        performance_index=list(index_stats.values()),
        performance_variants=list(variant_stats.values()),
    )


@views_bp.route("/trades/ndx-momentum", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_ndx_momentum() -> str:
    """Displays the NDX Momentum trade history and active positions.

    Returns:
        str: Rendered HTML template with NDX Momentum trades.
    """
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=Strategies.NDXMomentum, status=TradeStatus.ACTIVE
    )
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed = service.get_trades(
        strategies=Strategies.NDXMomentum, status=TradeStatus.CLOSED
    )
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_ndx_momentum.html",
        active_trades=active,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
    )


@views_bp.route("/trades/twopercent", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_twopercent() -> str:
    """Displays the Two Percent trade history and active positions.

    Returns:
        str: Rendered HTML template with Two Percent trades.
    """
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=Strategies.TwoPercent, status=TradeStatus.ACTIVE
    )
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed = service.get_trades(
        strategies=Strategies.TwoPercent, status=TradeStatus.CLOSED
    )
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_twopercent.html",
        active_trades=active,
        active_groups=active_groups,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
        index_stats={},
    )
