import logging
from collections.abc import Sequence
from typing import Any

from flask import render_template, request

from ...const import ExitReason, Strategies
from ...services.trade_manager.view_service import TradeViewData
from ...types import TradeStatus
from .blueprint import views_bp
from .dependencies import _get_trade_view_service, cache

STRATEGY_DISPLAY_MAP: dict[Strategies | str, str] = {
    Strategies.CrocSetup: "Croc Setup",
    Strategies.HoldTarget: "Croc Setup",
    Strategies.SplitTarget: "Croc Setup",
    Strategies.DipBuyer: "Dip Buyer",
    Strategies.TurnOverTiming: "Turnover",
    Strategies.TurnOverTiming_05: "Turnover",
    Strategies.TurnOverTiming_10: "Turnover",
    Strategies.TwoPercent: "Two Percent",
    Strategies.NDXMomentum: "NDX Momentum",
    Strategies.TGIM: "TGIM",
    Strategies.BridgeScout: "Bridge Scout",
    Strategies.BounceBandit: "Bounce Bandit",
}
logger = logging.getLogger(__name__)

CROC_STRATEGIES = [Strategies.HoldTarget, Strategies.SplitTarget]
TURNOVER_STRATEGIES = [
    Strategies.TurnOverTiming,
    Strategies.TurnOverTiming_05,
    Strategies.TurnOverTiming_10,
]


@views_bp.route("/trades", methods=["GET"])
@views_bp.route("/trades/", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_overview() -> str:
    """Displays an overview of all active trades across strategies.

    Returns:
        str: Rendered HTML trades dashboard template.
    """
    service = _get_trade_view_service()

    active_trades = service.get_trades(status=TradeStatus.ACTIVE)
    service.attach_sparklines(active_trades)
    summary_metrics = service.get_portfolio_summary(active_trades)
    strategy_stats = _build_strategy_overview_stats(active_trades, service)

    allocation_labels = list(strategy_stats.keys())
    allocation_values = [float(data["invested"]) for data in strategy_stats.values()]
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


def _build_strategy_overview_stats(
    active_trades: Sequence[TradeViewData | dict[str, Any]], service: Any
) -> dict[str, dict[str, float]]:
    """Builds aggregated trade counts, PnL, and invested volume per strategy group."""
    stats: dict[str, dict[str, float]] = {
        "Croc Setup": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Dip Buyer": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Turnover": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Two Percent": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "NDX Momentum": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "TGIM": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Bridge Scout": {"count": 0, "pnl": 0.0, "invested": 0.0},
        "Bounce Bandit": {"count": 0, "pnl": 0.0, "invested": 0.0},
    }

    for trade in active_trades:
        strat_key = service.resolve_strategy(trade)
        label = STRATEGY_DISPLAY_MAP.get(
            strat_key, str(trade.get("strategy", "Unknown"))
        )

        if label not in stats:
            stats[label] = {"count": 0, "pnl": 0.0, "invested": 0.0}

        stats[label]["count"] += 1
        stats[label]["pnl"] += float(trade.get("unrealized_pnl", 0.0) or 0.0)

        entry_price = float(trade.get("entry_price") or 0.0)
        initial_size = float(trade.get("initial_size") or 0.0)
        stats[label]["invested"] += entry_price * initial_size

    return stats


def _render_standard_strategy_trades(
    template_name: str,
    strategy: Strategies,
    include_index_stats: bool = True,
) -> str:
    """Renders standard strategy trade history and active positions view."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(strategies=strategy, status=TradeStatus.ACTIVE)
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed = service.get_trades(strategies=strategy, status=TradeStatus.CLOSED)
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active, closed_trades=closed)
    closed_summary = service.get_closed_summary(closed)
    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)

    context: dict[str, Any] = {
        "active_trades": active,
        "active_groups": active_groups,
        "closed_trades": closed,
        "history_groups": history_groups,
        "summary": summary_metrics,
        "closed_summary": closed_summary,
    }
    if include_index_stats:
        context["index_stats"] = {}

    return render_template(template_name, **context)


@views_bp.route("/trades/croc", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_croc() -> str:
    """Displays the Croc Setup trade history and active positions."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(strategies=CROC_STRATEGIES, status=TradeStatus.ACTIVE)
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed_all = service.get_trades(
        strategies=CROC_STRATEGIES, status=TradeStatus.CLOSED
    )
    closed = [trade for trade in closed_all if trade.get("exit_reason") != "EXPIRED"]
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active, closed_trades=closed)
    closed_summary = service.get_closed_summary(closed)
    index_stats = service.get_index_stats(closed)
    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)
    signal_stats = _aggregate_croc_signals(closed, service)

    return render_template(
        "trades_croc.html",
        active_trades=active,
        active_groups=active_groups,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
        index_stats=index_stats,
        signal_stats=signal_stats,
    )


def _aggregate_croc_signals(
    closed_trades: Sequence[TradeViewData | dict[str, Any]], service: Any
) -> dict[str, dict[str, Any]]:
    """Aggregates trade count, win/loss, and PnL by specific Croc entry signal."""
    signals: dict[str, dict[str, Any]] = {
        "Breakout (L20)": {"count": 0, "win": 0, "loss": 0, "pnl": 0.0},
        "Pullback (SMA20)": {"count": 0, "win": 0, "loss": 0, "pnl": 0.0},
        "Trend Continuation": {"count": 0, "win": 0, "loss": 0, "pnl": 0.0},
        "Early Entry (L10)": {"count": 0, "win": 0, "loss": 0, "pnl": 0.0},
        "Other": {"count": 0, "win": 0, "loss": 0, "pnl": 0.0},
    }
    for trade in closed_trades:
        ctx = trade.get("context")
        match_rule = ctx.get("match_rule") if isinstance(ctx, dict) else None
        raw_name = match_rule.get("name") if isinstance(match_rule, dict) else None
        signal_name = str(raw_name) if raw_name else "Other"
        if signal_name not in signals:
            signal_name = "Other"

        service._update_statistics(
            signals[signal_name], float(trade["realized_pnl"] or 0.0)
        )
    return signals


@views_bp.route("/trades/dip-buyer", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_dip_buyer() -> str:
    """Displays the Dip Buyer trade history and active positions."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=Strategies.DipBuyer, status=TradeStatus.ACTIVE
    )
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)

    closed = service.get_trades(
        strategies=Strategies.DipBuyer, status=TradeStatus.CLOSED
    )
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active, closed_trades=closed)
    closed_summary = service.get_closed_summary(closed)
    index_stats = service.get_index_stats(closed)
    weekday_stats = service.get_weekday_stats(closed)
    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_dip_buyer.html",
        active_trades=active,
        active_groups=active_groups,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary_metrics,
        closed_summary=closed_summary,
        index_stats=index_stats,
        weekday_stats=weekday_stats,
    )


@views_bp.route("/trades/turnover", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_turnover() -> str:
    """Displays the Turnover Timing trade history and active positions."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=TURNOVER_STRATEGIES, status=TradeStatus.ACTIVE
    )
    active.sort(key=lambda x: x["entry_date"] or "", reverse=True)
    for trade in active:
        cnt_val = trade.get("context", {}).get("green_candle_count")
        trade["green_candle_count"] = int(str(cnt_val)) if cnt_val is not None else None

    closed = service.get_trades(
        strategies=TURNOVER_STRATEGIES,
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED],
    )
    closed.sort(key=lambda x: x["exit_date"] or "", reverse=True)
    closed = closed[:limit]

    summary_metrics = service.get_portfolio_summary(active, closed_trades=closed)
    closed_summary = service.get_closed_summary(closed)
    index_stats = service.get_index_stats(closed)
    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)
    variant_stats = _aggregate_turnover_variants(closed, service)

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


def _aggregate_turnover_variants(
    closed_trades: Sequence[TradeViewData | dict[str, Any]], service: Any
) -> dict[str, dict[str, Any]]:
    """Aggregates performance statistics by Turnover timing variant."""
    variants: dict[str, dict[str, Any]] = {
        "Turnover 0.5": {
            "name": "Turnover 0.5",
            "version": "0.5",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
        },
        "Turnover 1.0": {
            "name": "Turnover 1.0",
            "version": "1.0",
            "count": 0,
            "win": 0,
            "loss": 0,
            "pnl": 0.0,
        },
    }
    for trade in closed_trades:
        strat_name = str(trade.get("strategy") or "")
        key = (
            "Turnover 0.5"
            if "0.5" in strat_name
            else ("Turnover 1.0" if "1.0" in strat_name else None)
        )
        if key:
            service._update_statistics(
                variants[key], float(trade["realized_pnl"] or 0.0)
            )

    for item in variants.values():
        pnl_val = float(item["pnl"])
        count_val = int(item["count"])
        item["average_pnl"] = pnl_val / count_val if count_val > 0 else 0.0

    return variants


@views_bp.route("/trades/ndx-momentum", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_ndx_momentum() -> str:
    """Displays the NDX Momentum trade history and active positions."""
    return _render_standard_strategy_trades(
        "trades_ndx_momentum.html",
        Strategies.NDXMomentum,
        include_index_stats=False,
    )


@views_bp.route("/trades/twopercent", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_twopercent() -> str:
    """Displays the Two Percent trade history and active positions."""
    return _render_standard_strategy_trades(
        "trades_twopercent.html",
        Strategies.TwoPercent,
        include_index_stats=True,
    )


@views_bp.route("/trades/tgim", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_tgim() -> str:
    """Displays the TGIM trade history and active positions."""
    return _render_standard_strategy_trades(
        "trades_tgim.html",
        Strategies.TGIM,
        include_index_stats=True,
    )


@views_bp.route("/trades/bridge-scout", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_bridge_scout() -> str:
    """Displays the Bridge Scout trade history and active positions."""
    return _render_standard_strategy_trades(
        "trades_bridge_scout.html",
        Strategies.BridgeScout,
        include_index_stats=True,
    )


@views_bp.route("/trades/bounce-bandit", methods=["GET"])
@views_bp.route("/trades/bounce_bandit", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_bounce_bandit() -> str:
    """Displays the Bounce Bandit trade history and active positions."""
    return _render_standard_strategy_trades(
        "trades_bounce_bandit.html",
        Strategies.BounceBandit,
        include_index_stats=True,
    )


def _prepare_active_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups and sorts active orders hierarchically."""
    if not orders:
        return []

    open_order_ids = _collect_open_order_ids(orders)
    groups = _group_orders_by_key(orders)

    sorted_group_keys = sorted(
        groups.keys(), key=lambda k: int(groups[k][0].get("order_id") or 0)
    )

    prepared_orders: list[dict[str, Any]] = []
    for group_key in sorted_group_keys:
        group_orders = groups[group_key]
        parent_is_open = _is_parent_open(
            group_orders[0].get("parent_id"), open_order_ids
        )

        for idx, order in enumerate(group_orders):
            order["is_child"] = False if idx == 0 else parent_is_open
            prepared_orders.append(order)

    return prepared_orders


def _group_orders_by_key(
    orders: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Groups orders by trade_group_id or order_id and sorts each group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for order in orders:
        group_key = str(order.get("trade_group_id") or order.get("order_id") or "")
        groups.setdefault(group_key, []).append(order)

    for group_orders in groups.values():
        group_orders.sort(key=lambda x: int(x.get("order_id") or 0))

    return groups


def _collect_open_order_ids(orders: list[dict[str, Any]]) -> set[int]:
    """Collects set of open integer order IDs."""
    open_ids: set[int] = set()
    for order in orders:
        val = order.get("order_id")
        if val is not None and str(val).isdigit():
            open_ids.add(int(val))
    return open_ids


def _is_parent_open(parent_id: Any, open_order_ids: set[int]) -> bool:
    """Determines whether parent order is open/active."""
    if (
        parent_id is None
        or parent_id == 0
        or str(parent_id).strip() in ("", "0", "None", "-")
    ):
        return True
    return str(parent_id).isdigit() and int(parent_id) in open_order_ids


@views_bp.route("/broker", methods=["GET"])
def view_broker_dashboard() -> str:
    """Displays the Trader Workstation (TWS) broker execution and reconciliation dashboard."""
    service = _get_trade_view_service()

    metrics_map = service.get_broker_summary()
    active_orders = service.get_broker_active_orders()
    all_active_orders = _prepare_active_orders(active_orders)
    error_orders = service.get_broker_error_orders()
    settlements = service.get_broker_settlements()
    discrepancies = service.get_reconciliation_discrepancies()
    active_trades = service.get_broker_active_trades()

    return render_template(
        "trades_broker.html",
        metrics=metrics_map,
        active_orders=all_active_orders,
        error_orders=error_orders,
        settlements=settlements,
        discrepancies=discrepancies,
        active_trades=active_trades,
    )
