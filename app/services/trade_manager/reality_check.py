"""Pure domain logic and data structures for the Backtest vs. Broker Reality Check.

Compares simulated backtest results (from signals.db) with actual broker
executions and settlements (from trading.db), calculating slippage in currency,
commission impact, and cumulative dual-equity performance.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

# Canonical future symbol mappings
FUTURES_STRATEGY_MAP: dict[str, str] = {
    "TGIM": "MES",
    "tgim": "MES",
    "BounceBandit": "MNQ",
    "bounce_bandit": "MNQ",
    "BridgeScout": "MNQ",
    "bridge_scout": "MNQ",
}

MIN_PARTS_FOR_STRATEGY_EXTRACTION: int = 2


@dataclass(frozen=True)
class PositionRealityCheck:
    """Represents a matched active/open position comparison between Backtest and Broker."""

    trade_id: int
    symbol: str
    strategy: str
    entry_date: str
    days_held: int
    quantity_bt: float
    quantity_broker: float
    entry_price_bt: Decimal
    entry_price_broker: Decimal
    entry_slippage: Decimal
    slippage_cost: Decimal
    open_pnl_bt: Decimal
    open_pnl_broker: Decimal
    order_status: str = "Filled"
    trade_group_id: str = ""
    strategy_filter: str = ""
    strategy_display: str = ""
    strategy_badge_class: str = "bg-slate-100 text-slate-700 border-slate-200"
    has_quantity_diff: bool = False
    has_entry_price_diff: bool = False
    has_pnl_diff: bool = False


@dataclass(frozen=True)
class HistoryRealityCheck:
    """Represents a matched closed trade comparison between Backtest and Broker."""

    trade_id: int
    symbol: str
    strategy: str
    date: str
    quantity_bt: float
    quantity_broker: float
    unit_label_bt: str
    unit_label_broker: str
    entry_price_bt: Decimal
    entry_price_broker: Decimal
    exit_price_bt: Decimal
    exit_price_broker: Decimal
    entry_slippage: Decimal
    exit_slippage: Decimal
    total_slippage: Decimal
    net_pnl_bt: Decimal
    net_pnl_broker: Decimal
    commissions: Decimal
    endpreis_delta: Decimal
    trade_group_id: str = ""
    strategy_filter: str = ""
    strategy_display: str = ""
    strategy_badge_class: str = "bg-slate-100 text-slate-700 border-slate-200"
    has_quantity_diff: bool = False
    has_entry_price_diff: bool = False
    has_exit_price_diff: bool = False
    has_pnl_diff: bool = False
    executions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RealityCheckSummary:
    """Aggregated financial KPIs for the Reality Check dashboard."""

    matched_count: int
    open_matched_count: int
    closed_matched_count: int
    net_pnl_bt: Decimal
    net_pnl_broker: Decimal
    total_commissions: Decimal
    avg_entry_slippage: Decimal
    avg_exit_slippage: Decimal
    pnl_delta: Decimal


@dataclass(frozen=True)
class EquityPoint:
    """Single data point along the cumulative equity timeline."""

    date: str
    trade_id: int
    symbol: str
    cumulative_pnl_bt: float
    cumulative_pnl_broker: float
    pnl_delta: float


def to_decimal(value: object, fallback: str = "0.00") -> Decimal:
    """Safely converts an arbitrary numeric object to Decimal with 2 decimal places.

    Args:
        value: Input numeric value, string or float.
        fallback: String representation to return on conversion failure.

    Returns:
        Decimal: Safe financial decimal value rounded to 2 decimal places.
    """
    if value is None:
        return Decimal(fallback)
    try:
        if isinstance(value, float):
            return Decimal(str(round(value, 2)))
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(fallback)


def parse_trade_id_from_trade_group_id(trade_group_id: str | None) -> int | None:
    """Extracts the integer signals trade_id from a broker trade_group_id.

    The trade_group_id follows the canonical format '{trade_id}_{strategy}_{symbol}'.

    Args:
        trade_group_id: Raw string identifier (e.g. '1110_DipBuyer_SNDK').

    Returns:
        int | None: Parsed trade ID, or None if prefix is not an integer.
    """
    if not trade_group_id:
        return None
    match = re.match(r"^(\d+)_", str(trade_group_id).strip())
    if match:
        return int(match.group(1))
    return None


def is_future_strategy(strategy_name: str) -> bool:
    """Determines whether a strategy uses futures execution rather than equity shares.

    Args:
        strategy_name: Canonical or display name of the strategy.

    Returns:
        bool: True if strategy routes to futures (MES or MNQ).
    """
    clean_name = strategy_name.lower().replace(" ", "").replace("_", "")
    for fut_key in FUTURES_STRATEGY_MAP:
        if fut_key.lower() in clean_name:
            return True
    return False


STRATEGY_DISPLAY_MAP: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("dip",), "Dip Buyer", "bg-indigo-50 text-indigo-700 border-indigo-200/60"),
    (
        ("turnover", "0.5"),
        "Turnover 0.5",
        "bg-amber-50 text-amber-700 border-amber-200/60",
    ),
    (
        ("turnover", "05"),
        "Turnover 0.5",
        "bg-amber-50 text-amber-700 border-amber-200/60",
    ),
    (
        ("turnover", "1.0"),
        "Turnover 1.0",
        "bg-amber-50 text-amber-700 border-amber-200/60",
    ),
    (
        ("turnover", "10"),
        "Turnover 1.0",
        "bg-amber-50 text-amber-700 border-amber-200/60",
    ),
    (("turnover",), "Turnover", "bg-amber-50 text-amber-700 border-amber-200/60"),
    (
        ("two", "2"),
        "Two Percent",
        "bg-purple-50 text-purple-700 border-purple-200/60",
    ),
    (
        ("ndx", "momentum"),
        "NDX Momentum",
        "bg-rose-50 text-rose-700 border-rose-200/60",
    ),
    (("tgim", "monday"), "TGIM", "bg-sky-50 text-sky-700 border-sky-200/60"),
    (
        ("bridge", "scout", "qqq_eom"),
        "Bridge Scout",
        "bg-teal-50 text-teal-700 border-teal-200/60",
    ),
    (
        ("bounce", "bandit", "qqq_meanrev"),
        "Bounce Bandit",
        "bg-violet-50 text-violet-700 border-violet-200/60",
    ),
)


def resolve_strategy_meta(strategy_name: str) -> tuple[str, str]:
    """Resolves a canonical human display name and Tailwind color badge class.

    Args:
        strategy_name: Raw strategy string from database or trade record.

    Returns:
        tuple[str, str]: (display_name, badge_css_class)
    """
    clean = strategy_name.lower().strip()
    if not clean:
        return "—", "bg-slate-100 text-slate-700 border-slate-200"

    for triggers, display, badge in STRATEGY_DISPLAY_MAP:
        # All triggers must match for compound checks (like turnover + 0.5),
        # or any trigger if single-element
        if len(triggers) > 1 and "turnover" in triggers:
            if all(t in clean for t in triggers):
                return display, badge
        elif any(t in clean for t in triggers):
            return display, badge

    display = strategy_name.replace("_", " ").title()
    return display, "bg-slate-100 text-slate-700 border-slate-200"


def matches_strategy_filter(
    strategy: str,
    strategy_filter_val: str,
    target_filter: str | None,
) -> bool:
    """Evaluates whether a strategy matches the user's active filter."""
    if not target_filter or target_filter == "all":
        return True
    return target_filter in (strategy, strategy_filter_val)


def _build_position_comparison(
    bt_trade: dict[str, Any],
    pos: dict[str, Any],
    trade_id: int,
    trade_group_id: str,
) -> PositionRealityCheck:
    """Constructs an immutable PositionRealityCheck comparison record."""
    strat = str(
        pos.get("strategy")
        or pos.get("strategy_name")
        or bt_trade.get("strategy")
        or bt_trade.get("strategy_name")
        or ""
    )
    if not strat and trade_group_id:
        parts = trade_group_id.split("_")
        if len(parts) >= MIN_PARTS_FOR_STRATEGY_EXTRACTION:
            strat = parts[1]

    strat_filter_val = str(pos.get("strategy_filter") or strat)
    strat_display, strat_badge = resolve_strategy_meta(strat)
    is_fut = is_future_strategy(strat)

    qty_bt = float(bt_trade.get("current_size") or bt_trade.get("initial_size") or 0.0)
    qty_broker = float(pos.get("current_size") or 0.0)

    entry_bt = to_decimal(bt_trade.get("entry_price"))
    entry_broker = to_decimal(pos.get("entry_price"))

    entry_slip = entry_broker - entry_bt
    slip_cost = entry_slip * to_decimal(qty_broker)

    open_pnl_bt = to_decimal(bt_trade.get("unrealized_pnl"))
    if open_pnl_bt == Decimal("0.00") and qty_bt > 0:
        current_bt_price = to_decimal(bt_trade.get("current_price"))
        if current_bt_price > Decimal("0.00"):
            open_pnl_bt = (current_bt_price - entry_bt) * to_decimal(qty_bt)

    open_pnl_broker = to_decimal(pos.get("unrealized_pnl"))

    has_qty_diff = not is_fut and (int(qty_bt) != int(qty_broker))
    has_entry_diff = entry_bt != entry_broker
    has_pnl_diff = open_pnl_bt != open_pnl_broker

    return PositionRealityCheck(
        trade_id=trade_id,
        symbol=str(pos.get("symbol") or bt_trade.get("symbol") or ""),
        strategy=strat,
        entry_date=str(pos.get("entry_date") or bt_trade.get("entry_date") or "-"),
        days_held=int(pos.get("days_held") or bt_trade.get("days_held") or 0),
        quantity_bt=qty_bt,
        quantity_broker=qty_broker,
        entry_price_bt=entry_bt,
        entry_price_broker=entry_broker,
        entry_slippage=entry_slip,
        slippage_cost=slip_cost,
        open_pnl_bt=open_pnl_bt,
        open_pnl_broker=open_pnl_broker,
        order_status=str(pos.get("tws_status") or "Filled"),
        trade_group_id=trade_group_id,
        strategy_filter=strat_filter_val,
        strategy_display=strat_display,
        strategy_badge_class=strat_badge,
        has_quantity_diff=has_qty_diff,
        has_entry_price_diff=has_entry_diff,
        has_pnl_diff=has_pnl_diff,
    )


def match_active_positions(
    signals_trades: Sequence[dict[str, Any]],
    broker_positions: Sequence[dict[str, Any]],
    strategy_filter: str | None = None,
) -> list[PositionRealityCheck]:
    """Matches active broker positions with active backtest trades.

    Only trades present in both systems are included. Unmatched records are ignored.

    Args:
        signals_trades: Active trades from signals.db.
        broker_positions: Active positions from trading.db.
        strategy_filter: Optional filter by strategy name.

    Returns:
        list[PositionRealityCheck]: Sorted list of matched active comparisons.
    """
    signals_map: dict[int, dict[str, Any]] = {
        trade["id"]: trade
        for trade in signals_trades
        if isinstance(trade.get("id"), int)
    }

    matched: list[PositionRealityCheck] = []

    for pos in broker_positions:
        trade_group_id = str(pos.get("trade_group_id") or "")
        trade_id = parse_trade_id_from_trade_group_id(trade_group_id)
        if trade_id is None:
            fallback_id = pos.get("id")
            trade_id = (
                fallback_id
                if isinstance(fallback_id, int) and fallback_id > 0
                else None
            )

        if trade_id is None or trade_id not in signals_map:
            continue

        bt_trade = signals_map[trade_id]
        strat = str(pos.get("strategy") or bt_trade.get("strategy") or "")
        strat_filter_val = str(pos.get("strategy_filter") or strat)

        if not matches_strategy_filter(strat, strat_filter_val, strategy_filter):
            continue

        matched.append(
            _build_position_comparison(bt_trade, pos, trade_id, trade_group_id)
        )

    matched.sort(key=lambda p: p.entry_date, reverse=True)
    return matched


def _build_history_comparison(
    bt_trade: dict[str, Any],
    settlement: dict[str, Any],
    trade_id: int,
    trade_group_id: str,
) -> HistoryRealityCheck:
    """Constructs an immutable HistoryRealityCheck comparison record."""
    strat = str(
        settlement.get("strategy_name")
        or settlement.get("strategy")
        or bt_trade.get("strategy")
        or bt_trade.get("strategy_name")
        or ""
    )
    if not strat and trade_group_id:
        parts = trade_group_id.split("_")
        if len(parts) >= MIN_PARTS_FOR_STRATEGY_EXTRACTION:
            strat = parts[1]

    strat_filter_val = str(settlement.get("strategy_filter") or strat)
    is_fut = is_future_strategy(strat)

    qty_bt = float(bt_trade.get("initial_size") or bt_trade.get("current_size") or 0.0)
    qty_broker = float(settlement.get("quantity") or 0.0)
    if qty_broker == 0.0:
        qty_broker = 1.0 if is_fut else qty_bt

    entry_bt = to_decimal(bt_trade.get("entry_price"))
    entry_broker = to_decimal(settlement.get("avg_entry_price"))
    exit_bt = to_decimal(bt_trade.get("exit_price"))
    exit_broker = to_decimal(settlement.get("avg_exit_price"))

    if not is_fut:
        entry_slip = entry_broker - entry_bt
        exit_slip = exit_broker - exit_bt
        total_slip = (entry_slip * to_decimal(qty_broker)) + (
            exit_slip * to_decimal(qty_broker)
        )
    else:
        entry_slip = to_decimal(settlement.get("price_diff_slippage"))
        exit_slip = Decimal("0.00")
        total_slip = entry_slip

    net_pnl_bt = to_decimal(bt_trade.get("realized_pnl"))
    net_pnl_broker = to_decimal(settlement.get("net_pnl"))
    commissions = to_decimal(settlement.get("total_commissions"))
    endpreis_delta = net_pnl_broker - net_pnl_bt

    date_val = str(settlement.get("settled_at") or bt_trade.get("exit_date") or "-")[
        :10
    ]

    executions_list = cast_executions(settlement.get("executions"))

    strat_display, strat_badge = resolve_strategy_meta(strat)
    has_qty_diff = not is_fut and (int(qty_bt) != int(qty_broker))
    has_entry_diff = entry_bt != entry_broker
    has_exit_diff = exit_bt != exit_broker
    has_pnl_diff = net_pnl_bt != net_pnl_broker

    return HistoryRealityCheck(
        trade_id=trade_id,
        symbol=str(settlement.get("symbol") or bt_trade.get("symbol") or ""),
        strategy=strat,
        date=date_val,
        quantity_bt=qty_bt,
        quantity_broker=qty_broker,
        unit_label_bt="Stk",
        unit_label_broker="Ktr." if is_fut else "Stk",
        entry_price_bt=entry_bt,
        entry_price_broker=entry_broker,
        exit_price_bt=exit_bt,
        exit_price_broker=exit_broker,
        entry_slippage=entry_slip,
        exit_slippage=exit_slip,
        total_slippage=total_slip,
        net_pnl_bt=net_pnl_bt,
        net_pnl_broker=net_pnl_broker,
        commissions=commissions,
        endpreis_delta=endpreis_delta,
        trade_group_id=trade_group_id,
        strategy_filter=strat_filter_val,
        strategy_display=strat_display,
        strategy_badge_class=strat_badge,
        has_quantity_diff=has_qty_diff,
        has_entry_price_diff=has_entry_diff,
        has_exit_price_diff=has_exit_diff,
        has_pnl_diff=has_pnl_diff,
        executions=executions_list,
    )


def match_closed_history(
    signals_closed_trades: Sequence[dict[str, Any]],
    broker_settlements: Sequence[dict[str, Any]],
    strategy_filter: str | None = None,
) -> list[HistoryRealityCheck]:
    """Matches broker settlements with closed backtest trades.

    Only trades present in both systems are included. Unmatched records are ignored.

    Args:
        signals_closed_trades: Closed trades from signals.db.
        broker_settlements: Settlements from trading.db.
        strategy_filter: Optional filter by strategy name.

    Returns:
        list[HistoryRealityCheck]: Chronologically reverse sorted matched comparisons.
    """
    signals_map: dict[int, dict[str, Any]] = {
        trade["id"]: trade
        for trade in signals_closed_trades
        if isinstance(trade.get("id"), int)
    }

    matched: list[HistoryRealityCheck] = []

    for settlement in broker_settlements:
        trade_group_id = str(settlement.get("trade_group_id") or "")
        trade_id = parse_trade_id_from_trade_group_id(trade_group_id)
        if trade_id is None or trade_id not in signals_map:
            continue

        bt_trade = signals_map[trade_id]
        strat = str(settlement.get("strategy_name") or bt_trade.get("strategy") or "")
        strat_filter_val = str(settlement.get("strategy_filter") or strat)

        if not matches_strategy_filter(strat, strat_filter_val, strategy_filter):
            continue

        matched.append(
            _build_history_comparison(bt_trade, settlement, trade_id, trade_group_id)
        )

    matched.sort(key=lambda h: h.date, reverse=True)
    return matched


def cast_executions(raw_executions: object) -> list[dict[str, Any]]:
    """Safely normalizes raw execution structures to a list of dictionaries."""
    if isinstance(raw_executions, list):
        return [dict(e) for e in raw_executions if isinstance(e, dict)]
    return []


def compute_reality_check_summary(
    positions: Sequence[PositionRealityCheck],
    history: Sequence[HistoryRealityCheck],
) -> RealityCheckSummary:
    """Computes aggregate reality check KPIs over matched positions and history.

    Args:
        positions: Matched active position comparisons.
        history: Matched closed history comparisons.

    Returns:
        RealityCheckSummary: Aggregated KPI record.
    """
    open_count = len(positions)
    closed_count = len(history)
    matched_count = open_count + closed_count

    net_pnl_bt = sum((h.net_pnl_bt for h in history), Decimal("0.00"))
    net_pnl_broker = sum((h.net_pnl_broker for h in history), Decimal("0.00"))
    total_commissions = sum((h.commissions for h in history), Decimal("0.00"))

    if closed_count > 0:
        avg_entry = sum((h.entry_slippage for h in history), Decimal("0.00")) / Decimal(
            str(closed_count)
        )
        avg_exit = sum((h.exit_slippage for h in history), Decimal("0.00")) / Decimal(
            str(closed_count)
        )
    else:
        avg_entry = Decimal("0.00")
        avg_exit = Decimal("0.00")

    pnl_delta = net_pnl_broker - net_pnl_bt

    return RealityCheckSummary(
        matched_count=matched_count,
        open_matched_count=open_count,
        closed_matched_count=closed_count,
        net_pnl_bt=net_pnl_bt.quantize(Decimal("0.01")),
        net_pnl_broker=net_pnl_broker.quantize(Decimal("0.01")),
        total_commissions=total_commissions.quantize(Decimal("0.01")),
        avg_entry_slippage=avg_entry.quantize(Decimal("0.01")),
        avg_exit_slippage=avg_exit.quantize(Decimal("0.01")),
        pnl_delta=pnl_delta.quantize(Decimal("0.01")),
    )


def compute_dual_equity_curve(
    history: Sequence[HistoryRealityCheck],
) -> list[EquityPoint]:
    """Generates chronological data points for the cumulative dual-equity curve.

    Args:
        history: Matched closed history records.

    Returns:
        list[EquityPoint]: Chronologically ordered equity timeline points.
    """
    if not history:
        return []

    chronological = sorted(history, key=lambda h: h.date)

    cum_bt: float = 0.0
    cum_broker: float = 0.0
    points: list[EquityPoint] = []

    for h in chronological:
        cum_bt += float(h.net_pnl_bt)
        cum_broker += float(h.net_pnl_broker)
        delta = round(cum_broker - cum_bt, 2)

        points.append(
            EquityPoint(
                date=h.date,
                trade_id=h.trade_id,
                symbol=h.symbol,
                cumulative_pnl_bt=round(cum_bt, 2),
                cumulative_pnl_broker=round(cum_broker, 2),
                pnl_delta=delta,
            )
        )

    return points
