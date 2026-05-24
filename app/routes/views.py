import logging
from typing import Any, TypedDict
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from flask import Blueprint, current_app, render_template, request
from ..extensions import cache

from ..types import TradeStatus
from ..const import Strategies, ExitReason
from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..models import BacktestMetrics
from ..services.backtester.analytics import BacktestAnalytics
from ..services.backtester.backtest_results import ResultsPersistence
from ..services.screener.view_service import ScreenerViewService
from ..services.trade_manager.view_service import TradeViewService
from ..tools import metrics

logger = logging.getLogger(__name__)
views_bp = Blueprint("views", __name__)


def _get_database_path(name: str = "signals") -> Path:
    """Retrieves the absolute path to a specific database."""
    configuration = current_app.config["APP_CONFIG"]
    return Path(configuration.get_db_path(name)).resolve()


def _get_signal_repository() -> SignalRepository:
    """Instantiates the signal repository."""
    session = DatabaseSession(str(_get_database_path("signals")))
    return SignalRepository(session)


def _get_screener_view_service() -> ScreenerViewService:
    """Instantiates the screener view service."""
    return ScreenerViewService(_get_signal_repository())


def _get_trade_view_service() -> TradeViewService:
    """Instantiates the trade view service."""
    return TradeViewService()


def _get_backtest_database_path() -> Path:
    """Retrieves the absolute path to the backtest.db."""
    # Logic to find backtest.db relative to the signals database directory
    signals_db_path = _get_database_path("signals")
    return signals_db_path.parent / "backtest.db"


def _prepare_backtest_metrics(summary_data: dict[str, Any]) -> BacktestMetrics:
    """Maps summary dictionary data to a BacktestMetrics object.

    Args:
        summary_data: Dictionary containing raw backtest results.

    Returns:
        BacktestMetrics: Populated metrics object.
    """

    return BacktestMetrics(
        total_trades=summary_data.get("total_trades", 0),
        win_rate=summary_data.get("win_rate", 0.0),
        profit_factor=summary_data.get("profit_factor", 0.0),
        net_profit=summary_data.get("net_profit", 0.0),
        maximum_drawdown=summary_data.get("maximum_drawdown", 0.0),
        sharpe_ratio=summary_data.get("sharpe_ratio", 0.0),
        expectancy=summary_data.get("expectancy", 0.0),
        system_quality_number=summary_data.get("sqn", 0.0),
        kelly_safe=summary_data.get("kelly_safe", 0.0),
        strategy_return=summary_data.get("strategy_return", 0.0),
        benchmark_return=summary_data.get("benchmark_return", 0.0),
        # Fields not present in summary but required by dataclass
        kelly_criterion=summary_data.get("kelly_safe", 0.0),
        average_win=0.0,
        average_loss=0.0,
        average_maximum_adverse_excursion=0.0,
        average_maximum_favorable_excursion=0.0,
        risk_of_ruin=0.0,
        kelly_mean=0.0,
        kelly_std=0.0,
        market_exposure_pct=summary_data.get("market_exposure_pct", 0.0),
        risk_adjusted_benchmark=0.0,
        exposure_efficiency=0.0,
        return_over_maximum_drawdown=0.0,
        diversification_score=summary_data.get("diversification_score", 0.0),
    )


def _prepare_strategy_metrics(
    strategy_list: list[dict[str, Any]],
) -> dict[str, BacktestMetrics]:
    """Converts a list of strategy metrics dicts to a map of BacktestMetrics.

    Args:
        strategy_list: List of dictionaries from the strategy_metrics table.

    Returns:
        dict[str, BacktestMetrics]: Mapping of strategy name to metrics.
    """
    return {
        strategy["strategy_name"]: _prepare_backtest_metrics(strategy)
        for strategy in strategy_list
    }


# --- ROUTES ---


# 1. Landing Pages (Übersichten)
class StrategyOverview(TypedDict):
    """Represents a strategy summary for the screener dashboard."""

    id: str
    name: str
    desc: str
    icon: str
    count: int
    is_active: bool


def _get_strategy_overview(
    signals_repository: SignalRepository, screener_service: ScreenerViewService
) -> list[StrategyOverview]:
    """
    Fetches the overview data for all trading strategies.

    Args:
        signals_repository: The initialized signal repository.
        screener_service: The initialized screener view service.

    Returns:
        list[StrategyOverview]: A list containing strategy details and signal counts.
    """
    # Signale zählen
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

    return [
        {
            "id": "croc",
            "name": "Croc Setup",
            "desc": "Trendfolge-Signale basierend auf Wochen- und Tageschart-Momentum.",
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
    ]


@views_bp.route("/screener", methods=["GET"])
def view_screener_overview() -> str:
    """Displays the overview page for all available screeners."""
    signals_repository = _get_signal_repository()
    screener_service = _get_screener_view_service()

    strategies = _get_strategy_overview(signals_repository, screener_service)

    return render_template(
        "screener.html",
        strategies=strategies,
    )


def generate_sparkline(dates: list, prices: list, is_positive: bool) -> str:
    """Generates a minimalistic sparkline chart (Spline, No Axes)."""
    color = "#10b981" if is_positive else "#ef4444"  # Emerald-500 or Rose-500
    fill_color = "rgba(16, 185, 129, 0.1)" if is_positive else "rgba(239, 68, 68, 0.1)"

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            line=dict(color=color, width=2, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor=fill_color,
            hoverinfo="skip",
        )
    )

    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=50,
        width=120,
    )
    return figure.to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


def generate_donut_chart(labels: list, values: list, colors: list) -> str:
    """Generates a clean donut chart for strategy allocation."""
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.8,
                textinfo="none",
                hoverinfo="label+percent+value",
                marker=dict(colors=colors),
                sort=False,
            )
        ]
    )

    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=180,
    )
    return figure.to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


@views_bp.route("/trades", methods=["GET"])
@views_bp.route("/trades/", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_overview() -> str:
    """Displays an overview of all active trades across strategies."""
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
        strat_key = service.resolve_strategy(trade)

        # Robust grouping
        if strat_key in croc_group:
            label = "Croc Setup"
        elif strat_key == Strategies.DipBuyer:
            label = "Dip Buyer"
        elif strat_key in turnover_group:
            label = "Turnover"
        elif strat_key == Strategies.TwoPercent:
            label = "Two Percent"
        elif strat_key == Strategies.NDXMomentum:
            label = "NDX Momentum"
        else:
            label = str(trade.get("strategy", "Unknown"))

        if label not in strategy_stats:
            strategy_stats[label] = {"count": 0, "pnl": 0.0, "invested": 0.0}

        strategy_stats[label]["count"] += 1
        strategy_stats[label]["pnl"] += trade.get("unrealized_pnl", 0.0)

        entry_price = float(trade.get("entry_price") or 0.0)
        initial_size = float(trade.get("initial_size") or 0.0)
        strategy_stats[label]["invested"] += entry_price * initial_size

    # Prepare Data for Donut Chart
    allocation_labels = list(strategy_stats.keys())
    allocation_values = [data["invested"] for data in strategy_stats.values()]
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


@views_bp.route("/analytics", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_analytics_dashboard() -> str:
    """Displays the Strategy Overview Dashboard with performance analytics."""
    service = _get_trade_view_service()

    # 1. Fetch Closed Trades (Excluding invalid signals)
    closed_trades = service.get_trades(
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
    )
    active_trades = service.get_trades(status=TradeStatus.ACTIVE)

    if not closed_trades and not active_trades:
        return render_template(
            "analytics.html", summary=None, strategies=[], weekly_trend={}
        )

    if not closed_trades:
        df = pd.DataFrame(columns=["exit_date", "realized_pnl", "strategy"])
    else:
        df = pd.DataFrame(closed_trades)
    df["exit_date_dt"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date_dt")

    # 2. Summary Metrics (Using metrics.py)
    # Portfolio equity curve for drawdown calculation
    initial_capital = 100_000.0
    df["cum_pnl"] = df["realized_pnl"].cumsum()
    df["equity"] = initial_capital + df["cum_pnl"]

    summary = {
        "net_pnl": float(df["realized_pnl"].sum()),
        "win_rate": metrics.calculate_win_rate(df["realized_pnl"]),
        "max_drawdown": metrics.calculate_max_drawdown(df["equity"]),
        "total_trades": len(df),
    }

    # 3. Strategy Analysis
    strategies_data = []
    # Identify unique strategy groups
    strat_groups = {
        "Croc Setup": [
            Strategies.CrocSetup,
            Strategies.HoldTarget,
            Strategies.SplitTarget,
            "croc",
        ],
        "Dip Buyer": [Strategies.DipBuyer],
        "Turnover": [
            Strategies.TurnOverTiming,
            Strategies.TurnOverTiming_05,
            Strategies.TurnOverTiming_10,
        ],
        "Two Percent": [Strategies.TwoPercent],
        "NDX Momentum": [Strategies.NDXMomentum],
    }

    # Map colors
    strat_colors = {
        "Croc Setup": "#10b981",  # Emerald-500
        "Dip Buyer": "#3b82f6",  # Blue-500
        "Turnover": "#f59e0b",  # Amber-500
        "Two Percent": "#8b5cf6",  # Violet-500
        "NDX Momentum": "#f97316",  # Orange-500
    }

    for name, filters in strat_groups.items():
        strat_df = df[df["strategy"].isin(filters)] if not df.empty else pd.DataFrame()

        # Calculate active trades and open PnL
        strat_active = [
            t
            for t in active_trades
            if service.resolve_strategy(t) in filters or t.get("strategy") in filters
        ]
        open_pnl = sum([t.get("unrealized_pnl", 0.0) for t in strat_active])

        if strat_df.empty and not strat_active:
            continue

        winning_trades = (
            strat_df[strat_df["realized_pnl"] > 0]
            if not strat_df.empty
            else pd.DataFrame()
        )
        losing_trades = (
            strat_df[strat_df["realized_pnl"] < 0]
            if not strat_df.empty
            else pd.DataFrame()
        )

        # Risk Multiples (R) calculation if risk is available in context
        # Fallback to simple expectancy if R is not easily derived here
        strat_expectancy = (
            metrics.calculate_expectancy(strat_df["realized_pnl"])
            if not strat_df.empty
            else 0.0
        )

        # Calculate Risk and RoR
        trade_risks_dollars = []
        for _, trade in strat_df.iterrows():
            entry = float(trade.get("entry_price") or 0.0)
            stop = float(trade.get("stop_loss") or 0.0)
            size = float(trade.get("initial_size") or 0.0)
            
            if entry > 0 and stop > 0 and size > 0:
                risk = abs(entry - stop) * size
                if risk > 0:
                    trade_risks_dollars.append(risk)
                    
        avg_risk_dollar = sum(trade_risks_dollars) / len(trade_risks_dollars) if trade_risks_dollars else 100.0
        avg_risk_pct = (avg_risk_dollar / initial_capital) * 100
        
        ror_pct = (float(strat_df["realized_pnl"].sum()) / initial_capital) * 100 if not strat_df.empty else 0.0
        expectancy_r = strat_expectancy / avg_risk_dollar if avg_risk_dollar > 0 else 0.0

        strategies_data.append(
            {
                "id": name.lower().replace(" ", "-"),
                "name": name,
                "color": strat_colors.get(name, "#64748b"),
                "pnl": float(strat_df["realized_pnl"].sum())
                if not strat_df.empty
                else 0.0,
                "open_pnl": open_pnl,
                "trades_count": len(strat_df),
                "metrics": {
                    "trades": len(strat_df),
                    "active_positions": len(strat_active),
                    "avg_risk": f"{avg_risk_pct:.2f}%",
                    "win_count": len(winning_trades),
                    "loss_count": len(losing_trades),
                    "avg_win": float(winning_trades["realized_pnl"].mean())
                    if not winning_trades.empty
                    else 0.0,
                    "avg_loss": float(losing_trades["realized_pnl"].mean())
                    if not losing_trades.empty
                    else 0.0,
                    "profit_factor": metrics.calculate_profit_factor(
                        strat_df["realized_pnl"]
                    )
                    if not strat_df.empty
                    else 0.0,
                    "expectancy": f"{expectancy_r:.2f} R",
                    "ror": f"{ror_pct:.2f}%",
                    "sharpe": metrics.calculate_sharpe_ratio(
                        strat_df["realized_pnl"], initial_capital
                    )
                    if not strat_df.empty
                    else 0.0,
                },
            }
        )

    # Sort strategies by PnL desc
    strategies_data.sort(key=lambda x: x["pnl"], reverse=True)

    # 4. Weekly Trend Data (Plotly) - Since 01.01.2026
    start_of_year = pd.Timestamp("2026-01-01")
    today = pd.Timestamp.now()

    # Create a full weekly range to ensure no gaps at the start
    date_range = pd.date_range(start=start_of_year, end=today, freq="W-SUN")

    # Filter trades for chart
    chart_df = df[df["exit_date_dt"] >= start_of_year].copy()

    if chart_df.empty:
        weekly_trend = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in strat_groups},
        }
    else:
        # Resample to weekly
        df_weekly = (
            chart_df.set_index("exit_date_dt")["realized_pnl"]
            .resample("W-SUN")
            .sum()
            .cumsum()
        )
        # Reindex to full range
        df_weekly = df_weekly.reindex(date_range, method="ffill").fillna(0.0)

        weekly_trend = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": df_weekly.tolist(),
            "strategies": {},
        }

        for name, filters in strat_groups.items():
            s_trades = chart_df[chart_df["strategy"].isin(filters)]
            if s_trades.empty:
                weekly_trend["strategies"][name] = [0.0] * len(date_range)
                continue

            s_df = (
                s_trades.set_index("exit_date_dt")["realized_pnl"]
                .resample("W-SUN")
                .sum()
                .cumsum()
            )
            s_df = s_df.reindex(date_range, method="ffill").fillna(0.0)
            weekly_trend["strategies"][name] = s_df.tolist()

    return render_template(
        "analytics.html",
        summary=summary,
        strategies=strategies_data,
        weekly_trend=weekly_trend,
        active_page="analytics",
    )


# 2. Screener Details
@views_bp.route("/screener/croc", methods=["GET"])
def view_screener_croc() -> str:
    """Displays the Croc Setup screener results."""
    limit = request.args.get("limit", 200, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates("Croc_", limit=limit)
    return render_template("screener_croc.html", results=results)


@views_bp.route("/screener/dip-buyer", methods=["GET"])
def view_screener_dip_buyer() -> str:
    """Displays the Dip Buyer screener results."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.DipBuyer, limit=limit)
    return render_template("screener_dip_buyer.html", results=results)


@views_bp.route("/screener/turnover", methods=["GET"])
def view_screener_turnover() -> str:
    """Displays the Turnover Timing screener results, aggregated by symbol."""
    limit = request.args.get("limit", 200, type=int)
    service = _get_screener_view_service()
    results = service.get_turnover_candidates(limit=limit)
    return render_template("screener_turnover.html", results=results)


@views_bp.route("/screener/twopercent", methods=["GET"])
def view_screener_twopercent() -> str:
    """Displays the Two Percent Screener with trade candidates."""
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.TwoPercent, limit=limit)
    return render_template("screener_twopercent.html", results=results)


@views_bp.route("/screener/ndx-momentum", methods=["GET"])
def view_screener_ndx_momentum() -> str:
    """Displays the NDX Momentum screener results."""
    limit = request.args.get("limit", 50, type=int)
    service = _get_screener_view_service()
    results = service.get_candidates(Strategies.NDXMomentum, limit=limit)
    return render_template("screener_ndx_momentum.html", results=results)


# 3. Trades Details
@views_bp.route("/trades/croc", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_croc() -> str:
    """Displays the Croc Setup trade history and active positions."""
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    # Filtering
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

    # Summaries
    summary_metrics = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    # Aggregations
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
            signal_stats[signal_name] = {"count": 0, "win": 0, "loss": 0, "pnl": 0.0}

        pnl = trade["realized_pnl"]
        service._update_stat(signal_stats[signal_name], pnl)

    # Add Avg PnL
    for value in signal_stats.values():
        value["average_pnl"] = (
            value["pnl"] / value["count"] if value["count"] > 0 else 0.0
        )

    sorted_signals = dict(
        sorted(signal_stats.items(), key=lambda item: item[1]["count"], reverse=True)
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
    limit = request.args.get("limit", 100, type=int)
    service = _get_trade_view_service()

    active = service.get_trades(
        strategies=Strategies.DipBuyer, status=TradeStatus.ACTIVE
    )
    # Sort by Entry Date Descending
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
    # active_groups removed as we use flat table now
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
    """Displays the Turnover Timing trade history and active positions."""
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
        trade["green_candle_count"] = trade.get("context", {}).get("green_candle_count", 0)

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
    summary = service.get_portfolio_summary(active)
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
            service._update_stat(variant_stats[variant_key], trade["realized_pnl"])

    # Calc Averages for Variants
    for item in variant_stats.values():
        item["average_pnl"] = item["pnl"] / item["count"] if item["count"] > 0 else 0.0

    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_turnover.html",
        summary=summary,
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
    """Displays the NDX Momentum trade history and active positions."""
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

    summary = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_ndx_momentum.html",
        active_trades=active,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary,
        closed_summary=closed_summary,
    )


@views_bp.route("/trades/twopercent", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_trades_twopercent() -> str:
    """Displays the Two Percent trade history and active positions."""
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

    summary = service.get_portfolio_summary(active)
    closed_summary = service.get_closed_summary(closed)

    active_groups = service.group_trades_by_symbol(active)
    history_groups = service.group_trades_history(closed)

    return render_template(
        "trades_twopercent.html",
        active_trades=active,
        active_groups=active_groups,
        closed_trades=closed,
        history_groups=history_groups,
        summary=summary,
        closed_summary=closed_summary,
        index_stats={},  # Standard compatibility
    )


@views_bp.route("/backtest", methods=["GET"])
def view_backtest_dashboard() -> str:
    """Displays the backtest dashboard by retrieving pre-calculated results."""
    # 1. Configuration & Paths
    backtest_database_path = _get_backtest_database_path()
    market_database_path = _get_database_path("stocks")

    # 2. Results Persistence
    persistence = ResultsPersistence(str(backtest_database_path))
    run_identifier = (
        request.args.get("run_id", type=int) or persistence.get_latest_run_id()
    )

    if not run_identifier:
        return f"No backtest results found. Please run the backtester first. DB Path: {backtest_database_path}"

    # 3. Data Retrieval (DRY Principle: No simulations in the route)
    run_data = persistence.get_run_results(run_identifier)
    if not run_data:
        return f"Results for Run ID {run_identifier} not found."

    summary_data = run_data["summary"]

    # 4. Metric Preparation (Object Mapping)
    main_metrics = _prepare_backtest_metrics(summary_data)
    strategy_metrics_map = _prepare_strategy_metrics(run_data["strategies"])
    portfolio_metrics = run_data["portfolio"]

    # 5. Chart Data Preparation
    # Convert lists of dicts from DB back to DataFrames for chart functions
    equity_dataframe = pd.DataFrame(run_data.get("equity_curves", []))
    regime_dataframe = pd.DataFrame(run_data.get("regime_data", []))
    exposure_dataframe = pd.DataFrame(run_data.get("exposure_data", []))

    # Identify Dates for Benchmarks
    start_date_str = summary_data["start_date"]
    end_date_str = summary_data["end_date"]

    # 6. Chart Generation
    analytics = BacktestAnalytics(
        str(backtest_database_path), str(market_database_path)
    )
    dashboard_charts = _generate_dashboard_charts(
        analytics=analytics,
        main_metrics=main_metrics,
        equity_dataframe=equity_dataframe,
        regime_dataframe=regime_dataframe,
        exposure_dataframe=exposure_dataframe,
        start_date=start_date_str,
        end_date=end_date_str,
    )

    # 7. Rendering
    trade_lists = analytics.get_trade_lists()

    safety_impact = run_data.get("safety_impact", {})
    final_eq = safety_impact.get("final_equity", 100_000.0)
    kelly_metrics_view = {
        "net_profit": final_eq - 100_000.0,
        "total_return": (final_eq - 100_000.0) / 100_000.0,
        "max_drawdown": portfolio_metrics.get("leveraged_max_drawdown", 0.0)
        if portfolio_metrics
        else 0.0,
    }

    return render_template(
        "backtest_dashboard.html",
        run_id=run_identifier,
        metrics=main_metrics,
        kelly_metrics=kelly_metrics_view,
        strategy_metrics=strategy_metrics_map,
        wfa_results=run_data.get("wfa", []),
        stress_results=run_data.get("stress", {}),
        funnel_data=run_data.get("funnel", []),
        quality_data=run_data.get("quality", []),
        start_date=start_date_str,
        end_date=end_date_str,
        **dashboard_charts,
        recent_trades=trade_lists["recent"],
        top_trades=trade_lists["top"],
        worst_trades=trade_lists["worst"],
    )


def _generate_dashboard_charts(
    analytics: BacktestAnalytics,
    main_metrics: BacktestMetrics,
    equity_dataframe: pd.DataFrame,
    regime_dataframe: pd.DataFrame,
    exposure_dataframe: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    """Orchestrates chart generation for the backtest dashboard.

    Args:
        analytics: Analytics engine for benchmark fetching.
        equity_dataframe: Time series of equity curves.
        regime_dataframe: Time series of regime/VIX data.
        exposure_dataframe: Time series of strategy utilization.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        dict[str, str]: Map of template variable names to HTML chart strings.
    """
    from ..services.backtester.charts import (
        generate_backtest_charts,
        generate_profit_factor_gauge,
        generate_win_rate_gauge,
        generate_sqn_gauge,
        generate_regime_overlay_chart,
        generate_price_of_safety_chart,
        generate_exposure_heatmap,
        generate_risk_reward_scatter,
    )

    # 1. Benchmarks
    initial_capital = 100_000.0
    spy_dataframe = analytics.fetch_benchmark_data(
        "SPY", start_date, end_date, initial_capital=initial_capital
    )
    qqq_dataframe = analytics.fetch_benchmark_data(
        "QQQ", start_date, end_date, initial_capital=initial_capital
    )

    # 2. Split Equity Curves
    base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Base"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Portfolio"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Kelly"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Safety"]
    kelly_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Kelly"]
    safety_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Safety"]

    # 3. Generate individual charts
    chart_equity_base, chart_drawdown_base = (
        "<div>No Data</div>",
        "<div>No Data</div>",
    )
    if not base_equity.empty:
        chart_equity_base, chart_drawdown_base = generate_backtest_charts(
            base_equity["date"],
            base_equity["equity"],
            base_equity["drawdown_pct"],
            benchmark_df=spy_dataframe,
            id_prefix="base",
        )

    chart_equity_kelly, chart_drawdown_kelly = (
        "<div>No Data</div>",
        "<div>No Data</div>",
    )
    if not kelly_equity.empty:
        chart_equity_kelly, chart_drawdown_kelly = generate_backtest_charts(
            kelly_equity["date"],
            kelly_equity["equity"],
            kelly_equity["drawdown_pct"],
            benchmark_df=spy_dataframe,
            id_prefix="kelly",
        )

    # Specialized Charts
    # Rename 'vix_close' to 'vix' for generate_regime_overlay_chart compatibility
    regime_input = regime_dataframe.rename(columns={"vix_close": "vix"})

    # FIX: Merge 'equity' from base_equity into regime_input for the overlay chart
    if not base_equity.empty:
        regime_input = pd.merge(
            regime_input, base_equity[["date", "equity"]], on="date", how="left"
        )
    else:
        regime_input["equity"] = 0.0

    # Rename exposure columns for generate_exposure_heatmap compatibility
    # Expected format: exposure_<strategy_name>
    exposure_pivot = exposure_dataframe.pivot(
        index="date", columns="strategy_name", values="exposure_value"
    ).reset_index()
    exposure_pivot.columns = [
        f"exposure_{col}" if col != "date" else col for col in exposure_pivot.columns
    ]

    return {
        "chart_equity": chart_equity_base,
        "chart_underwater": chart_drawdown_base,
        "chart_equity_kelly": chart_equity_kelly,
        "chart_underwater_kelly": chart_drawdown_kelly,
        "chart_regime": generate_regime_overlay_chart(regime_input),
        "chart_safety": generate_price_of_safety_chart(
            kelly_equity,
            base_equity,
            safety_equity,
            spy_dataframe=spy_dataframe,
            qqq_dataframe=qqq_dataframe,
        ),
        "chart_exposure": generate_exposure_heatmap(exposure_pivot),
        "chart_risk": generate_risk_reward_scatter(safety_equity),
        "chart_pf": generate_profit_factor_gauge(main_metrics.profit_factor),
        "chart_wr": generate_win_rate_gauge(main_metrics.win_rate * 100),
        "chart_sqn": generate_sqn_gauge(main_metrics.system_quality_number),
    }
