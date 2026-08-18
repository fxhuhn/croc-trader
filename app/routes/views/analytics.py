"""Routes and views for performance analytics and allocation dashboard."""

import logging

import numpy as np
import pandas as pd
from flask import render_template, request

from ...const import ExitReason, Strategies
from ...database.repositories.market import MarketRepository
from ...services.trade_manager.view_service import TradeViewService
from ...tools import metrics
from ...tools.portfolio_analytics import (
    GERMAN_MONTH_NAMES,
    apply_depot_and_ev_allocations,
    calculate_active_months,
    calculate_benchmark_monthly_returns,
    calculate_concurrent_exposure,
    calculate_evm_allocations,
    calculate_kelly_metrics,
    calculate_mean_variance_allocations,
    calculate_mean_variance_dashboard_data,
    calculate_monthly_matrix_data,
    calculate_strategy_risk_and_expectancy,
    calculate_unweighted_monthly_pct,
    calculate_win_loss_rois,
    extract_roi_series,
)
from ...types import TradeData, TradeStatus
from .blueprint import views_bp
from .dependencies import (
    _get_trade_view_service,
    cache,
)

logger = logging.getLogger(__name__)

# Re-exports for backwards compatibility with tests
_calculate_unweighted_monthly_pct = calculate_unweighted_monthly_pct
_calculate_evm_allocations = calculate_evm_allocations
_calculate_mean_variance_allocations = calculate_mean_variance_allocations
_calculate_mean_variance_dashboard_data = calculate_mean_variance_dashboard_data

DEFAULT_INITIAL_CAPITAL: float = 100_000.0

STRATEGY_GROUPS: dict[str, list[Strategies | str]] = {
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
    "TGIM": [Strategies.TGIM],
    "Bridge Scout": [Strategies.BridgeScout],
    "Bounce Bandit": [Strategies.BounceBandit],
}

STRATEGY_COLORS: dict[str, str] = {
    "Croc Setup": "#10b981",
    "Dip Buyer": "#6366f1",
    "Turnover": "#f59e0b",
    "Two Percent": "#a855f7",
    "NDX Momentum": "#f43f5e",
    "TGIM": "#0284c7",
    "Bridge Scout": "#0ea5e9",
    "Bounce Bandit": "#8b5cf6",
}


@views_bp.route("/analytics", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_analytics_dashboard() -> str:
    """Displays the Strategy Overview Dashboard with performance analytics."""
    service = _get_trade_view_service()
    closed_trades = service.get_trades(
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
    )
    active_trades = service.get_trades(status=TradeStatus.ACTIVE)

    today = pd.Timestamp.now()
    current_month_name = f"{GERMAN_MONTH_NAMES[today.month]} {today.year}"

    if not closed_trades and not active_trades:
        return _render_empty_dashboard(current_month_name)

    dataframe = _prepare_closed_trades_dataframe(closed_trades)
    initial_capital = DEFAULT_INITIAL_CAPITAL
    summary = _calculate_summary_metrics(dataframe, initial_capital, today)

    strategies_data = _build_strategies_dashboard(
        dataframe, active_trades, service, initial_capital
    )
    monthly_evm = _calculate_monthly_evm(dataframe, today)
    monthly_mv = _calculate_monthly_mv(dataframe, today)
    mean_variance_data = calculate_mean_variance_dashboard_data(
        dataframe, STRATEGY_GROUPS
    )
    weekly_trend, weekly_pnl = _build_weekly_trend_data(dataframe, today)

    return render_template(
        "analytics.html",
        summary=summary,
        strategies=strategies_data,
        weekly_trend=weekly_trend,
        weekly_pnl=weekly_pnl,
        monthly_evm=monthly_evm,
        monthly_mv=monthly_mv,
        mean_variance_data=mean_variance_data,
        current_month_name=current_month_name,
        active_page="analytics",
    )


def _render_empty_dashboard(current_month_name: str) -> str:
    """Renders the analytics dashboard with empty fallback state."""
    empty_summary = {
        "return_ytd_pct": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "profit_factor": 0.0,
        "sample_size": 0,
        "net_pnl": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
    }
    return render_template(
        "analytics.html",
        summary=empty_summary,
        strategies=[],
        weekly_trend={},
        weekly_pnl={},
        monthly_evm={"months": [], "allocations": {}},
        monthly_mv={"months": [], "max_sharpe": {}, "risk_parity": {}},
        mean_variance_data={"strategies": [], "has_low_data": True},
        current_month_name=current_month_name,
        active_page="analytics",
    )


def _prepare_closed_trades_dataframe(
    closed_trades: list[TradeData] | list[dict[str, object]],
) -> pd.DataFrame:
    """Normalizes and prepares closed trades DataFrame."""
    if not closed_trades:
        dataframe = pd.DataFrame(columns=["exit_date", "realized_pnl", "strategy"])
    else:
        dataframe = pd.DataFrame(closed_trades)

    for column_name in (
        "exit_date",
        "realized_pnl",
        "strategy",
        "entry_price",
        "initial_size",
        "entry_date",
        "stop_loss",
    ):
        if column_name not in dataframe.columns:
            dataframe[column_name] = np.nan

    exit_dates = pd.to_datetime(dataframe["exit_date"], errors="coerce")
    if hasattr(exit_dates.dt, "tz") and exit_dates.dt.tz is not None:
        dataframe["exit_date_dt"] = exit_dates.dt.tz_localize(None)
    else:
        dataframe["exit_date_dt"] = exit_dates
    return dataframe.sort_values("exit_date_dt")


def _calculate_summary_metrics(
    dataframe: pd.DataFrame,
    initial_capital: float,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Calculates top-level summary metrics scoped to YTD on invested capital."""
    current_date = as_of_date if as_of_date is not None else pd.Timestamp.now()
    start_of_year = pd.Timestamp(year=current_date.year, month=1, day=1)

    if dataframe.empty or "exit_date_dt" not in dataframe.columns:
        ytd_dataframe = pd.DataFrame(columns=dataframe.columns)
    else:
        ytd_dataframe = dataframe[dataframe["exit_date_dt"] >= start_of_year]

    if ytd_dataframe.empty:
        return {
            "return_ytd_pct": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "profit_factor": 0.0,
            "sample_size": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
        }

    pnl_series = pd.to_numeric(ytd_dataframe["realized_pnl"], errors="coerce").fillna(
        0.0
    )

    _, portfolio_row, _ = calculate_monthly_matrix_data(
        dataframe, current_date.year, STRATEGY_GROUPS
    )
    return_ytd_pct = float(portfolio_row.get("gesamt", 0.0))

    cumulative_pnl = pnl_series.cumsum()

    equity_curve = initial_capital + cumulative_pnl
    max_drawdown = metrics.calculate_max_drawdown(equity_curve, initial_capital)

    roi_series = extract_roi_series(ytd_dataframe)
    active_months = calculate_active_months(ytd_dataframe)
    trades_per_year = (
        (len(roi_series) / active_months) * 12.0 if active_months > 0.0 else 252.0
    )

    sharpe_ratio = metrics.calculate_sharpe_ratio_from_roi(
        roi_series, trades_per_year=trades_per_year
    )
    sortino_ratio = metrics.calculate_sortino_ratio_from_roi(
        roi_series, trades_per_year=trades_per_year
    )
    profit_factor = metrics.calculate_profit_factor(pnl_series)

    return {
        "return_ytd_pct": return_ytd_pct,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "profit_factor": profit_factor,
        "sample_size": len(ytd_dataframe),
        "net_pnl": float(pnl_series.sum()),
        "win_rate": metrics.calculate_win_rate(pnl_series),
        "total_trades": len(ytd_dataframe),
    }


def _build_strategies_dashboard(
    dataframe: pd.DataFrame,
    active_trades: list[TradeData] | list[dict[str, object]],
    service: TradeViewService,
    initial_capital: float,
) -> list[dict[str, object]]:
    """Builds per-strategy performance metrics and applies portfolio allocation weighting."""
    strategies_data: list[dict[str, object]] = []

    for name, filters in STRATEGY_GROUPS.items():
        strategy_dataframe = (
            dataframe[dataframe["strategy"].isin(filters)]
            if not dataframe.empty
            else pd.DataFrame()
        )
        strategy_active_trades = [
            trade
            for trade in active_trades
            if service.resolve_strategy(trade) in filters
            or trade.get("strategy") in filters
        ]
        if strategy_dataframe.empty and not strategy_active_trades:
            continue

        item = _compute_strategy_item(
            name, strategy_dataframe, strategy_active_trades, initial_capital
        )
        strategies_data.append(item)

    apply_depot_and_ev_allocations(strategies_data, initial_capital)
    strategies_data.sort(key=lambda x: float(x["pnl"]), reverse=True)
    return strategies_data


def _compute_strategy_item(
    name: str,
    strategy_dataframe: pd.DataFrame,
    strategy_active_trades: list[dict[str, object]],
    initial_capital: float,
) -> dict[str, object]:
    """Calculates all metrics for a single strategy."""
    open_pnl = sum(
        float(trade.get("unrealized_pnl", 0.0) or 0.0)
        for trade in strategy_active_trades
    )
    roi_series = extract_roi_series(strategy_dataframe)
    average_roi = float(roi_series.mean()) if not roi_series.empty else 0.0
    active_months = calculate_active_months(strategy_dataframe)

    trades_per_month = len(strategy_dataframe) / active_months
    expected_value_per_month = trades_per_month * average_roi
    max_concurrent, percentile_95 = calculate_concurrent_exposure(
        strategy_dataframe, strategy_active_trades
    )

    avg_risk_text, expectancy_text = calculate_strategy_risk_and_expectancy(
        strategy_dataframe, initial_capital
    )
    win_rate, risk_reward_ratio, kelly_criterion = calculate_kelly_metrics(
        roi_series, strategy_active_trades
    )
    average_win_roi, average_loss_roi = calculate_win_loss_rois(strategy_dataframe)

    pnl = (
        float(strategy_dataframe["realized_pnl"].sum())
        if not strategy_dataframe.empty
        else 0.0
    )
    rate_of_return = (
        (pnl / initial_capital) * 100.0 if not strategy_dataframe.empty else 0.0
    )

    return {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "color": STRATEGY_COLORS.get(name, "#64748b"),
        "pnl": pnl,
        "open_pnl": open_pnl,
        "trades_count": len(strategy_dataframe),
        "metrics": {
            "trades": len(strategy_dataframe),
            "active_positions": len(strategy_active_trades),
            "avg_risk": avg_risk_text,
            "win_count": len(strategy_dataframe[strategy_dataframe["realized_pnl"] > 0])
            if not strategy_dataframe.empty
            else 0,
            "loss_count": len(
                strategy_dataframe[strategy_dataframe["realized_pnl"] < 0]
            )
            if not strategy_dataframe.empty
            else 0,
            "avg_win": average_win_roi,
            "avg_loss": average_loss_roi,
            "profit_factor": metrics.calculate_profit_factor(
                strategy_dataframe["realized_pnl"]
            )
            if not strategy_dataframe.empty
            else 0.0,
            "expectancy": expectancy_text,
            "ror": f"{rate_of_return:.2f}%",
            "max_concurrent": max_concurrent,
            "percentile_95": percentile_95,
            "avg_roi": f"{average_roi * 100.0:.2f}%",
            "sharpe": metrics.calculate_sharpe_ratio(
                strategy_dataframe["realized_pnl"], initial_capital
            )
            if not strategy_dataframe.empty
            else 0.0,
            "win_rate": win_rate,
            "risk_reward_ratio": risk_reward_ratio,
            "kelly_criterion": kelly_criterion,
            "trades_per_month": trades_per_month,
            "ev_per_trade_roi": average_roi,
            "ev_per_month": expected_value_per_month,
        },
    }


def _calculate_monthly_evm(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> dict[str, object]:
    """Generates monthly cumulative EV/M allocation time series."""
    start_date = pd.Timestamp(year=today.year, month=1, day=1)
    date_range = pd.date_range(start=start_date, end=today, freq="MS")
    labels = [f"{GERMAN_MONTH_NAMES[m.month]} {m.year}" for m in date_range]
    allocations: dict[str, list[float]] = {name: [] for name in STRATEGY_GROUPS}

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )
        slice_dataframe = (
            dataframe[dataframe["exit_date_dt"] <= slice_end]
            if not dataframe.empty
            else pd.DataFrame()
        )
        month_allocations = calculate_evm_allocations(slice_dataframe, STRATEGY_GROUPS)
        for name in STRATEGY_GROUPS:
            allocations[name].append(month_allocations[name])

    return {"months": labels, "allocations": allocations}


def _calculate_monthly_mv(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> dict[str, object]:
    """Generates monthly cumulative Max-Sharpe and Risk-Parity allocation time series."""
    start_date = pd.Timestamp(year=today.year, month=1, day=1)
    date_range = pd.date_range(start=start_date, end=today, freq="MS")
    labels = [f"{GERMAN_MONTH_NAMES[m.month]} {m.year}" for m in date_range]
    max_sharpe_allocations: dict[str, list[float]] = {
        name: [] for name in STRATEGY_GROUPS
    }
    risk_parity_allocations: dict[str, list[float]] = {
        name: [] for name in STRATEGY_GROUPS
    }

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )
        slice_dataframe = (
            dataframe[dataframe["exit_date_dt"] <= slice_end]
            if not dataframe.empty
            else pd.DataFrame()
        )
        max_sharpe = calculate_mean_variance_allocations(
            slice_dataframe, STRATEGY_GROUPS, model_type="max_sharpe"
        )
        risk_parity = calculate_mean_variance_allocations(
            slice_dataframe, STRATEGY_GROUPS, model_type="risk_parity"
        )
        for name in STRATEGY_GROUPS:
            max_sharpe_allocations[name].append(max_sharpe[name])
            risk_parity_allocations[name].append(risk_parity[name])

    return {
        "months": labels,
        "max_sharpe": max_sharpe_allocations,
        "risk_parity": risk_parity_allocations,
    }


def _build_weekly_trend_data(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> tuple[dict[str, object], dict[str, object]]:
    """Builds cumulative and non-cumulative weekly performance series for Plotly charts."""
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    days_until_saturday = (5 - today.weekday()) % 7
    week_saturday = today.normalize() + pd.Timedelta(days=days_until_saturday)
    date_range = pd.date_range(start=start_of_year, end=week_saturday, freq="W-SAT")
    dates_formatted = [date_val.strftime("%Y-%m-%d") for date_val in date_range]
    week_labels = [
        f"{date_val.strftime('%d.%m.%Y')} · KW {date_val.isocalendar().week}"
        for date_val in date_range
    ]

    chart_dataframe = dataframe[dataframe["exit_date_dt"] >= start_of_year].copy()
    if chart_dataframe.empty:
        empty_trend = {
            "dates": dates_formatted,
            "week_labels": week_labels,
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in STRATEGY_GROUPS},
        }
        return empty_trend, empty_trend.copy()

    aggregate_cumsum = (
        chart_dataframe.set_index("exit_date_dt")["realized_pnl"]
        .resample("W-SAT")
        .sum()
        .cumsum()
        .reindex(date_range, method="ffill")
        .fillna(0.0)
    )
    aggregate_pnl = (
        chart_dataframe.set_index("exit_date_dt")["realized_pnl"]
        .resample("W-SAT")
        .sum()
        .reindex(date_range, fill_value=0.0)
        .fillna(0.0)
    )

    weekly_trend: dict[str, object] = {
        "dates": dates_formatted,
        "week_labels": week_labels,
        "aggregate": aggregate_cumsum.tolist(),
        "strategies": {},
    }
    weekly_pnl: dict[str, object] = {
        "dates": dates_formatted,
        "week_labels": week_labels,
        "aggregate": aggregate_pnl.tolist(),
        "strategies": {},
    }

    trend_strategies: dict[str, list[float]] = {}
    pnl_strategies: dict[str, list[float]] = {}

    for name, filters in STRATEGY_GROUPS.items():
        strategy_trades = chart_dataframe[chart_dataframe["strategy"].isin(filters)]
        if strategy_trades.empty:
            trend_strategies[name] = [0.0] * len(date_range)
            pnl_strategies[name] = [0.0] * len(date_range)
            continue

        strategy_resampled = (
            strategy_trades.set_index("exit_date_dt")["realized_pnl"]
            .resample("W-SAT")
            .sum()
        )
        trend_strategies[name] = (
            strategy_resampled.cumsum()
            .reindex(date_range, method="ffill")
            .fillna(0.0)
            .tolist()
        )
        pnl_strategies[name] = (
            strategy_resampled.reindex(date_range, fill_value=0.0).fillna(0.0).tolist()
        )

    weekly_trend["strategies"] = trend_strategies
    weekly_pnl["strategies"] = pnl_strategies

    return weekly_trend, weekly_pnl


@views_bp.route("/analytics/monthly-matrix", methods=["GET"])
@views_bp.route("/analytics/monthlymatrix", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_analytics_monthly_matrix() -> str:
    """Displays the Desktop-optimized Monthly Performance Matrix view."""
    service = _get_trade_view_service()
    current_year = pd.Timestamp.now().year
    raw_year = request.args.get("year")
    try:
        selected_year = int(raw_year) if raw_year else current_year
    except (ValueError, TypeError):
        selected_year = current_year

    available_years = [2024, 2025, 2026]
    if selected_year not in available_years:
        available_years.append(selected_year)
        available_years.sort()

    month_names = [
        "Jan",
        "Feb",
        "Mär",
        "Apr",
        "Mai",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Dez",
    ]

    closed_trades = service.get_trades(
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
    )
    dataframe = _prepare_closed_trades_dataframe(closed_trades)

    matrix_rows, portfolio_row, portfolio_models_rows = calculate_monthly_matrix_data(
        dataframe, selected_year, STRATEGY_GROUPS
    )
    benchmark_rows = _fetch_benchmark_rows(service.market_repository, selected_year)

    return render_template(
        "analytics_monthly_matrix.html",
        selected_year=selected_year,
        available_years=available_years,
        months=month_names,
        matrix_rows=matrix_rows,
        portfolio_row=portfolio_row,
        portfolio_models_rows=portfolio_models_rows,
        benchmark_rows=benchmark_rows,
        active_page="analytics",
        active_subpage="monthly_matrix",
    )


def _fetch_benchmark_rows(
    market_repository: MarketRepository, selected_year: int
) -> list[dict[str, object]]:
    """Fetches benchmark performance series for SPY and QQQ."""
    benchmark_symbols = [("SPY (S&P 500)", "SPY"), ("QQQ (Nasdaq 100)", "QQQ")]
    return [
        _fetch_single_benchmark(market_repository, symbol, label, selected_year)
        for label, symbol in benchmark_symbols
    ]


def _fetch_single_benchmark(
    market_repository: MarketRepository, symbol: str, label: str, selected_year: int
) -> dict[str, object]:
    """Fetches and calculates monthly returns for a single benchmark index."""
    try:
        benchmark_dataframe = market_repository.get_symbol_history_raw(
            symbol, start_date=f"{selected_year}-01-01"
        )
    except Exception as fetch_error:
        logger.warning(
            "Failed to fetch historical benchmark data for %s (%d): %s",
            symbol,
            selected_year,
            fetch_error,
        )
        benchmark_dataframe = pd.DataFrame()

    return calculate_benchmark_monthly_returns(
        benchmark_dataframe, label, selected_year
    )
