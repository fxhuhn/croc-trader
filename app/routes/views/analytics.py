"""Routes and views for performance analytics and allocation dashboard."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from flask import render_template, request

from ...const import ExitReason, Strategies
from ...tools import metrics
from ...tools.portfolio_analytics import (
    GERMAN_MONTH_NAMES,
    calculate_active_months,
    calculate_concurrent_exposure,
    calculate_evm_allocations,
    calculate_mean_variance_allocations,
    calculate_mean_variance_dashboard_data,
    calculate_monthly_matrix_data,
    calculate_unweighted_monthly_pct,
    extract_roi_series,
)
from ...types import TradeStatus
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

STRATEGY_GROUPS: dict[str, list[Any]] = {
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
    "Bridge Scout": [Strategies.BridgeScout, "bridge_scout"],
    "Bounce Bandit": [Strategies.BounceBandit, "bounce_bandit"],
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
    initial_capital = 100000.0
    summary = _calculate_summary_metrics(dataframe, initial_capital)

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
        "net_pnl": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
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
    )


def _prepare_closed_trades_dataframe(
    closed_trades: list[dict[str, Any]],
) -> pd.DataFrame:
    """Normalizes and prepares closed trades DataFrame."""
    if not closed_trades:
        dataframe = pd.DataFrame(columns=["exit_date", "realized_pnl", "strategy"])
    else:
        dataframe = pd.DataFrame(closed_trades)

    for col in (
        "exit_date",
        "realized_pnl",
        "strategy",
        "entry_price",
        "initial_size",
        "entry_date",
        "stop_loss",
    ):
        if col not in dataframe.columns:
            dataframe[col] = np.nan

    dataframe["exit_date_dt"] = pd.to_datetime(dataframe["exit_date"], errors="coerce")
    return dataframe.sort_values("exit_date_dt")


def _calculate_summary_metrics(
    dataframe: pd.DataFrame, initial_capital: float
) -> dict[str, Any]:
    """Calculates top-level summary metrics."""
    cum_pnl = dataframe["realized_pnl"].fillna(0.0).cumsum()
    equity_curve = initial_capital + cum_pnl
    return {
        "net_pnl": float(dataframe["realized_pnl"].fillna(0.0).sum()),
        "win_rate": metrics.calculate_win_rate(dataframe["realized_pnl"].fillna(0.0)),
        "max_drawdown": metrics.calculate_max_drawdown(equity_curve, initial_capital),
        "total_trades": len(dataframe),
    }


def _build_strategies_dashboard(
    dataframe: pd.DataFrame,
    active_trades: list[dict[str, Any]],
    service: Any,
    initial_capital: float,
) -> list[dict[str, Any]]:
    """Builds per-strategy performance metrics and applies portfolio allocation weighting."""
    strategies_data: list[dict[str, Any]] = []

    for name, filters in STRATEGY_GROUPS.items():
        strat_df = (
            dataframe[dataframe["strategy"].isin(filters)]
            if not dataframe.empty
            else pd.DataFrame()
        )
        strat_active = [
            t
            for t in active_trades
            if service.resolve_strategy(t) in filters or t.get("strategy") in filters
        ]
        if strat_df.empty and not strat_active:
            continue

        item = _compute_strategy_item(name, strat_df, strat_active, initial_capital)
        strategies_data.append(item)

    _apply_depot_and_ev_allocations(strategies_data)
    strategies_data.sort(key=lambda x: x["pnl"], reverse=True)
    return strategies_data


def _compute_strategy_item(
    name: str,
    strat_df: pd.DataFrame,
    strat_active: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    """Calculates all metrics for a single strategy."""
    open_pnl = sum([float(t.get("unrealized_pnl", 0.0) or 0.0) for t in strat_active])
    roi_series = extract_roi_series(strat_df)
    avg_roi = float(roi_series.mean()) if not roi_series.empty else 0.0
    active_months = calculate_active_months(strat_df)

    trades_per_month = len(strat_df) / active_months
    ev_per_month = trades_per_month * avg_roi
    max_concurrent, percentile_95 = calculate_concurrent_exposure(
        strat_df, strat_active
    )

    avg_risk_text, expectancy_text = _compute_risk_and_expectancy(
        strat_df, initial_capital
    )
    win_rate, rrr, kelly = _compute_kelly_metrics(roi_series, strat_active)
    avg_win_roi, avg_loss_roi = _compute_win_loss_rois(strat_df)

    pnl = float(strat_df["realized_pnl"].sum()) if not strat_df.empty else 0.0
    ror = (pnl / initial_capital) * 100.0 if not strat_df.empty else 0.0

    return {
        "id": name.lower().replace(" ", "-"),
        "name": name,
        "color": STRATEGY_COLORS.get(name, "#64748b"),
        "pnl": pnl,
        "open_pnl": open_pnl,
        "trades_count": len(strat_df),
        "metrics": {
            "trades": len(strat_df),
            "active_positions": len(strat_active),
            "avg_risk": avg_risk_text,
            "win_count": len(strat_df[strat_df["realized_pnl"] > 0])
            if not strat_df.empty
            else 0,
            "loss_count": len(strat_df[strat_df["realized_pnl"] < 0])
            if not strat_df.empty
            else 0,
            "avg_win": avg_win_roi,
            "avg_loss": avg_loss_roi,
            "profit_factor": metrics.calculate_profit_factor(strat_df["realized_pnl"])
            if not strat_df.empty
            else 0.0,
            "expectancy": expectancy_text,
            "ror": f"{ror:.2f}%",
            "max_concurrent": max_concurrent,
            "percentile_95": percentile_95,
            "avg_roi": f"{avg_roi * 100.0:.2f}%",
            "sharpe": metrics.calculate_sharpe_ratio(
                strat_df["realized_pnl"], initial_capital
            )
            if not strat_df.empty
            else 0.0,
            "win_rate": win_rate,
            "risk_reward_ratio": rrr,
            "kelly_criterion": kelly,
            "trades_per_month": trades_per_month,
            "ev_per_trade_roi": avg_roi,
            "ev_per_month": ev_per_month,
        },
    }


def _compute_risk_and_expectancy(
    strat_df: pd.DataFrame, initial_capital: float
) -> tuple[str, str]:
    """Computes average dollar/percentage risk and expectancy in R-multiples."""
    if strat_df.empty:
        return "N/A", "N/A"

    entry_prices = pd.to_numeric(strat_df["entry_price"], errors="coerce").fillna(0.0)
    stop_losses = pd.to_numeric(strat_df["stop_loss"], errors="coerce").fillna(0.0)
    initial_sizes = pd.to_numeric(strat_df["initial_size"], errors="coerce").fillna(0.0)

    valid_mask = (entry_prices > 0.0) & (stop_losses > 0.0) & (initial_sizes > 0.0)
    risks = (entry_prices - stop_losses).abs() * initial_sizes
    valid_risks = risks[valid_mask & (risks > 0.0)]

    if valid_risks.empty:
        return "N/A", "N/A"

    avg_risk_dollar = float(valid_risks.mean())
    avg_risk_pct = (avg_risk_dollar / initial_capital) * 100.0
    expectancy = metrics.calculate_expectancy(strat_df["realized_pnl"])
    expectancy_r = expectancy / avg_risk_dollar if avg_risk_dollar > 0.0 else 0.0
    return f"{avg_risk_pct:.2f}%", f"{expectancy_r:.2f} R"


def _compute_kelly_metrics(
    roi_series: pd.Series, strat_active: list[dict[str, Any]]
) -> tuple[float, float, float]:
    """Calculates Win Rate, Risk Reward Ratio, and Kelly Criterion from closed & active ROIs."""
    active_rois = []
    for trade in strat_active:
        upnl = float(trade.get("unrealized_pnl") or 0.0)
        ep = float(trade.get("entry_price") or 0.0)
        qty = float(trade.get("quantity") or trade.get("initial_size") or 0.0)
        inv = ep * qty
        active_rois.append(upnl / inv if inv > 0.0 else 0.0)

    combined_roi = pd.Series(roi_series.tolist() + active_rois)
    if combined_roi.empty:
        return 0.0, 0.0, 0.0

    win_rate = metrics.calculate_win_rate(combined_roi)
    rrr = metrics.calculate_risk_reward_ratio(combined_roi)
    kelly = metrics.calculate_kelly_criterion(win_rate, rrr)
    return win_rate, rrr, kelly


def _compute_win_loss_rois(strat_df: pd.DataFrame) -> tuple[float, float]:
    """Computes arithmetic average win ROI and loss ROI percentages."""
    if strat_df.empty:
        return 0.0, 0.0

    win_trades = strat_df[strat_df["realized_pnl"] > 0]
    loss_trades = strat_df[strat_df["realized_pnl"] < 0]

    def avg_roi_pct(subset: pd.DataFrame) -> float:
        if subset.empty:
            return 0.0
        e = pd.to_numeric(subset["entry_price"], errors="coerce").fillna(0.0)
        s = pd.to_numeric(subset["initial_size"], errors="coerce").fillna(0.0)
        inv = e * s
        mask = inv > 0.0
        if not mask.any():
            return 0.0
        return float((subset.loc[mask, "realized_pnl"] / inv[mask]).mean() * 100.0)

    return avg_roi_pct(win_trades), avg_roi_pct(loss_trades)


def _apply_depot_and_ev_allocations(strategies_data: list[dict[str, Any]]) -> None:
    """Scales Kelly values to depot limit and computes EV/M allocation percentages."""
    total_proposed = sum(
        max(0.0, float(s["metrics"]["kelly_criterion"])) for s in strategies_data
    )
    depot_mult = 1.0 / total_proposed if total_proposed > 1.0 else 1.0

    total_ev = sum(
        max(0.0, float(s["metrics"]["ev_per_month"])) for s in strategies_data
    )

    for s in strategies_data:
        raw_kelly = float(s["metrics"]["kelly_criterion"])
        s["metrics"]["suggested_allocation"] = (
            raw_kelly * depot_mult if raw_kelly > 0.0 else 0.0
        )
        raw_ev = float(s["metrics"]["ev_per_month"])
        ev_weight = (raw_ev / total_ev) if (raw_ev > 0.0 and total_ev > 0.0) else 0.0
        s["metrics"]["ev_allocation"] = ev_weight
        s["metrics"]["ev_allocation_100k"] = ev_weight * 100000.0


def _calculate_monthly_evm(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> dict[str, Any]:
    """Generates monthly cumulative EV/M allocation time series."""
    date_range = pd.date_range(start="2026-01-01", end=today, freq="MS")
    labels = [f"{GERMAN_MONTH_NAMES[m.month]} {m.year}" for m in date_range]
    allocations = {name: [] for name in STRATEGY_GROUPS}

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )
        slice_df = (
            dataframe[dataframe["exit_date_dt"] <= slice_end]
            if not dataframe.empty
            else pd.DataFrame()
        )
        month_allocs = calculate_evm_allocations(slice_df, STRATEGY_GROUPS)
        for name in STRATEGY_GROUPS:
            allocations[name].append(month_allocs[name])

    return {"months": labels, "allocations": allocations}


def _calculate_monthly_mv(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> dict[str, Any]:
    """Generates monthly cumulative Max-Sharpe and Risk-Parity allocation time series."""
    date_range = pd.date_range(start="2026-01-01", end=today, freq="MS")
    labels = [f"{GERMAN_MONTH_NAMES[m.month]} {m.year}" for m in date_range]
    ms_allocs = {name: [] for name in STRATEGY_GROUPS}
    rp_allocs = {name: [] for name in STRATEGY_GROUPS}

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )
        slice_df = (
            dataframe[dataframe["exit_date_dt"] <= slice_end]
            if not dataframe.empty
            else pd.DataFrame()
        )
        ms = calculate_mean_variance_allocations(
            slice_df, STRATEGY_GROUPS, model_type="max_sharpe"
        )
        rp = calculate_mean_variance_allocations(
            slice_df, STRATEGY_GROUPS, model_type="risk_parity"
        )
        for name in STRATEGY_GROUPS:
            ms_allocs[name].append(ms[name])
            rp_allocs[name].append(rp[name])

    return {"months": labels, "max_sharpe": ms_allocs, "risk_parity": rp_allocs}


def _build_weekly_trend_data(
    dataframe: pd.DataFrame, today: pd.Timestamp
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Builds cumulative and non-cumulative weekly performance series for Plotly charts."""
    start_of_year = pd.Timestamp("2026-01-01")
    days_until_sat = (5 - today.weekday()) % 7
    week_sat = today.normalize() + pd.Timedelta(days=days_until_sat)
    date_range = pd.date_range(start=start_of_year, end=week_sat, freq="W-SAT")
    dates_formatted = [d.strftime("%Y-%m-%d") for d in date_range]
    week_labels = [
        f"{d.strftime('%d.%m.%Y')} · KW {d.isocalendar().week}" for d in date_range
    ]

    chart_df = dataframe[dataframe["exit_date_dt"] >= start_of_year].copy()
    if chart_df.empty:
        empty_trend = {
            "dates": dates_formatted,
            "week_labels": week_labels,
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in STRATEGY_GROUPS},
        }
        return empty_trend, empty_trend.copy()

    agg_cumsum = (
        chart_df.set_index("exit_date_dt")["realized_pnl"]
        .resample("W-SAT")
        .sum()
        .cumsum()
        .reindex(date_range, method="ffill")
        .fillna(0.0)
    )
    agg_pnl = (
        chart_df.set_index("exit_date_dt")["realized_pnl"]
        .resample("W-SAT")
        .sum()
        .reindex(date_range, fill_value=0.0)
        .fillna(0.0)
    )

    weekly_trend: dict[str, Any] = {
        "dates": dates_formatted,
        "week_labels": week_labels,
        "aggregate": agg_cumsum.tolist(),
        "strategies": {},
    }
    weekly_pnl: dict[str, Any] = {
        "dates": dates_formatted,
        "week_labels": week_labels,
        "aggregate": agg_pnl.tolist(),
        "strategies": {},
    }

    for name, filters in STRATEGY_GROUPS.items():
        s_trades = chart_df[chart_df["strategy"].isin(filters)]
        if s_trades.empty:
            weekly_trend["strategies"][name] = [0.0] * len(date_range)
            weekly_pnl["strategies"][name] = [0.0] * len(date_range)
            continue

        s_resample = (
            s_trades.set_index("exit_date_dt")["realized_pnl"].resample("W-SAT").sum()
        )
        weekly_trend["strategies"][name] = (
            s_resample.cumsum().reindex(date_range, method="ffill").fillna(0.0).tolist()
        )
        weekly_pnl["strategies"][name] = (
            s_resample.reindex(date_range, fill_value=0.0).fillna(0.0).tolist()
        )

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


def _fetch_benchmark_rows(market_repo: Any, selected_year: int) -> list[dict[str, Any]]:
    """Fetches benchmark performance series for SPY and QQQ."""
    benchmark_symbols = [("SPY (S&P 500)", "SPY"), ("QQQ (Nasdaq 100)", "QQQ")]
    return [
        _fetch_single_benchmark(market_repo, sym, label, selected_year)
        for label, sym in benchmark_symbols
    ]


def _fetch_single_benchmark(
    market_repo: Any, symbol: str, label: str, selected_year: int
) -> dict[str, Any]:
    """Fetches and calculates monthly returns for a single benchmark index."""
    try:
        b_df = market_repo.get_symbol_history_raw(
            symbol, start_date=f"{selected_year}-01-01"
        )
    except Exception:
        b_df = pd.DataFrame()

    if b_df.empty:
        return {"name": label, "months": [0.0] * 12, "gesamt": 0.0}

    b_df["date_dt"] = pd.to_datetime(b_df["date"], errors="coerce")
    b_df = b_df[b_df["date_dt"].dt.year == selected_year].sort_values("date_dt")
    if b_df.empty:
        return {"name": label, "months": [0.0] * 12, "gesamt": 0.0}

    b_monthly: list[float] = []
    for month in range(1, 13):
        m_df = b_df[b_df["date_dt"].dt.month == month]
        if not m_df.empty:
            open_p = float(m_df.iloc[0]["open"] or m_df.iloc[0]["close"])
            close_p = float(m_df.iloc[-1]["close"])
            pct = ((close_p - open_p) / open_p * 100.0) if open_p > 0.0 else 0.0
        else:
            pct = 0.0
        b_monthly.append(round(pct, 1))

    year_open = float(b_df.iloc[0]["open"] or b_df.iloc[0]["close"])
    year_close = float(b_df.iloc[-1]["close"])
    b_gesamt = (
        ((year_close - year_open) / year_open * 100.0) if year_open > 0.0 else 0.0
    )
    return {"name": label, "months": b_monthly, "gesamt": round(b_gesamt, 1)}
