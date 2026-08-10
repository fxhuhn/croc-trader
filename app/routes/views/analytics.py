"""Routes and views for performance analytics and allocation dashboard."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from flask import render_template, request

from ...const import STRATEGY_ALIASES, ExitReason, Strategies
from ...tools import metrics
from ...tools.portfolio_optimization import (
    build_covariance_matrix,
    calculate_risk_contributions,
    compute_downside_deviation,
    optimize_max_sharpe_weights,
    optimize_risk_parity_weights,
)
from ...types import TradeStatus
from .blueprint import views_bp
from .dependencies import (
    _get_trade_view_service,
    cache,
)

logger = logging.getLogger(__name__)

MIN_SERIES_LEN: int = 2
LOW_DATA_THRESHOLD: int = 5
CALC_EPSILON: float = 1e-6


@views_bp.route("/analytics", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_analytics_dashboard() -> str:
    """Displays the Strategy Overview Dashboard with performance analytics.

    Returns:
        str: Rendered HTML dashboard template.
    """
    service = _get_trade_view_service()

    # 1. Fetch Closed Trades (Excluding invalid signals)
    closed_trades = service.get_trades(
        status=TradeStatus.CLOSED,
        exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
    )
    active_trades = service.get_trades(status=TradeStatus.ACTIVE)

    if not closed_trades and not active_trades:
        empty_summary = {
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
        }
        german_month_names = {
            1: "Januar",
            2: "Februar",
            3: "März",
            4: "April",
            5: "Mai",
            6: "Juni",
            7: "Juli",
            8: "August",
            9: "September",
            10: "Oktober",
            11: "November",
            12: "Dezember",
        }
        today = pd.Timestamp.now()
        current_month_name = f"{german_month_names[today.month]} {today.year}"
        empty_monthly_evm = {"months": [], "allocations": {}}
        empty_monthly_mv = {
            "months": [],
            "max_sharpe": {},
            "risk_parity": {},
        }
        empty_mv_data = {"strategies": [], "has_low_data": True}

        return render_template(
            "analytics.html",
            summary=empty_summary,
            strategies=[],
            weekly_trend={},
            weekly_pnl={},
            monthly_evm=empty_monthly_evm,
            monthly_mv=empty_monthly_mv,
            mean_variance_data=empty_mv_data,
            current_month_name=current_month_name,
        )

    if not closed_trades:
        dataframe = pd.DataFrame(columns=["exit_date", "realized_pnl", "strategy"])
    else:
        dataframe = pd.DataFrame(closed_trades)

    expected_trade_columns = (
        "exit_date",
        "realized_pnl",
        "strategy",
        "entry_price",
        "initial_size",
        "entry_date",
        "stop_loss",
    )
    for column_name in expected_trade_columns:
        if column_name not in dataframe.columns:
            dataframe[column_name] = np.nan

    dataframe["exit_date_dt"] = pd.to_datetime(dataframe["exit_date"], errors="coerce")
    dataframe = dataframe.sort_values("exit_date_dt")

    # 2. Summary Metrics (Using metrics.py)
    # Portfolio equity curve for drawdown calculation
    initial_capital = 100000.0
    dataframe["cum_pnl"] = dataframe["realized_pnl"].fillna(0.0).cumsum()
    dataframe["equity"] = initial_capital + dataframe["cum_pnl"]

    summary = {
        "net_pnl": float(dataframe["realized_pnl"].fillna(0.0).sum()),
        "win_rate": metrics.calculate_win_rate(dataframe["realized_pnl"].fillna(0.0)),
        "max_drawdown": metrics.calculate_max_drawdown(
            dataframe["equity"], initial_capital
        ),
        "total_trades": len(dataframe),
    }

    # 3. Strategy Analysis
    strategies_data = []
    # Identify unique strategy groups
    strategy_groups = {
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

    # Map colors to match the exact hex codes configured in the template
    strategy_colors = {
        "Croc Setup": "#10b981",  # Emerald-500
        "Dip Buyer": "#6366f1",  # Indigo-500
        "Turnover": "#f59e0b",  # Amber-500
        "Two Percent": "#a855f7",  # Purple-500
        "NDX Momentum": "#f43f5e",  # Rose-500
        "TGIM": "#0284c7",  # Sky-600
        "Bridge Scout": "#0ea5e9",  # Sky-500
        "Bounce Bandit": "#8b5cf6",  # Violet-500
    }

    for name, filters in strategy_groups.items():
        if not dataframe.empty:
            strategy_dataframe = dataframe[dataframe["strategy"].isin(filters)]
        else:
            strategy_dataframe = pd.DataFrame()

        # Calculate active trades and open PnL
        strategy_active = [
            trade
            for trade in active_trades
            if service.resolve_strategy(trade) in filters
            or trade.get("strategy") in filters
        ]
        open_profit_and_loss = sum(
            [float(t.get("unrealized_pnl", 0.0) or 0.0) for t in strategy_active]
        )

        if strategy_dataframe.empty and not strategy_active:
            continue

        winning_trades = (
            strategy_dataframe[strategy_dataframe["realized_pnl"] > 0]
            if not strategy_dataframe.empty
            else pd.DataFrame()
        )
        losing_trades = (
            strategy_dataframe[strategy_dataframe["realized_pnl"] < 0]
            if not strategy_dataframe.empty
            else pd.DataFrame()
        )

        strategy_expectancy = (
            metrics.calculate_expectancy(strategy_dataframe["realized_pnl"])
            if not strategy_dataframe.empty
            else 0.0
        )

        # Average ROI Calculation (per-trade arithmetic mean)
        average_roi = 0.0
        average_roi_display_text = "0.00%"
        roi_series = pd.Series(dtype=float)
        if not strategy_dataframe.empty:
            entry_prices = pd.to_numeric(
                strategy_dataframe["entry_price"], errors="coerce"
            ).fillna(0.0)
            initial_sizes = pd.to_numeric(
                strategy_dataframe["initial_size"], errors="coerce"
            ).fillna(0.0)
            invested_capital = entry_prices * initial_sizes

            valid_roi_mask = invested_capital > 0.0
            if valid_roi_mask.any():
                roi_series = (
                    strategy_dataframe.loc[valid_roi_mask, "realized_pnl"]
                    / invested_capital[valid_roi_mask]
                )
                average_roi = float(roi_series.mean())
                average_roi_display_text = f"{average_roi * 100.0:.2f}%"

        # Frequency Model (EV/M) Calculations — entry-to-exit span
        active_months = 1.0
        if not strategy_dataframe.empty and "entry_date" in strategy_dataframe.columns:
            entry_dates = pd.to_datetime(
                strategy_dataframe["entry_date"], errors="coerce"
            ).dropna()
            exit_dates = (
                strategy_dataframe["exit_date_dt"].dropna()
                if "exit_date_dt" in strategy_dataframe.columns
                else pd.Series(dtype="datetime64[ns]")
            )
            if not entry_dates.empty and not exit_dates.empty:
                first_date = entry_dates.min()
                last_date = exit_dates.max()
                days_span = max(1.0, float((last_date - first_date).days))
                active_months = max(1.0, days_span / 30.44)

        trades_per_month = len(strategy_dataframe) / active_months
        ev_per_trade_roi = average_roi
        ev_per_month = trades_per_month * ev_per_trade_roi

        # Max Open Positions Calculation
        max_concurrent = 0
        events = []
        if (
            not strategy_dataframe.empty
            and "entry_date" in strategy_dataframe.columns
            and "exit_date" in strategy_dataframe.columns
        ):
            entries = pd.to_datetime(
                strategy_dataframe["entry_date"], errors="coerce"
            ).dropna()
            exits = pd.to_datetime(
                strategy_dataframe["exit_date"], errors="coerce"
            ).dropna()
            for date in entries:
                events.append((date, 1))
            for date in exits:
                events.append((date, -1))

        for trade in strategy_active:
            if trade.get("entry_date"):
                try:
                    events.append((pd.to_datetime(trade["entry_date"]), 1))
                except Exception as error:
                    logger.warning(
                        "Invalid entry_date for trade %s: %s", trade.get("id"), error
                    )

        if events:
            events.sort(key=lambda x: (x[0], x[1]))
            current_open = 0
            for _, change in events:
                current_open += change
                max_concurrent = max(max_concurrent, current_open)

        # 95th Percentile Concurrent Trades Calculation
        percentile_95 = 0
        if events:
            # Group events by normalized date
            def normalize_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
                if ts.tzinfo is not None:
                    return ts.tz_convert(None).normalize()
                return ts.normalize()

            events_by_day = {}
            for date, change in events:
                day = normalize_timestamp(date)
                if day not in events_by_day:
                    events_by_day[day] = []
                events_by_day[day].append((date, change))

            sorted_days = sorted(events_by_day.keys())
            if sorted_days:
                min_date = sorted_days[0]
                max_date = sorted_days[-1]
                daily_range = pd.date_range(start=min_date, end=max_date, freq="D")

                daily_max = {}
                current_open = 0
                for day in daily_range:
                    day_events = events_by_day.get(day, [])
                    if day_events:
                        day_events.sort(key=lambda x: (x[0], x[1]))
                        max_on_day = current_open
                        for _, change in day_events:
                            current_open += change
                            max_on_day = max(max_on_day, current_open)
                        daily_max[day] = max_on_day
                    else:
                        daily_max[day] = current_open

                percentile_95 = int(round(np.percentile(list(daily_max.values()), 95)))

        # Vectorized Risk Calculations
        average_risk_dollar = 0.0
        has_valid_risk = False
        if not strategy_dataframe.empty:
            entry_prices = pd.to_numeric(
                strategy_dataframe["entry_price"], errors="coerce"
            ).fillna(0.0)
            stop_losses = pd.to_numeric(
                strategy_dataframe["stop_loss"], errors="coerce"
            ).fillna(0.0)
            initial_sizes = pd.to_numeric(
                strategy_dataframe["initial_size"], errors="coerce"
            ).fillna(0.0)

            valid_trades_mask = (
                (entry_prices > 0.0) & (stop_losses > 0.0) & (initial_sizes > 0.0)
            )
            risks_series = (entry_prices - stop_losses).abs() * initial_sizes
            valid_risks = risks_series[valid_trades_mask & (risks_series > 0.0)]

            if not valid_risks.empty:
                average_risk_dollar = float(valid_risks.mean())
                has_valid_risk = True

        if has_valid_risk:
            average_risk_percent = (average_risk_dollar / initial_capital) * 100.0
            expectancy_r = (
                strategy_expectancy / average_risk_dollar
                if average_risk_dollar > 0.0
                else 0.0
            )
            average_risk_display_text = f"{average_risk_percent:.2f}%"
            expectancy_display_text = f"{expectancy_r:.2f} R"
        else:
            average_risk_display_text = "N/A"
            expectancy_display_text = "N/A"

        rate_of_return_percent = (
            (float(strategy_dataframe["realized_pnl"].sum()) / initial_capital) * 100.0
            if not strategy_dataframe.empty
            else 0.0
        )

        # Sharpe Ratio — fixed annualization factor (Sharpe 1994 standard)
        sharpe_annualization_factor = 252.0

        # Calculate ROI series for Kelly Criterion
        closed_roi_list = []
        if "roi_series" in locals():
            closed_roi_list = roi_series.tolist()

        active_roi_list = []
        for trade in strategy_active:
            upnl = float(trade.get("unrealized_pnl") or 0.0)
            ep = float(trade.get("entry_price") or 0.0)
            # Active trades might use 'quantity' or 'initial_size'
            qty = float(trade.get("quantity") or trade.get("initial_size") or 0.0)
            inv = ep * qty
            if inv > 0.0:
                active_roi_list.append(upnl / inv)
            else:
                active_roi_list.append(0.0)

        total_roi_series = pd.Series(closed_roi_list + active_roi_list)

        # Calculate Kelly Criterion Allocation Metrics using ROI
        strategy_win_rate = (
            metrics.calculate_win_rate(total_roi_series)
            if not total_roi_series.empty
            else 0.0
        )
        strategy_risk_reward_ratio = (
            metrics.calculate_risk_reward_ratio(total_roi_series)
            if not total_roi_series.empty
            else 0.0
        )
        strategy_kelly_criterion = (
            metrics.calculate_kelly_criterion(
                strategy_win_rate, strategy_risk_reward_ratio
            )
            if not total_roi_series.empty
            else 0.0
        )

        # Calculate per-trade average W/L ROI (arithmetic mean of individual ROIs)
        avg_win_roi = 0.0
        if not winning_trades.empty:
            win_entry = pd.to_numeric(
                winning_trades["entry_price"], errors="coerce"
            ).fillna(0.0)
            win_size = pd.to_numeric(
                winning_trades["initial_size"], errors="coerce"
            ).fillna(0.0)
            win_invested = win_entry * win_size
            valid_wins = win_invested > 0.0
            if valid_wins.any():
                avg_win_roi = float(
                    (
                        winning_trades.loc[valid_wins, "realized_pnl"]
                        / win_invested[valid_wins]
                    ).mean()
                    * 100.0
                )

        avg_loss_roi = 0.0
        if not losing_trades.empty:
            loss_entry = pd.to_numeric(
                losing_trades["entry_price"], errors="coerce"
            ).fillna(0.0)
            loss_size = pd.to_numeric(
                losing_trades["initial_size"], errors="coerce"
            ).fillna(0.0)
            loss_invested = loss_entry * loss_size
            valid_losses = loss_invested > 0.0
            if valid_losses.any():
                avg_loss_roi = float(
                    (
                        losing_trades.loc[valid_losses, "realized_pnl"]
                        / loss_invested[valid_losses]
                    ).mean()
                    * 100.0
                )

        strategies_data.append(
            {
                "id": name.lower().replace(" ", "-"),
                "name": name,
                "color": strategy_colors.get(name, "#64748b"),
                "pnl": float(strategy_dataframe["realized_pnl"].sum())
                if not strategy_dataframe.empty
                else 0.0,
                "open_pnl": open_profit_and_loss,
                "trades_count": len(strategy_dataframe),
                "metrics": {
                    "trades": len(strategy_dataframe),
                    "active_positions": len(strategy_active),
                    "avg_risk": average_risk_display_text,
                    "win_count": len(winning_trades),
                    "loss_count": len(losing_trades),
                    "avg_win": float(avg_win_roi),
                    "avg_loss": float(avg_loss_roi),
                    "profit_factor": metrics.calculate_profit_factor(
                        strategy_dataframe["realized_pnl"]
                    )
                    if not strategy_dataframe.empty
                    else 0.0,
                    "expectancy": expectancy_display_text,
                    "ror": f"{rate_of_return_percent:.2f}%",
                    "max_concurrent": max_concurrent,
                    "percentile_95": percentile_95,
                    "avg_roi": average_roi_display_text,
                    "sharpe": metrics.calculate_sharpe_ratio(
                        strategy_dataframe["realized_pnl"],
                        initial_capital,
                        annualization_factor=sharpe_annualization_factor,
                    )
                    if not strategy_dataframe.empty
                    else 0.0,
                    "win_rate": strategy_win_rate,
                    "risk_reward_ratio": strategy_risk_reward_ratio,
                    "kelly_criterion": strategy_kelly_criterion,
                    "trades_per_month": trades_per_month,
                    "ev_per_trade_roi": ev_per_trade_roi,
                    "ev_per_month": ev_per_month,
                },
            }
        )

    # Calculate total proposed raw exposure from positive Kelly values
    total_proposed_exposure = sum(
        max(0.0, float(strategy_item["metrics"]["kelly_criterion"]))
        for strategy_item in strategies_data
    )

    # Determine depot-level scaling multiplier (capped at 1.0 maximum)
    maximum_depot_exposure = 1.0
    if total_proposed_exposure > maximum_depot_exposure:
        depot_multiplier = maximum_depot_exposure / total_proposed_exposure
    else:
        depot_multiplier = 1.0

    # Calculate total EV/M for weighting
    total_ev_per_month = sum(
        max(0.0, float(strategy_item["metrics"]["ev_per_month"]))
        for strategy_item in strategies_data
    )

    # Apply scaling multiplier to compute professional Suggested Allocations
    for strategy_item in strategies_data:
        raw_kelly = float(strategy_item["metrics"]["kelly_criterion"])
        suggested_allocation = raw_kelly * depot_multiplier if raw_kelly > 0.0 else 0.0
        strategy_item["metrics"]["suggested_allocation"] = suggested_allocation

        # EV/M Allocation
        raw_ev = float(strategy_item["metrics"]["ev_per_month"])
        if raw_ev > 0.0 and total_ev_per_month > 0.0:
            ev_weight = raw_ev / total_ev_per_month
        else:
            ev_weight = 0.0
        strategy_item["metrics"]["ev_allocation"] = ev_weight
        strategy_item["metrics"]["ev_allocation_100k"] = ev_weight * 100000.0

    # Sort strategies by realized pnl desc
    strategies_data.sort(key=lambda x: x["pnl"], reverse=True)

    # 3.5. Monthly EV/M Calculation (Cumulative)
    german_month_names = {
        1: "Januar",
        2: "Februar",
        3: "März",
        4: "April",
        5: "Mai",
        6: "Juni",
        7: "Juli",
        8: "August",
        9: "September",
        10: "Oktober",
        11: "November",
        12: "Dezember",
    }

    today = pd.Timestamp.now()
    current_month_name = f"{german_month_names[today.month]} {today.year}"

    # Generate list of month starts from 2026-01-01 to today
    monthly_date_range = pd.date_range(start="2026-01-01", end=today, freq="MS")
    monthly_labels = [
        f"{german_month_names[m.month]} {m.year}" for m in monthly_date_range
    ]

    # Pre-allocate dictionary arrays for allocations
    monthly_allocations = {name: [] for name in strategy_groups}

    for month_start in monthly_date_range:
        # Cumulative slice: all closed trades exiting up to the end of the month
        month_end = month_start + pd.offsets.MonthEnd(0)

        # If it is the current month, we slice up to today's date to include all trades up to now
        if month_start.year == today.year and month_start.month == today.month:
            slice_end = today
        else:
            slice_end = month_end

        slice_df = (
            dataframe[dataframe["exit_date_dt"] <= slice_end]
            if not dataframe.empty
            else pd.DataFrame()
        )

        # Calculate EV/M for each strategy on this slice
        strategy_evs = {}
        for name, filters in strategy_groups.items():
            strat_slice_df = (
                slice_df[slice_df["strategy"].isin(filters)]
                if not slice_df.empty
                else pd.DataFrame()
            )
            trades_count = len(strat_slice_df)

            average_roi = 0.0
            if not strat_slice_df.empty:
                entry_prices = pd.to_numeric(
                    strat_slice_df["entry_price"], errors="coerce"
                ).fillna(0.0)
                initial_sizes = pd.to_numeric(
                    strat_slice_df["initial_size"], errors="coerce"
                ).fillna(0.0)
                invested_capital = entry_prices * initial_sizes
                valid_roi_mask = invested_capital > 0.0
                if valid_roi_mask.any():
                    roi_per_trade = (
                        strat_slice_df.loc[valid_roi_mask, "realized_pnl"]
                        / invested_capital[valid_roi_mask]
                    )
                    average_roi = float(roi_per_trade.mean())

            # active_months: entry-to-exit span for accurate frequency
            active_months = 1.0
            if not strat_slice_df.empty and "entry_date" in strat_slice_df.columns:
                slice_entry_dates = pd.to_datetime(
                    strat_slice_df["entry_date"], errors="coerce"
                ).dropna()
                slice_exit_dates = (
                    strat_slice_df["exit_date_dt"].dropna()
                    if "exit_date_dt" in strat_slice_df.columns
                    else pd.Series(dtype="datetime64[ns]")
                )
                if not slice_entry_dates.empty and not slice_exit_dates.empty:
                    first_date = slice_entry_dates.min()
                    last_date = slice_exit_dates.max()
                    days_span = max(1.0, float((last_date - first_date).days))
                    active_months = max(1.0, days_span / 30.44)

            trades_per_month = trades_count / active_months
            ev_per_month = trades_per_month * average_roi
            strategy_evs[name] = ev_per_month

        # Calculate total positive EV/M for weighting in this month
        total_ev_per_month = sum(max(0.0, val) for val in strategy_evs.values())

        # Save allocations
        for name in strategy_groups:
            raw_ev = strategy_evs[name]
            if raw_ev > 0.0 and total_ev_per_month > 0.0:
                allocation = raw_ev / total_ev_per_month
            else:
                allocation = 0.0
            monthly_allocations[name].append(allocation)

    monthly_evm = {
        "months": monthly_labels,
        "allocations": monthly_allocations,
    }

    # 3.6. Monthly Mean-Variance & Risk Parity Calculation (Cumulative)
    monthly_mv_max_sharpe = {name: [] for name in strategy_groups}
    monthly_mv_risk_parity = {name: [] for name in strategy_groups}

    for month_start in monthly_date_range:
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

        ms_allocs = _calculate_mean_variance_allocations(
            slice_df, strategy_groups, model_type="max_sharpe"
        )
        rp_allocs = _calculate_mean_variance_allocations(
            slice_df, strategy_groups, model_type="risk_parity"
        )

        for name in strategy_groups:
            monthly_mv_max_sharpe[name].append(ms_allocs[name])
            monthly_mv_risk_parity[name].append(rp_allocs[name])

    monthly_mv = {
        "months": monthly_labels,
        "max_sharpe": monthly_mv_max_sharpe,
        "risk_parity": monthly_mv_risk_parity,
    }

    mean_variance_data = _calculate_mean_variance_dashboard_data(
        dataframe, strategy_groups
    )

    # 4. Weekly Trend Data (Plotly) - Since 01.01.2026
    start_of_year = pd.Timestamp("2026-01-01")

    # Create a full weekly range up to the end of the current week (Saturday)
    days_until_saturday = (5 - today.weekday()) % 7
    current_week_saturday = today.normalize() + pd.Timedelta(days=days_until_saturday)
    date_range = pd.date_range(
        start=start_of_year, end=current_week_saturday, freq="W-SAT"
    )

    # Filter trades for chart
    chart_dataframe = dataframe[dataframe["exit_date_dt"] >= start_of_year].copy()

    if chart_dataframe.empty:
        weekly_trend = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in strategy_groups},
        }
        weekly_profit_and_loss = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in strategy_groups},
        }
    else:
        # Cumulative Weekly Trend
        dataframe_weekly = (
            chart_dataframe.set_index("exit_date_dt")["realized_pnl"]
            .resample("W-SAT")
            .sum()
            .cumsum()
        )
        dataframe_weekly = dataframe_weekly.reindex(date_range, method="ffill").fillna(
            0.0
        )

        weekly_trend = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": dataframe_weekly.tolist(),
            "strategies": {},
        }

        # Non-Cumulative Weekly Performance
        dataframe_weekly_profit_and_loss = (
            chart_dataframe.set_index("exit_date_dt")["realized_pnl"]
            .resample("W-SAT")
            .sum()
        )
        dataframe_weekly_profit_and_loss = dataframe_weekly_profit_and_loss.reindex(
            date_range, fill_value=0.0
        ).fillna(0.0)

        weekly_profit_and_loss = {
            "dates": [d.strftime("%Y-%m-%d") for d in date_range],
            "aggregate": dataframe_weekly_profit_and_loss.tolist(),
            "strategies": {},
        }

        for name, filters in strategy_groups.items():
            strategy_trades = chart_dataframe[chart_dataframe["strategy"].isin(filters)]
            if strategy_trades.empty:
                weekly_trend["strategies"][name] = [0.0] * len(date_range)
                weekly_profit_and_loss["strategies"][name] = [0.0] * len(date_range)
                continue

            # Strategy Cumulative Trend
            strategy_cumulative = (
                strategy_trades.set_index("exit_date_dt")["realized_pnl"]
                .resample("W-SAT")
                .sum()
                .cumsum()
            )
            strategy_cumulative = strategy_cumulative.reindex(
                date_range, method="ffill"
            ).fillna(0.0)
            weekly_trend["strategies"][name] = strategy_cumulative.tolist()

            # Strategy Non-Cumulative Weekly profit and loss
            strategy_weekly_profit_and_loss = (
                strategy_trades.set_index("exit_date_dt")["realized_pnl"]
                .resample("W-SAT")
                .sum()
            )
            strategy_weekly_profit_and_loss = strategy_weekly_profit_and_loss.reindex(
                date_range, fill_value=0.0
            ).fillna(0.0)
            weekly_profit_and_loss["strategies"][name] = (
                strategy_weekly_profit_and_loss.tolist()
            )

    return render_template(
        "analytics.html",
        summary=summary,
        strategies=strategies_data,
        weekly_trend=weekly_trend,
        weekly_pnl=weekly_profit_and_loss,
        monthly_evm=monthly_evm,
        monthly_mv=monthly_mv,
        mean_variance_data=mean_variance_data,
        current_month_name=current_month_name,
        active_page="analytics",
    )


def _calculate_unweighted_monthly_pct(month_df: pd.DataFrame) -> float:
    """Calculates the unweighted average percentage return of trades in a month.

    Args:
        month_df: DataFrame of closed trades for the month.

    Returns:
        float: Unweighted average percentage return across trades.
    """
    if month_df.empty:
        return 0.0

    pnl = pd.to_numeric(month_df["realized_pnl"], errors="coerce").fillna(0.0)
    entry_prices = pd.to_numeric(month_df["entry_price"], errors="coerce").fillna(0.0)
    initial_sizes = pd.to_numeric(month_df["initial_size"], errors="coerce").fillna(0.0)

    invested = entry_prices * initial_sizes
    valid_mask = invested > 0.0

    if not valid_mask.any():
        return 0.0

    trade_pcts = (pnl[valid_mask] / invested[valid_mask]) * 100.0
    return float(trade_pcts.mean())


def _calculate_evm_allocations(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
) -> dict[str, float]:
    """Calculates EV/M strategy weights from a cumulative closed trades slice.

    Args:
        slice_df: DataFrame of closed trades up to cutoff date.
        strategy_groups: Strategy group name to filter list mapping.

    Returns:
        dict[str, float]: Mapping from strategy group name to percentage weight (0.0 to 1.0).
    """
    num_strats = len(strategy_groups)
    default_weight = 1.0 / num_strats if num_strats > 0 else 0.0

    if slice_df.empty or "strategy" not in slice_df.columns:
        return dict.fromkeys(strategy_groups, default_weight)

    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    strategy_evs: dict[str, float] = {}
    for name, filters in strategy_groups.items():
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]
        trades_count = len(strat_slice_df)

        average_roi = 0.0
        if not strat_slice_df.empty:
            entry_prices = pd.to_numeric(
                strat_slice_df["entry_price"], errors="coerce"
            ).fillna(0.0)
            initial_sizes = pd.to_numeric(
                strat_slice_df["initial_size"], errors="coerce"
            ).fillna(0.0)
            invested_capital = entry_prices * initial_sizes
            valid_roi_mask = invested_capital > 0.0
            if valid_roi_mask.any():
                roi_per_trade = (
                    strat_slice_df.loc[valid_roi_mask, "realized_pnl"]
                    / invested_capital[valid_roi_mask]
                )
                average_roi = float(roi_per_trade.mean())

        active_months = 1.0
        if not strat_slice_df.empty and "entry_date" in strat_slice_df.columns:
            slice_entry_dates = pd.to_datetime(
                strat_slice_df["entry_date"], errors="coerce"
            ).dropna()
            slice_exit_dates = (
                strat_slice_df["exit_date_dt"].dropna()
                if "exit_date_dt" in strat_slice_df.columns
                else pd.Series(dtype="datetime64[ns]")
            )
            if not slice_entry_dates.empty and not slice_exit_dates.empty:
                first_date = slice_entry_dates.min()
                last_date = slice_exit_dates.max()
                days_span = max(1.0, float((last_date - first_date).days))
                active_months = max(1.0, days_span / 30.44)

        trades_per_month = trades_count / active_months
        ev_per_month = trades_per_month * average_roi
        strategy_evs[name] = ev_per_month

    total_ev = sum(max(0.0, val) for val in strategy_evs.values())
    if total_ev > 0.0:
        return {
            name: (max(0.0, strategy_evs[name]) / total_ev) for name in strategy_groups
        }

    return dict.fromkeys(strategy_groups, default_weight)


def _calculate_mean_variance_allocations(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
    model_type: str = "max_sharpe",
) -> dict[str, float]:
    """Calculates Mean-Variance or Risk Parity strategy weights from a cumulative closed trades slice.

    Args:
        slice_df: DataFrame of closed trades up to cutoff date.
        strategy_groups: Strategy group name to filter list mapping.
        model_type: 'max_sharpe' or 'risk_parity'.

    Returns:
        dict[str, float]: Mapping from strategy group name to percentage weight (0.0 to 1.0).
    """
    num_strats = len(strategy_groups)
    default_weight = 1.0 / num_strats if num_strats > 0 else 0.0

    if slice_df.empty or "strategy" not in slice_df.columns:
        return dict.fromkeys(strategy_groups, default_weight)

    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    strat_returns: list[pd.Series] = []
    strat_names: list[str] = list(strategy_groups.keys())
    mus: list[float] = []

    for name in strat_names:
        filters = strategy_groups[name]
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]

        roi_series = pd.Series(dtype=float)
        if not strat_slice_df.empty:
            entry_prices = pd.to_numeric(
                strat_slice_df["entry_price"], errors="coerce"
            ).fillna(0.0)
            initial_sizes = pd.to_numeric(
                strat_slice_df["initial_size"], errors="coerce"
            ).fillna(0.0)
            invested_capital = entry_prices * initial_sizes
            valid_roi_mask = invested_capital > 0.0
            if valid_roi_mask.any():
                roi_series = (
                    strat_slice_df.loc[valid_roi_mask, "realized_pnl"]
                    / invested_capital[valid_roi_mask]
                )

        strat_returns.append(roi_series)
        mean_roi = float(roi_series.mean()) if not roi_series.empty else 0.0
        mus.append(mean_roi)

    max_len = max((len(s) for s in strat_returns), default=0)
    if max_len < MIN_SERIES_LEN:
        return dict.fromkeys(strategy_groups, default_weight)

    padded_dict = {
        name: ser.reset_index(drop=True)
        for name, ser in zip(strat_names, strat_returns, strict=True)
    }
    returns_df = pd.DataFrame(padded_dict)

    cov_matrix = build_covariance_matrix(returns_df)
    mu_vector = np.array(mus, dtype=float)

    if model_type == "risk_parity":
        opt_weights = optimize_risk_parity_weights(cov_matrix)
    else:
        opt_weights = optimize_max_sharpe_weights(mu_vector, cov_matrix)

    return {name: float(w) for name, w in zip(strat_names, opt_weights, strict=True)}


def _calculate_mean_variance_dashboard_data(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
) -> dict[str, Any]:
    """Computes comprehensive Section 6 metrics (Max Sharpe, Risk Parity, MCR, TRC, PRC).

    Args:
        slice_df: DataFrame of closed trades.
        strategy_groups: Strategy group mapping.

    Returns:
        dict[str, Any]: Structured dashboard data containing strategy details and data flags.
    """
    num_strats = len(strategy_groups)
    default_weight = 1.0 / num_strats if num_strats > 0 else 0.0
    strat_names = list(strategy_groups.keys())

    if slice_df.empty or "strategy" not in slice_df.columns:
        empty_strats = [
            {
                "name": name,
                "trades_per_month": 0.0,
                "mu": 0.0,
                "sigma": 0.0,
                "sigma_d": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "weight_max_sharpe": default_weight * 100.0,
                "weight_risk_parity": default_weight * 100.0,
                "mcr_max_sharpe": 0.0,
                "trc_max_sharpe": 0.0,
                "prc_max_sharpe": default_weight * 100.0,
                "mcr_risk_parity": 0.0,
                "trc_risk_parity": 0.0,
                "prc_risk_parity": default_weight * 100.0,
                "trades_count": 0,
            }
            for name in strat_names
        ]
        return {
            "strategies": empty_strats,
            "has_low_data": True,
        }

    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    strat_returns: list[pd.Series] = []
    trades_counts: list[int] = []
    trades_per_months: list[float] = []
    mus: list[float] = []
    sigmas: list[float] = []
    sigma_ds: list[float] = []
    sharpes: list[float] = []
    sortinos: list[float] = []

    has_low_data = False

    for name in strat_names:
        filters = strategy_groups[name]
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]
        t_count = len(strat_slice_df)
        trades_counts.append(t_count)
        if t_count < LOW_DATA_THRESHOLD:
            has_low_data = True

        roi_series = pd.Series(dtype=float)
        average_roi = 0.0
        if not strat_slice_df.empty:
            entry_prices = pd.to_numeric(
                strat_slice_df["entry_price"], errors="coerce"
            ).fillna(0.0)
            initial_sizes = pd.to_numeric(
                strat_slice_df["initial_size"], errors="coerce"
            ).fillna(0.0)
            invested_capital = entry_prices * initial_sizes
            valid_roi_mask = invested_capital > 0.0
            if valid_roi_mask.any():
                roi_series = (
                    strat_slice_df.loc[valid_roi_mask, "realized_pnl"]
                    / invested_capital[valid_roi_mask]
                )
                average_roi = float(roi_series.mean())

        active_months = 1.0
        if not strat_slice_df.empty and "entry_date" in strat_slice_df.columns:
            slice_entry_dates = pd.to_datetime(
                strat_slice_df["entry_date"], errors="coerce"
            ).dropna()
            slice_exit_dates = (
                strat_slice_df["exit_date_dt"].dropna()
                if "exit_date_dt" in strat_slice_df.columns
                else pd.Series(dtype="datetime64[ns]")
            )
            if not slice_entry_dates.empty and not slice_exit_dates.empty:
                first_date = slice_entry_dates.min()
                last_date = slice_exit_dates.max()
                days_span = max(1.0, float((last_date - first_date).days))
                active_months = max(1.0, days_span / 30.44)

        t_per_m = t_count / active_months
        trades_per_months.append(t_per_m)
        mu_val = t_per_m * average_roi
        mus.append(mu_val)

        strat_returns.append(roi_series)
        sig = float(roi_series.std(ddof=1)) if len(roi_series) > 1 else 0.0
        sigmas.append(sig)

        sig_d = compute_downside_deviation(roi_series) if not roi_series.empty else 0.0
        sigma_ds.append(sig_d)

        sh = (mu_val / sig) if sig > CALC_EPSILON else 0.0
        sharpes.append(sh)

        so = (mu_val / sig_d) if sig_d > CALC_EPSILON else 0.0
        sortinos.append(so)

    padded_dict = {
        name: ser.reset_index(drop=True)
        for name, ser in zip(strat_names, strat_returns, strict=True)
    }
    returns_df = pd.DataFrame(padded_dict)

    cov_matrix = build_covariance_matrix(returns_df)
    mu_vector = np.array(mus, dtype=float)

    weights_ms = optimize_max_sharpe_weights(mu_vector, cov_matrix)
    weights_rp = optimize_risk_parity_weights(cov_matrix)

    mcr_ms, trc_ms, prc_ms = calculate_risk_contributions(weights_ms, cov_matrix)
    mcr_rp, trc_rp, prc_rp = calculate_risk_contributions(weights_rp, cov_matrix)

    strategies_res = []
    for idx, name in enumerate(strat_names):
        strategies_res.append(
            {
                "name": name,
                "trades_per_month": trades_per_months[idx],
                "mu": mus[idx] * 100.0,
                "sigma": sigmas[idx] * 100.0,
                "sigma_d": sigma_ds[idx] * 100.0,
                "sharpe": sharpes[idx],
                "sortino": sortinos[idx],
                "weight_max_sharpe": weights_ms[idx] * 100.0,
                "weight_risk_parity": weights_rp[idx] * 100.0,
                "mcr_max_sharpe": mcr_ms[idx] * 100.0,
                "trc_max_sharpe": trc_ms[idx] * 100.0,
                "prc_max_sharpe": prc_ms[idx],
                "mcr_risk_parity": mcr_rp[idx] * 100.0,
                "trc_risk_parity": trc_rp[idx] * 100.0,
                "prc_risk_parity": prc_rp[idx],
                "trades_count": trades_counts[idx],
            }
        )

    return {
        "strategies": strategies_res,
        "has_low_data": has_low_data,
    }


@views_bp.route("/analytics/monthly-matrix", methods=["GET"])
@views_bp.route("/analytics/monthlymatrix", methods=["GET"])
@cache.cached(timeout=86400, query_string=True)
def view_analytics_monthly_matrix() -> str:
    """Displays the Desktop-optimized Monthly Performance Matrix view.

    Returns:
        str: Rendered HTML template.
    """
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

    if closed_trades:
        dataframe = pd.DataFrame(closed_trades)
        dataframe["exit_date_dt"] = pd.to_datetime(
            dataframe["exit_date"], errors="coerce"
        )
    else:
        dataframe = pd.DataFrame(
            columns=[
                "exit_date_dt",
                "realized_pnl",
                "strategy",
                "entry_price",
                "initial_size",
                "entry_date",
            ]
        )

    for column_name in ("realized_pnl", "entry_price", "initial_size", "entry_date"):
        if column_name not in dataframe.columns:
            dataframe[column_name] = np.nan

    year_dataframe = (
        dataframe[dataframe["exit_date_dt"].dt.year == selected_year].copy()
        if not dataframe.empty
        else pd.DataFrame()
    )

    strategy_groups = {
        "Croc Setup": [
            Strategies.CrocSetup,
            Strategies.HoldTarget,
            Strategies.SplitTarget,
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

    if not year_dataframe.empty:
        resolved_strategies = year_dataframe["strategy"].apply(
            lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
        )
    else:
        resolved_strategies = pd.Series(dtype=object)

    matrix_rows = []

    for name, filters in strategy_groups.items():
        if not year_dataframe.empty:
            strat_df = year_dataframe[resolved_strategies.isin(filters)].copy()
        else:
            strat_df = pd.DataFrame()

        monthly_pcts: list[float] = []

        for month in range(1, 13):
            if not strat_df.empty:
                month_df = strat_df[strat_df["exit_date_dt"].dt.month == month]
            else:
                month_df = pd.DataFrame()

            pct = _calculate_unweighted_monthly_pct(month_df)
            monthly_pcts.append(round(pct, 1))

        compounded_factor = 1.0
        for val in monthly_pcts:
            compounded_factor *= 1.0 + val / 100.0
        gesamt_pct = (compounded_factor - 1.0) * 100.0

        matrix_rows.append(
            {
                "name": name,
                "months": monthly_pcts,
                "gesamt": round(gesamt_pct, 1),
            }
        )

    num_strategies = len(matrix_rows)
    portfolio_monthly_pcts: list[float] = []

    for month_idx in range(12):
        if num_strategies > 0:
            month_sum = sum(row["months"][month_idx] for row in matrix_rows)
            port_pct = month_sum / num_strategies
        else:
            port_pct = 0.0
        portfolio_monthly_pcts.append(round(port_pct, 1))

    port_compounded_factor = 1.0
    for val in portfolio_monthly_pcts:
        port_compounded_factor *= 1.0 + val / 100.0
    port_gesamt_pct = (port_compounded_factor - 1.0) * 100.0

    portfolio_row = {
        "name": "Portfolio",
        "months": portfolio_monthly_pcts,
        "gesamt": round(port_gesamt_pct, 1),
    }

    # Calculate Frequenz-Modell (EV/M) row for Portfoliomodell category
    # January (month_idx 0) is identical to Standard (equal weight).
    # Month m > 1 uses EV/M strategy allocation calculated up to end of month m-1.
    evm_monthly_pcts: list[float] = []
    for month_idx in range(12):
        if month_idx == 0:
            evm_monthly_pcts.append(portfolio_monthly_pcts[0])
        else:
            prior_month_end = pd.Timestamp(
                year=selected_year, month=month_idx, day=1
            ) + pd.offsets.MonthEnd(0)
            slice_df = (
                dataframe[dataframe["exit_date_dt"] <= prior_month_end]
                if not dataframe.empty
                else pd.DataFrame()
            )
            weights = _calculate_evm_allocations(slice_df, strategy_groups)
            weighted_return = sum(
                weights[row["name"]] * row["months"][month_idx] for row in matrix_rows
            )
            evm_monthly_pcts.append(weighted_return)

    evm_compounded_factor = 1.0
    for val in evm_monthly_pcts:
        evm_compounded_factor *= 1.0 + val / 100.0
    evm_gesamt_pct = (evm_compounded_factor - 1.0) * 100.0

    # Calculate Risikoadjustierte Modelle (Max-Sharpe & Risk Parity) for Portfoliomodell category
    mv_ms_monthly_pcts: list[float] = []
    mv_rp_monthly_pcts: list[float] = []
    for month_idx in range(12):
        if month_idx == 0:
            mv_ms_monthly_pcts.append(portfolio_monthly_pcts[0])
            mv_rp_monthly_pcts.append(portfolio_monthly_pcts[0])
        else:
            prior_month_end = pd.Timestamp(
                year=selected_year, month=month_idx, day=1
            ) + pd.offsets.MonthEnd(0)
            slice_df = (
                dataframe[dataframe["exit_date_dt"] <= prior_month_end]
                if not dataframe.empty
                else pd.DataFrame()
            )
            ms_weights = _calculate_mean_variance_allocations(
                slice_df, strategy_groups, model_type="max_sharpe"
            )
            rp_weights = _calculate_mean_variance_allocations(
                slice_df, strategy_groups, model_type="risk_parity"
            )

            ms_weighted_return = sum(
                ms_weights[row["name"]] * row["months"][month_idx]
                for row in matrix_rows
            )
            rp_weighted_return = sum(
                rp_weights[row["name"]] * row["months"][month_idx]
                for row in matrix_rows
            )

            mv_ms_monthly_pcts.append(ms_weighted_return)
            mv_rp_monthly_pcts.append(rp_weighted_return)

    mv_ms_compounded = 1.0
    for val in mv_ms_monthly_pcts:
        mv_ms_compounded *= 1.0 + val / 100.0
    mv_ms_gesamt_pct = (mv_ms_compounded - 1.0) * 100.0

    mv_rp_compounded = 1.0
    for val in mv_rp_monthly_pcts:
        mv_rp_compounded *= 1.0 + val / 100.0
    mv_rp_gesamt_pct = (mv_rp_compounded - 1.0) * 100.0

    portfolio_models_rows = [
        {
            "name": "Standard",
            "months": portfolio_monthly_pcts,
            "gesamt": round(port_gesamt_pct, 1),
        },
        {
            "name": "Frequenz-Modell (EV/M)",
            "months": [round(p, 1) for p in evm_monthly_pcts],
            "gesamt": round(evm_gesamt_pct, 1),
        },
        {
            "name": "Risikoadjustiert (Max-Sharpe)",
            "months": [round(p, 1) for p in mv_ms_monthly_pcts],
            "gesamt": round(mv_ms_gesamt_pct, 1),
        },
        {
            "name": "Risikoadjustiert (Risk Parity)",
            "months": [round(p, 1) for p in mv_rp_monthly_pcts],
            "gesamt": round(mv_rp_gesamt_pct, 1),
        },
    ]

    benchmark_rows = []
    benchmark_symbols = [("SPY (S&P 500)", "SPY"), ("QQQ (Nasdaq 100)", "QQQ")]

    market_repo = service.market_repository
    for label, sym in benchmark_symbols:
        b_monthly: list[float] = []
        try:
            b_df = market_repo.get_symbol_history_raw(
                sym, start_date=f"{selected_year}-01-01"
            )
        except Exception:
            b_df = pd.DataFrame()

        if not b_df.empty:
            b_df["date_dt"] = pd.to_datetime(b_df["date"], errors="coerce")
            b_df = b_df[b_df["date_dt"].dt.year == selected_year].sort_values("date_dt")

            for month in range(1, 13):
                month_b_df = b_df[b_df["date_dt"].dt.month == month]
                if not month_b_df.empty:
                    open_p = float(
                        month_b_df.iloc[0]["open"] or month_b_df.iloc[0]["close"]
                    )
                    close_p = float(month_b_df.iloc[-1]["close"])
                    pct = ((close_p - open_p) / open_p * 100.0) if open_p > 0.0 else 0.0
                else:
                    pct = 0.0
                b_monthly.append(round(pct, 1))

            if not b_df.empty:
                year_open = float(b_df.iloc[0]["open"] or b_df.iloc[0]["close"])
                year_close = float(b_df.iloc[-1]["close"])
                b_gesamt = (
                    ((year_close - year_open) / year_open * 100.0)
                    if year_open > 0.0
                    else 0.0
                )
            else:
                b_gesamt = 0.0
        else:
            b_monthly = [0.0] * 12
            b_gesamt = 0.0

        benchmark_rows.append(
            {
                "name": label,
                "months": b_monthly,
                "gesamt": round(b_gesamt, 1),
            }
        )

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
