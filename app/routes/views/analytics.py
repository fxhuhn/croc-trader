"""Routes and views for performance analytics and allocation dashboard."""

import logging

import numpy as np
import pandas as pd
from flask import render_template

from ...const import ExitReason, Strategies
from ...tools import metrics
from ...types import TradeStatus
from .blueprint import views_bp
from .dependencies import (
    _get_trade_view_service,
    cache,
)

logger = logging.getLogger(__name__)


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

        return render_template(
            "analytics.html",
            summary=empty_summary,
            strategies=[],
            weekly_trend={},
            weekly_pnl={},
            monthly_evm=empty_monthly_evm,
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
    }

    # Map colors to match the exact hex codes configured in the template
    strategy_colors = {
        "Croc Setup": "#10b981",  # Emerald-500
        "Dip Buyer": "#6366f1",  # Indigo-500
        "Turnover": "#f59e0b",  # Amber-500
        "Two Percent": "#a855f7",  # Purple-500
        "NDX Momentum": "#f43f5e",  # Rose-500
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

    # 4. Weekly Trend Data (Plotly) - Since 01.01.2026
    start_of_year = pd.Timestamp("2026-01-01")

    # Create a full weekly range to ensure no gaps at the start
    date_range = pd.date_range(start=start_of_year, end=today, freq="W-SUN")

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
            .resample("W-SUN")
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
            .resample("W-SUN")
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
                .resample("W-SUN")
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
                .resample("W-SUN")
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
        current_month_name=current_month_name,
        active_page="analytics",
    )
