"""Routes and views for performance analytics and allocation dashboard."""

import pandas as pd
from flask import render_template

from ...const import Strategies, ExitReason
from ...types import TradeStatus
from ...tools import metrics
from .blueprint import views_bp
from .dependencies import (
    _get_trade_view_service,
    cache,
)


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
        return render_template(
            "analytics.html",
            summary=empty_summary,
            strategies=[],
            weekly_trend={},
            weekly_pnl={},
        )

    if not closed_trades:
        dataframe = pd.DataFrame(columns=["exit_date", "realized_pnl", "strategy"])
    else:
        dataframe = pd.DataFrame(closed_trades)
    dataframe["exit_date_dt"] = pd.to_datetime(dataframe["exit_date"])
    dataframe = dataframe.sort_values("exit_date_dt")

    # 2. Summary Metrics (Using metrics.py)
    # Portfolio equity curve for drawdown calculation
    initial_capital = 100000.0
    dataframe["cum_pnl"] = dataframe["realized_pnl"].cumsum()
    dataframe["equity"] = initial_capital + dataframe["cum_pnl"]

    summary = {
        "net_pnl": float(dataframe["realized_pnl"].sum()),
        "win_rate": metrics.calculate_win_rate(dataframe["realized_pnl"]),
        "max_drawdown": metrics.calculate_max_drawdown(dataframe["equity"]),
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

        # Vectorized Risk Calculations
        average_risk_dollar = 100.0
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

        average_risk_percent = (average_risk_dollar / initial_capital) * 100.0
        rate_of_return_percent = (
            (float(strategy_dataframe["realized_pnl"].sum()) / initial_capital) * 100.0
            if not strategy_dataframe.empty
            else 0.0
        )
        expectancy_r = (
            strategy_expectancy / average_risk_dollar
            if average_risk_dollar > 0.0
            else 0.0
        )

        # Combine realized P&L of closed trades with unrealized P&L of active trades
        closed_profit_and_loss_list = (
            strategy_dataframe["realized_pnl"].tolist()
            if not strategy_dataframe.empty
            else []
        )
        active_profit_and_loss_list = [
            float(trade.get("unrealized_pnl") or 0.0) for trade in strategy_active
        ]
        total_performance_series = pd.Series(
            closed_profit_and_loss_list + active_profit_and_loss_list
        )

        # Calculate Kelly Criterion Allocation Metrics using total performance
        strategy_win_rate = (
            metrics.calculate_win_rate(total_performance_series)
            if not total_performance_series.empty
            else 0.0
        )
        strategy_risk_reward_ratio = (
            metrics.calculate_risk_reward_ratio(total_performance_series)
            if not total_performance_series.empty
            else 0.0
        )
        strategy_kelly_criterion = (
            metrics.calculate_kelly_criterion(
                strategy_win_rate, strategy_risk_reward_ratio
            )
            if not total_performance_series.empty
            else 0.0
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
                    "avg_risk": f"{average_risk_percent:.2f}%",
                    "win_count": len(winning_trades),
                    "loss_count": len(losing_trades),
                    "avg_win": float(winning_trades["realized_pnl"].mean())
                    if not winning_trades.empty
                    else 0.0,
                    "avg_loss": float(losing_trades["realized_pnl"].mean())
                    if not losing_trades.empty
                    else 0.0,
                    "profit_factor": metrics.calculate_profit_factor(
                        strategy_dataframe["realized_pnl"]
                    )
                    if not strategy_dataframe.empty
                    else 0.0,
                    "expectancy": f"{expectancy_r:.2f} R",
                    "ror": f"{rate_of_return_percent:.2f}%",
                    "sharpe": metrics.calculate_sharpe_ratio(
                        strategy_dataframe["realized_pnl"], initial_capital
                    )
                    if not strategy_dataframe.empty
                    else 0.0,
                    "win_rate": strategy_win_rate,
                    "risk_reward_ratio": strategy_risk_reward_ratio,
                    "kelly_criterion": strategy_kelly_criterion,
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

    # Apply scaling multiplier to compute professional Suggested Allocations
    for strategy_item in strategies_data:
        raw_kelly = float(strategy_item["metrics"]["kelly_criterion"])
        suggested_allocation = raw_kelly * depot_multiplier if raw_kelly > 0.0 else 0.0
        strategy_item["metrics"]["suggested_allocation"] = suggested_allocation

    # Sort strategies by realized pnl desc
    strategies_data.sort(key=lambda x: x["pnl"], reverse=True)

    # 4. Weekly Trend Data (Plotly) - Since 01.01.2026
    start_of_year = pd.Timestamp("2026-01-01")
    today = pd.Timestamp.now()

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
        active_page="analytics",
    )
