"""Unit tests for app/services/backtester/runner.py display and utility functions."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from rich.console import Console

from app.services.backtester.analytics import (
    BacktestMetrics,
    PortfolioMetrics,
)
from app.services.backtester.runner import (
    _display_allocation_comparison,
    _display_audit_reports,
    _display_capacity_ratio_analysis,
    _display_capacity_simulation,
    _display_diversification,
    _display_impact_analysis,
    _display_performance_table,
    _display_period_analysis,
    _display_portfolio_funnel,
    _display_portfolio_kelly,
    _display_quality_distribution,
    _display_regime_comparison,
    _display_safety_switch_steps,
    _display_strategy_breakdown,
    _display_switch_tournament,
    _display_walk_forward_insights,
    _export_portfolio_configuration,
    _load_portfolio_configuration,
)


def _make_dummy_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        total_trades=10,
        win_rate=0.6,
        profit_factor=1.8,
        net_profit=5000.0,
        expectancy=500.0,
        kelly_mean=0.2,
        kelly_safe=0.15,
        system_quality_number=2.5,
        maximum_drawdown=0.1,
        strategy_return=0.25,
        market_exposure_pct=0.5,
        risk_adjusted_benchmark=0.12,
        exposure_efficiency=1.2,
        return_over_maximum_drawdown=2.5,
        sharpe_ratio=1.5,
        kelly_criterion=0.2,
        average_win=100.0,
        average_loss=-50.0,
        average_maximum_adverse_excursion=-30.0,
        average_maximum_favorable_excursion=120.0,
        risk_of_ruin=0.01,
        benchmark_return=0.15,
        kelly_std=0.05,
        diversification_score=75.0,
    )


def _make_dummy_portfolio_metrics() -> PortfolioMetrics:
    return PortfolioMetrics(
        combined_mean_kelly=0.2,
        safe_kelly_25=0.15,
        max_concurrent_trades=5,
        max_concurrent_trades_days=10,
        percentile_95_concurrent_trades=4.0,
        suggested_multiplier=1.2,
        uncapped_multiplier=1.5,
        max_total_exposure=80.0,
        uncapped_max_total_exposure=120.0,
        leveraged_max_drawdown=0.15,
        uncapped_leveraged_max_drawdown=0.25,
        correlation_fail_rate=0.0,
        strategy_capacity_ratios={"DipBuyer": 1.5},
        max_trades_per_strategy={"DipBuyer": 3},
        max_trades_per_strategy_days={"DipBuyer": 5},
        percentile_95_trades_per_strategy={"DipBuyer": 2.5},
    )


def test_display_performance_table() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    metrics = _make_dummy_metrics()

    _display_performance_table(console, metrics, 0.15)
    output = buf.getvalue()
    assert "Strategy Performance" in output
    assert "Win Rate" in output


def test_display_strategy_breakdown() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    metrics = _make_dummy_metrics()
    mapping = {"DipBuyer": metrics}

    _display_strategy_breakdown(console, mapping)
    output = buf.getvalue()
    assert "Strategy Breakdown" in output

    # Empty mapping path
    _display_strategy_breakdown(console, {})


def test_display_safety_switch_steps() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    events = [
        {
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            "reason": "Drawdown limit",
            "days": 9,
            "saved_profit": 1500.0,
        },
        {
            "start_date": "2026-02-01",
            "end_date": None,
            "reason": "Market stress",
            "days": 5,
            "saved_profit": 500.0,
        },
    ]

    _display_safety_switch_steps(console, events)  # type: ignore[arg-type]
    output = buf.getvalue()
    assert "Safety Switch Events" in output

    _display_safety_switch_steps(console, [])


def test_display_impact_analysis() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    simulation = {
        "saved_loss": 3000.0,
        "opportunity_cost": 1000.0,
        "margin_interest_paid": 50.0,
        "net_efficiency": 1950.0,
    }

    _display_impact_analysis(console, simulation)  # type: ignore[arg-type]
    output = buf.getvalue()
    assert "Impact Analysis" in output


def test_display_switch_tournament() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    results = [
        {
            "logic": "AlwaysOn",
            "net_efficiency": 500.0,
            "total_return": 2000.0,
            "max_drawdown": 0.1,
            "grade": "A",
        },
        {
            "logic": "SafetySwitch",
            "net_efficiency": -100.0,
            "total_return": 1500.0,
            "max_drawdown": 0.05,
            "grade": "B",
        },
    ]

    _display_switch_tournament(console, results)
    output = buf.getvalue()
    assert "Safety Logic Comparison" in output
    _display_switch_tournament(console, [])


def test_display_regime_comparison() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    regimes: dict[str, dict[str, object]] = {
        "DipBuyer": {
            "BULL": {"Return": 2000.0, "Sample_Count": 10.0},
            "BEAR": {"Return": -500.0, "Sample_Count": 5.0},
        }
    }

    _display_regime_comparison(console, regimes)
    output = buf.getvalue()
    assert "Regime-Specific Performance" in output
    _display_regime_comparison(console, {})


def test_display_walk_forward_insights() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    df = pd.DataFrame(
        [
            {
                "Window": "W1",
                "Degradation": 0.05,
                "OOS_PF": 1.8,
                "Recommendation": "STABLE",
            },
            {
                "Window": "W2",
                "Degradation": 0.25,
                "OOS_PF": 1.1,
                "Recommendation": "WARNING",
            },
            {
                "Window": "W3",
                "Degradation": 0.50,
                "OOS_PF": 0.8,
                "Recommendation": "CRITICAL",
            },
        ]
    )

    _display_walk_forward_insights(console, df)
    output = buf.getvalue()
    assert "WFA Actionable Insights" in output
    _display_walk_forward_insights(console, pd.DataFrame())


def test_display_portfolio_kelly() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    p_metrics = _make_dummy_portfolio_metrics()

    _display_portfolio_kelly(console, p_metrics)
    output = buf.getvalue()
    assert "Portfolio Optimization" in output


def test_display_period_analysis() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    df = pd.DataFrame(
        [
            {
                "window_trades": 20,
                "sharpe_proxy": 1.5,
                "max_drawdown_pnl": -500.0,
                "win_rate": 0.6,
            }
        ]
    )

    _display_period_analysis(console, df)
    output = buf.getvalue()
    assert "Rolling Performance Windows" in output
    _display_period_analysis(console, pd.DataFrame())


def test_display_quality_distribution() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    df = pd.DataFrame(
        [
            {"grade": "A", "weakest_link": "RSI", "profit": 1500.0},
            {"grade": "B", "weakest_link": "ATR", "profit": 500.0},
        ]
    )

    _display_quality_distribution(console, df)
    output = buf.getvalue()
    assert "Trade Quality Distribution" in output
    _display_quality_distribution(console, pd.DataFrame())


def test_display_diversification() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    matrices = {
        "correlation": pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], columns=["A", "B"])
    }

    _display_diversification(console, 75.0, matrices)
    output = buf.getvalue()
    assert "Diversification Analysis" in output


def test_display_portfolio_funnel() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    funnel_data = [
        {
            "name": "DipBuyer",
            "kelly": 0.2,
            "raw_share": 25.0,
            "status": "ACTIVE",
            "reason": None,
            "final_allocation": 25.0,
        },
        {
            "name": "TGIM",
            "kelly": 0.0,
            "raw_share": 0.0,
            "status": "INACTIVE",
            "reason": "Low Expectancy",
            "final_allocation": 0.0,
        },
    ]

    _display_portfolio_funnel(console, funnel_data)
    output = buf.getvalue()
    assert "Portfolio Allocation (Funnel Logic)" in output
    _display_portfolio_funnel(console, [])


def test_display_capacity_ratio_analysis() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    p_metrics = _make_dummy_portfolio_metrics()

    _display_capacity_ratio_analysis(console, p_metrics)
    output = buf.getvalue()
    assert "Per-Strategy Capacity Ratio Analysis" in output


def test_display_allocation_comparison() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    funnel_data = [
        {
            "name": "DipBuyer",
            "final_allocation": 25.0,
            "final_allocation_p95": 30.0,
            "status": "ACTIVE",
            "reason": None,
        },
        {
            "name": "BounceBandit",
            "final_allocation": 20.0,
            "final_allocation_p95": 15.0,
            "status": "ACTIVE",
            "reason": None,
        },
    ]

    _display_allocation_comparison(console, funnel_data)
    output = buf.getvalue()
    assert "ALLOCATION COMPARISON" in output
    _display_allocation_comparison(console, [])


def test_display_audit_reports() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    analytics = MagicMock()
    analytics.run_constrained_kelly_simulation.return_value = {
        "multipliers": {"DipBuyer": 1.2},
        "theoretical_equity": 100000.0,
    }
    analytics.run_regime_comparison.return_value = {}
    analytics.run_safety_tournament.return_value = []

    _display_audit_reports(
        console,
        analytics=analytics,
        strategy_performance_map={"DipBuyer": _make_dummy_metrics()},
        safety_simulation={
            "multipliers": {"DipBuyer": 1.1},
            "final_equity": 105000.0,
            "saved_loss": 3000.0,
            "opportunity_cost": 1000.0,
            "net_efficiency": 2000.0,
            "events": [],
        },
        regime_dataframe=pd.DataFrame(),
    )
    output = buf.getvalue()
    assert "Regime-Specific Performance Audit" in output
    output = buf.getvalue()
    assert "Regime-Specific Performance Audit" in output


def test_display_capacity_simulation() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)

    sim = {
        "capacity_limit": 500000.0,
        "slippage_impact_pct": 0.05,
    }

    _display_capacity_simulation(console, sim)
    output = buf.getvalue()
    assert "Capacity Simulation" in output


def test_load_and_export_portfolio_configuration(tmp_path: Path) -> None:
    p_metrics = _make_dummy_portfolio_metrics()
    funnel_data = [
        {
            "name": "dip_buyer",
            "final_allocation": 25.0,
            "final_allocation_p95": 20.0,
            "status": "ACTIVE",
        }
    ]

    from unittest.mock import mock_open

    m = mock_open()
    with patch("builtins.open", m):
        _export_portfolio_configuration(
            funnel_data=funnel_data,
            portfolio_metrics=p_metrics,
            equity=100000.0,
        )

    loaded = _load_portfolio_configuration()
    assert loaded is not None
