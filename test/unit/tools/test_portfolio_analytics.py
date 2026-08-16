from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest

from app.const import Strategies
from app.tools.portfolio_analytics import (
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


@pytest.fixture
def sample_trade_dataframe() -> pd.DataFrame:
    """Fixture providing sample multi-strategy closed trades DataFrame."""
    return pd.DataFrame(
        [
            {
                "strategy": "croc_setup",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-15",
                "exit_date_dt": pd.Timestamp("2026-01-15"),
                "entry_price": 100.0,
                "initial_size": 10.0,
                "stop_loss": 95.0,
                "realized_pnl": 50.0,
            },
            {
                "strategy": "croc_setup",
                "entry_date": "2026-01-10",
                "exit_date": "2026-01-20",
                "exit_date_dt": pd.Timestamp("2026-01-20"),
                "entry_price": 105.0,
                "initial_size": 10.0,
                "stop_loss": 100.0,
                "realized_pnl": -20.0,
            },
            {
                "strategy": "dip_buyer",
                "entry_date": "2026-01-08",
                "exit_date": "2026-01-18",
                "exit_date_dt": pd.Timestamp("2026-01-18"),
                "entry_price": 50.0,
                "initial_size": 20.0,
                "stop_loss": 45.0,
                "realized_pnl": 80.0,
            },
            {
                "strategy": "turnover_timing",
                "entry_date": "2026-02-01",
                "exit_date": "2026-02-10",
                "exit_date_dt": pd.Timestamp("2026-02-10"),
                "entry_price": 200.0,
                "initial_size": 5.0,
                "stop_loss": 190.0,
                "realized_pnl": 100.0,
            },
        ]
    )


@pytest.fixture
def strategy_groups() -> dict[str, list[object]]:
    return {
        "Croc Setup": [Strategies.CrocSetup],
        "Dip Buyer": [Strategies.DipBuyer],
        "Turnover": [Strategies.TurnOverTiming],
    }


def test_calculate_unweighted_monthly_pct_empty_and_valid() -> None:
    assert calculate_unweighted_monthly_pct(pd.DataFrame()) == 0.0

    df = pd.DataFrame(
        {
            "realized_pnl": [50.0, -20.0],
            "entry_price": [100.0, 100.0],
            "initial_size": [10.0, 10.0],
        }
    )
    # Invested = 1000 each. ROI = 50/1000 = 5%, -20/1000 = -2%. Mean = 1.5%
    assert pytest.approx(calculate_unweighted_monthly_pct(df)) == 1.5


def test_calculate_active_months(sample_trade_dataframe: pd.DataFrame) -> None:
    assert calculate_active_months(pd.DataFrame()) == 1.0
    months = calculate_active_months(sample_trade_dataframe)
    assert months >= 1.0


def test_extract_roi_series(sample_trade_dataframe: pd.DataFrame) -> None:
    empty_res = extract_roi_series(pd.DataFrame())
    assert empty_res.empty

    roi = extract_roi_series(sample_trade_dataframe)
    assert len(roi) == 4
    # First trade: 50 / (100 * 10) = 0.05
    assert pytest.approx(roi.iloc[0]) == 0.05


def test_calculate_evm_allocations(
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list[object]]
) -> None:
    # Empty DataFrame fallback to equal weights
    empty_alloc = calculate_evm_allocations(pd.DataFrame(), strategy_groups)
    assert len(empty_alloc) == 3
    assert pytest.approx(sum(empty_alloc.values()), abs=1e-5) == 1.0

    # Populated allocation
    alloc = calculate_evm_allocations(sample_trade_dataframe, strategy_groups)
    assert pytest.approx(sum(alloc.values()), abs=1e-5) == 1.0
    assert all(val >= 0.0 for val in alloc.values())


def test_calculate_mean_variance_allocations(
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list[object]]
) -> None:
    empty_alloc = calculate_mean_variance_allocations(pd.DataFrame(), strategy_groups)
    assert pytest.approx(sum(empty_alloc.values()), abs=1e-5) == 1.0

    # Test Max Sharpe and Risk Parity
    ms_alloc = calculate_mean_variance_allocations(
        sample_trade_dataframe, strategy_groups, model_type="max_sharpe"
    )
    assert pytest.approx(sum(ms_alloc.values()), abs=1e-5) == 1.0

    rp_alloc = calculate_mean_variance_allocations(
        sample_trade_dataframe, strategy_groups, model_type="risk_parity"
    )
    assert pytest.approx(sum(rp_alloc.values()), abs=1e-5) == 1.0


def test_calculate_mean_variance_dashboard_data(
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list[object]]
) -> None:
    empty_data = calculate_mean_variance_dashboard_data(pd.DataFrame(), strategy_groups)
    assert empty_data["has_low_data"] is True
    assert len(empty_data["strategies"]) == 3

    data = calculate_mean_variance_dashboard_data(
        sample_trade_dataframe, strategy_groups
    )
    assert len(data["strategies"]) == 3
    assert "mu" in data["strategies"][0]
    assert "weight_max_sharpe" in data["strategies"][0]


def test_calculate_concurrent_exposure(sample_trade_dataframe: pd.DataFrame) -> None:
    max_c, p95 = calculate_concurrent_exposure(pd.DataFrame(), [])
    assert max_c == 0
    assert p95 == 0

    active = [{"entry_date": "2026-01-12"}]
    max_c, p95 = calculate_concurrent_exposure(sample_trade_dataframe, active)
    assert max_c >= 1
    assert p95 >= 0


def test_calculate_monthly_matrix_data(
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list[object]]
) -> None:
    matrix_rows, portfolio_row, model_rows = calculate_monthly_matrix_data(
        sample_trade_dataframe, selected_year=2026, strategy_groups=strategy_groups
    )
    assert len(matrix_rows) == 3
    assert portfolio_row["name"] == "Portfolio"
    assert len(portfolio_row["months"]) == 12
    assert len(model_rows) == 4


def test_calculate_strategy_risk_and_expectancy(
    sample_trade_dataframe: pd.DataFrame,
) -> None:
    """Verifies strategy risk percentage and expectancy calculation."""
    avg_risk_text, expectancy_text = calculate_strategy_risk_and_expectancy(
        pd.DataFrame(), initial_capital=100_000.0
    )
    assert avg_risk_text == "N/A"
    assert expectancy_text == "N/A"

    croc_trades = sample_trade_dataframe[
        sample_trade_dataframe["strategy"] == "croc_setup"
    ]
    risk_pct, exp_r = calculate_strategy_risk_and_expectancy(
        croc_trades, initial_capital=100_000.0
    )
    assert "%" in risk_pct
    assert "R" in exp_r


def test_calculate_kelly_metrics_and_win_loss_rois(
    sample_trade_dataframe: pd.DataFrame,
) -> None:
    """Verifies Kelly criterion, win rate, and win/loss ROI calculations."""
    croc_trades = sample_trade_dataframe[
        sample_trade_dataframe["strategy"] == "croc_setup"
    ]
    roi_series = extract_roi_series(croc_trades)
    win_rate, rrr, kelly = calculate_kelly_metrics(roi_series, [])
    assert 0.0 <= win_rate <= 100.0
    assert rrr >= 0.0

    win_roi, loss_roi = calculate_win_loss_rois(croc_trades)
    assert win_roi > 0.0
    assert loss_roi < 0.0

    empty_win, empty_loss = calculate_win_loss_rois(pd.DataFrame())
    assert empty_win == 0.0
    assert empty_loss == 0.0


def test_apply_depot_and_ev_allocations() -> None:
    """Verifies Kelly and EV allocation scaling across strategies."""
    strategies_data: list[dict[str, Any]] = [
        {
            "name": "Strategy A",
            "metrics": {
                "kelly_criterion": 0.8,
                "ev_per_month": 500.0,
            },
        },
        {
            "name": "Strategy B",
            "metrics": {
                "kelly_criterion": 0.6,
                "ev_per_month": 500.0,
            },
        },
    ]
    apply_depot_and_ev_allocations(strategies_data, initial_capital=100_000.0)

    # Total proposed = 1.4 > 1.0, should scale down
    metrics_a: dict[str, Any] = strategies_data[0]["metrics"]
    metrics_b: dict[str, Any] = strategies_data[1]["metrics"]
    alloc_a = float(metrics_a["suggested_allocation"])
    alloc_b = float(metrics_b["suggested_allocation"])
    assert pytest.approx(alloc_a + alloc_b, abs=1e-5) == 1.0
    assert float(metrics_a["ev_allocation"]) == 0.5
    assert float(metrics_a["ev_allocation_100k"]) == 50_000.0


def test_calculate_benchmark_monthly_returns() -> None:
    """Verifies benchmark returns calculation from price series."""
    empty_benchmark = calculate_benchmark_monthly_returns(pd.DataFrame(), "SPY", 2026)
    assert empty_benchmark["name"] == "SPY"
    assert len(empty_benchmark["months"]) == 12
    assert empty_benchmark["gesamt"] == 0.0

    prices_df = pd.DataFrame(
        [
            {"date": "2026-01-02", "open": 100.0, "close": 105.0},
            {"date": "2026-01-30", "open": 106.0, "close": 110.0},
            {"date": "2026-02-02", "open": 110.0, "close": 112.0},
            {"date": "2026-02-27", "open": 112.0, "close": 120.0},
        ]
    )
    res = calculate_benchmark_monthly_returns(prices_df, "SPY (S&P 500)", 2026)
    assert res["name"] == "SPY (S&P 500)"
    assert res["months"][0] == 10.0  # (110 - 100) / 100 = 10%
    assert res["months"][1] == pytest.approx(9.1, abs=0.1)  # (120 - 110) / 110
    assert res["gesamt"] == 20.0  # (120 - 100) / 100 = 20%
