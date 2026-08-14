"""Unit tests for app/tools/portfolio_analytics.py."""

import pandas as pd
import pytest

from app.const import Strategies
from app.tools.portfolio_analytics import (
    calculate_active_months,
    calculate_concurrent_exposure,
    calculate_evm_allocations,
    calculate_mean_variance_allocations,
    calculate_mean_variance_dashboard_data,
    calculate_monthly_matrix_data,
    calculate_unweighted_monthly_pct,
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
def strategy_groups() -> dict[str, list]:
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
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list]
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
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list]
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
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list]
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
    sample_trade_dataframe: pd.DataFrame, strategy_groups: dict[str, list]
) -> None:
    matrix_rows, portfolio_row, model_rows = calculate_monthly_matrix_data(
        sample_trade_dataframe, selected_year=2026, strategy_groups=strategy_groups
    )
    assert len(matrix_rows) == 3
    assert portfolio_row["name"] == "Portfolio"
    assert len(portfolio_row["months"]) == 12
    assert len(model_rows) == 4
