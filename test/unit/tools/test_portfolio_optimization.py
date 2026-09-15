"""Unit tests for the portfolio optimization mathematical engine."""

import numpy as np
import pandas as pd
import pytest

from app.tools.portfolio_optimization import (
    build_covariance_matrix,
    calculate_risk_contributions,
    compute_downside_deviation,
    optimize_max_sharpe_weights,
    optimize_risk_parity_weights,
)


def test_build_covariance_matrix_empty() -> None:
    """Verifies build_covariance_matrix behavior on empty input."""
    df_empty = pd.DataFrame()
    cov = build_covariance_matrix(df_empty)
    assert cov.shape == (0, 0)


def test_build_covariance_matrix_shrinkage() -> None:
    """Verifies covariance matrix symmetry and shrinkage when trades are few."""
    df_few = pd.DataFrame(
        {
            "StratA": [0.01, -0.02, 0.03],
            "StratB": [0.02, 0.01, -0.01],
        }
    )
    cov = build_covariance_matrix(df_few, shrinkage_threshold=5)
    assert cov.shape == (2, 2)
    # Check symmetry
    assert pytest.approx(cov[0, 1]) == cov[1, 0]
    # Positive diagonal
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0


def test_optimize_max_sharpe_weights() -> None:
    """Verifies Max-Sharpe weight optimization constraints: sum to 1.0, non-negative."""
    mu = np.array([0.15, 0.08, 0.05])
    cov = np.array(
        [
            [0.04, 0.005, 0.0],
            [0.005, 0.02, 0.001],
            [0.0, 0.001, 0.01],
        ]
    )

    weights = optimize_max_sharpe_weights(mu, cov)
    assert len(weights) == 3
    assert pytest.approx(np.sum(weights), abs=1e-5) == 1.0
    assert np.all(weights >= -1e-7)
    # Max Sharpe portfolio ratio must be >= individual strategy Sharpe ratios
    port_return = float(np.dot(weights, mu))
    port_vol = np.sqrt(float(np.dot(weights.T, np.dot(cov, weights))))
    port_sharpe = port_return / port_vol
    ind_sharpes = mu / np.sqrt(np.diag(cov))
    assert port_sharpe >= np.max(ind_sharpes) - 1e-5


def test_optimize_risk_parity_weights() -> None:
    """Verifies Risk Parity weight optimization constraints: sum to 1.0, non-negative, equal risk contributions."""
    cov = np.array(
        [
            [0.04, 0.0],
            [0.0, 0.01],
        ]
    )

    weights = optimize_risk_parity_weights(cov)
    assert len(weights) == 2
    assert pytest.approx(np.sum(weights), abs=1e-5) == 1.0
    assert np.all(weights >= 0.0)

    # Lower volatility strategy (Strat 2 with var 0.01) must receive HIGHER weight than Strat 1 (var 0.04)
    # to achieve equal risk contribution
    assert weights[1] > weights[0]

    _, trc, prc = calculate_risk_contributions(weights, cov)
    # Percentage risk contributions should be approx 50% each
    assert pytest.approx(prc[0], abs=1.0) == 50.0
    assert pytest.approx(prc[1], abs=1.0) == 50.0


def test_calculate_risk_contributions() -> None:
    """Verifies MCR, TRC, and PRC calculation consistency."""
    weights = np.array([0.6, 0.4])
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])

    mcr, trc, prc = calculate_risk_contributions(weights, cov)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    assert len(mcr) == 2
    assert len(trc) == 2
    assert pytest.approx(np.sum(trc), abs=1e-5) == port_vol
    assert pytest.approx(np.sum(prc), abs=1e-3) == 100.0


def test_compute_downside_deviation() -> None:
    """Verifies downside deviation calculation for negative returns only."""
    returns = pd.Series([0.05, -0.02, 0.03, -0.04, 0.01])
    d_dev = compute_downside_deviation(returns)
    assert d_dev > 0.0


def test_optimize_risk_parity_weights_multi_asset_exact_erc() -> None:
    """Verifies that the convex Spinu solver achieves equal risk contributions across heterogeneous assets."""
    cov = np.diag([0.09, 0.04, 0.01, 0.0025])
    weights = optimize_risk_parity_weights(cov)

    assert len(weights) == 4
    assert pytest.approx(np.sum(weights), abs=1e-5) == 1.0
    # Inverse volatility ordering: asset with 0.0025 var must have largest weight
    assert weights[3] > weights[2] > weights[1] > weights[0]

    _, _, prc = calculate_risk_contributions(weights, cov)
    for p in prc:
        assert pytest.approx(p, abs=0.1) == 25.0


def test_optimize_risk_parity_weights_edge_cases() -> None:
    """Verifies boundary handling for 0 and 1 asset inputs."""
    empty_cov = np.empty((0, 0))
    assert len(optimize_risk_parity_weights(empty_cov)) == 0

    single_cov = np.array([[0.04]])
    single_w = optimize_risk_parity_weights(single_cov)
    assert len(single_w) == 1
    assert pytest.approx(single_w[0]) == 1.0


def test_build_covariance_matrix_shrinkage_with_nans() -> None:
    """Verifies that sparse DataFrames containing NaNs correctly trigger Ledoit-Wolf shrinkage."""
    df_sparse = pd.DataFrame(
        {
            "StratA": [0.01, 0.02, -0.01, 0.03, 0.01, -0.02],
            "StratB": [
                np.nan,
                np.nan,
                np.nan,
                0.02,
                np.nan,
                0.01,
            ],  # Only 2 valid observations
        }
    )
    cov = build_covariance_matrix(df_sparse, shrinkage_threshold=5)
    assert cov.shape == (2, 2)
    assert cov[0, 0] > 0.0
    assert cov[1, 1] > 0.0
    assert pytest.approx(cov[0, 1]) == cov[1, 0]
