"""Core mathematical engine for portfolio optimization and risk contribution analysis.

This module provides pure vectorized functions for calculating covariance matrices,
Max-Sharpe and Risk Parity (Equal Risk Contribution) portfolio weights, Marginal and
Total Risk Contributions (MCR, TRC), and Monte-Carlo resampled efficiency.

It follows the Functional Core principle: referentially transparent calculations
without side effects or I/O.
"""

from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPSILON: float = 1e-6

MIN_RETURN_SERIES_LEN: int = 2


def build_covariance_matrix(
    returns_df: pd.DataFrame,
    shrinkage_threshold: int = 5,
) -> np.ndarray:
    """Builds an N x N covariance matrix from strategy return series.

    Applies Ledoit-Wolf-style shrinkage toward an equal-variance target if any strategy
    has fewer than the specified threshold of active return observations.

    Args:
        returns_df: DataFrame where each column is a strategy return series.
        shrinkage_threshold: Minimum observation threshold before applying shrinkage.

    Returns:
        np.ndarray: N x N symmetric positive semi-definite covariance matrix.
    """
    num_strats = returns_df.shape[1]
    if num_strats == 0:
        return np.empty((0, 0))

    if returns_df.empty or len(returns_df) < MIN_RETURN_SERIES_LEN:
        return np.eye(num_strats) * 1e-4

    # Fill NaNs with 0 (no return on periods without trades)
    clean_returns = returns_df.fillna(0.0).to_numpy()
    sample_cov = np.cov(clean_returns, rowvar=False)

    # Ensure 2D matrix even for single strategy
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]])

    # Check if shrinkage is required (count non-NaN, non-zero active observations)
    valid_obs_count = int(
        (returns_df.notna() & (returns_df != 0.0)).astype(int).sum(axis=0).min()
    )
    if valid_obs_count < shrinkage_threshold:
        # Target: Diagonal matrix with average sample variance
        avg_var = float(np.mean(np.diag(sample_cov)))
        target = np.eye(num_strats) * max(avg_var, 1e-4)
        shrinkage_intensity = 0.3
        cov_matrix = (
            1.0 - shrinkage_intensity
        ) * sample_cov + shrinkage_intensity * target
    else:
        cov_matrix = sample_cov

    # Add small regularization diagonal jitter for numerical stability
    cov_matrix += np.eye(num_strats) * 1e-8
    return cov_matrix


def calculate_risk_contributions(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates Marginal Risk Contribution (MCR), Total Risk Contribution (TRC), and Percentage Risk Contribution (PRC).

    Formulas:
        Portfolio Volatility: σ_p = √(wᵀ Σ w)
        MCR_i = (Σ w)_i / σ_p
        TRC_i = w_i * MCR_i
        PRC_i = TRC_i / σ_p

    Args:
        weights: Array of portfolio strategy weights (sum to 1.0).
        cov_matrix: N x N covariance matrix Σ.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (MCR array, TRC array, PRC array).
    """
    num_strats = len(weights)
    if num_strats == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    portfolio_variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_volatility = np.sqrt(max(portfolio_variance, EPSILON))

    cov_times_w = np.dot(cov_matrix, weights)
    mcr = cov_times_w / portfolio_volatility
    trc = weights * mcr

    if portfolio_volatility > EPSILON:
        prc = (trc / portfolio_volatility) * 100.0
    else:
        prc = np.ones(num_strats) * (100.0 / num_strats)

    return mcr, trc, prc


def optimize_max_sharpe_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Computes Max-Sharpe portfolio weights using Sequential Least Squares Programming (SLSQP).

    Objective:
        Maximize (wᵀ μ - R_f) / √(wᵀ Σ w)

    Constraints:
        Σ w_i = 1.0
        w_i >= 0.0  (No-Short)

    Args:
        expected_returns: Vector of expected strategy returns μ.
        cov_matrix: N x N covariance matrix Σ.
        risk_free_rate: Risk-free rate.

    Returns:
        np.ndarray: Vector of optimal weights w*.
    """
    num_strats = len(expected_returns)
    if num_strats == 0:
        return np.array([], dtype=float)

    default_weights: np.ndarray = np.ones(num_strats) / float(num_strats)

    if np.all(expected_returns <= EPSILON):
        return default_weights

    def negative_sharpe_objective(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, expected_returns)) - risk_free_rate
        port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_vol = float(np.sqrt(max(port_var, EPSILON)))
        return float(-port_return / port_vol)

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0) for _ in range(num_strats)]

    result = minimize(
        negative_sharpe_objective,
        x0=default_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        return default_weights

    optimized_weights: np.ndarray = np.maximum(0.0, result.x)
    total_w = float(np.sum(optimized_weights))
    if total_w > EPSILON:
        return cast(np.ndarray, optimized_weights / total_w)

    return default_weights


def optimize_risk_parity_weights(
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """Computes Equal Risk Contribution (Risk Parity) portfolio weights.

    Uses the strictly convex log-barrier formulation of Maillard, Roncalli & Teïletche (2010)
    and Spinu (2013):
        min_y ( 1/2 * yᵀ Σ y - Σ ln(y_i) )  subject to y > 0
        w* = y* / (Σ y_i*)

    This formulation guarantees strict convexity, a unique global minimum, and eliminates
    the local minima and vanishing gradient stalls of the non-convex pairwise difference formulation.

    Args:
        cov_matrix: N x N symmetric positive semi-definite covariance matrix Σ.

    Returns:
        np.ndarray: Vector of optimal Risk Parity weights w*.
    """
    num_strats = cov_matrix.shape[0]
    if num_strats == 0:
        return np.array([], dtype=float)
    if num_strats == 1:
        return np.array([1.0], dtype=float)

    default_weights: np.ndarray = np.ones(num_strats) / float(num_strats)

    # Initial guess inversely proportional to volatility (naive risk parity)
    variances = np.diag(cov_matrix)
    stds = np.sqrt(np.maximum(variances, 1e-8))
    inv_vols = 1.0 / stds
    y0 = inv_vols / np.sum(inv_vols)

    def log_barrier_objective(y: np.ndarray) -> float:
        safe_y = np.maximum(y, 1e-12)
        quad = float(np.dot(safe_y.T, np.dot(cov_matrix, safe_y)))
        log_barrier = float(np.sum(np.log(safe_y)))
        return 0.5 * quad - log_barrier

    def log_barrier_gradient(y: np.ndarray) -> np.ndarray:
        safe_y = np.maximum(y, 1e-12)
        return cast(np.ndarray, np.dot(cov_matrix, safe_y) - (1.0 / safe_y))

    bounds = [(1e-8, None) for _ in range(num_strats)]

    result = minimize(
        log_barrier_objective,
        x0=y0,
        jac=log_barrier_gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )

    if not result.success or np.any(result.x <= 0.0):
        return default_weights

    optimized_weights: np.ndarray = np.maximum(0.0, result.x)
    total_weight = float(np.sum(optimized_weights))
    if total_weight > EPSILON:
        return cast(np.ndarray, optimized_weights / total_weight)

    return default_weights


def compute_downside_deviation(
    returns_series: pd.Series,
    target_return: float = 0.0,
) -> float:
    """Calculates downside deviation (semi-deviation) for Sortino ratio calculation.

    Args:
        returns_series: Series of returns.
        target_return: Minimum acceptable return benchmark.

    Returns:
        float: Downside deviation.
    """
    if returns_series.empty:
        return 0.0

    underperformance = np.minimum(0.0, returns_series.to_numpy() - target_return)
    squared_downside = underperformance**2
    return float(np.sqrt(np.mean(squared_downside)))
