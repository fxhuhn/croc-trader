"""Core mathematical metrics for trading performance analysis.

This module provides pure, vectorized functions for calculating standard
trading metrics using pandas and numpy. It follows the Functional Core
principle, ensuring referential transparency and ease of testing.
"""

import math

import numpy as np
import pandas as pd

# Constants for maintainability
EPSILON: float = 1e-6
MIN_SAMPLE_SIZE: int = 2


def calculate_win_rate(trades_pnl: pd.Series) -> float:
    """Calculates the percentage of winning trades.

    Formula: Win Rate = Winning Trades / Total Trades

    Source:
        Tharp, Van K. *Trade Your Way to Financial Freedom*,
        2nd ed., McGraw-Hill, 2007, Ch. 7.

    Args:
        trades_pnl: Series of realized profit and loss per trade.

    Returns:
        float: Win rate as a decimal (0.0 to 1.0).
    """
    if trades_pnl.empty:
        return 0.0

    winning_trades_count = (trades_pnl > EPSILON).sum()
    return float(winning_trades_count / len(trades_pnl))


def calculate_profit_factor(trades_pnl: pd.Series) -> float:
    """Calculates the gross profit divided by gross loss.

    Formula: Profit Factor = Σ(Profits) / |Σ(Losses)|

    Source:
        Kaufman, Perry J. *Trading Systems and Methods*,
        6th ed., Wiley, 2020, Ch. 21.

    Args:
        trades_pnl: Series of realized profit and loss per trade.

    Returns:
        float: Profit factor (returns math.inf if no losses).
    """
    gross_profit = trades_pnl[trades_pnl > EPSILON].sum()
    gross_loss = abs(trades_pnl[trades_pnl < -EPSILON].sum())

    if gross_loss < EPSILON:
        return math.inf if gross_profit > EPSILON else 0.0

    return float(gross_profit / gross_loss)


def calculate_expectancy(trades_pnl: pd.Series) -> float:
    """Calculates the average profit/loss per trade.

    Formula: E[X] = Σ(xᵢ) / n ≡ (W% × AvgW) − (L% × AvgL)

    Source:
        Tharp, Van K. *Trade Your Way to Financial Freedom*,
        2nd ed., McGraw-Hill, 2007, Ch. 7.

    Args:
        trades_pnl: Series of realized profit and loss per trade.

    Returns:
        float: Arithmetical mean of all trades.
    """
    if trades_pnl.empty:
        return 0.0
    return float(trades_pnl.mean())


def calculate_risk_reward_ratio(trades_pnl: pd.Series) -> float:
    """Calculates the average win divided by the average loss.

    Formula: Payoff Ratio = Mean(Wins) / |Mean(Losses)|

    Source:
        Tharp, Van K. *Trade Your Way to Financial Freedom*,
        2nd ed., McGraw-Hill, 2007, Ch. 7.

    Args:
        trades_pnl: Series of realized profit and loss per trade.

    Returns:
        float: Reward-to-Risk ratio.
    """
    winning_trades = trades_pnl[trades_pnl > EPSILON]
    losing_trades = trades_pnl[trades_pnl < -EPSILON]

    if winning_trades.empty:
        return 0.0

    if losing_trades.empty:
        return math.inf  # Return infinity representation if there are no losses

    average_win = winning_trades.mean()
    average_loss = abs(losing_trades.mean())

    return float(average_win / average_loss) if average_loss > EPSILON else 0.0


def calculate_drawdown_series(
    equity_curve: pd.Series, initial_value: float | None = None
) -> pd.Series:
    """Calculates the percentage peak-to-trough drawdown time series.

    Formula: DD_t = (Equity_t − CumMax_t) / CumMax_t

    Source:
        Magdon-Ismail, M. & Atiya, A.F. *"Maximum Drawdown"*,
        Risk Magazine, Oct 2004.

    Args:
        equity_curve: Series of portfolio equity values over time.
        initial_value: Optional initial capital. If provided, prepended to the series
                       to ensure the initial peak is correctly registered.

    Returns:
        pd.Series: Series of drawdown percentages as decimals (0.0 to -1.0).
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)

    if initial_value is not None:
        equity_curve = pd.concat(
            [pd.Series([initial_value]), equity_curve], ignore_index=True
        )

    running_maximum = equity_curve.cummax()
    valid_max = running_maximum.replace(0.0, np.nan)
    drawdown = (equity_curve - running_maximum) / valid_max
    return drawdown.fillna(0.0)


def calculate_max_drawdown(
    equity_curve: pd.Series, initial_value: float | None = None
) -> float:
    """Calculates the maximum peak-to-trough decline.

    Formula: MDD = Min((Equity − CumMax) / CumMax)

    Source:
        Magdon-Ismail, M. & Atiya, A.F. *"Maximum Drawdown"*,
        Risk Magazine, Oct 2004.

    Args:
        equity_curve: Series of portfolio equity values over time.
        initial_value: Optional initial capital. If provided, prepended to the series
                       to ensure the initial peak is correctly registered.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g., -0.15 for 15%).
    """
    drawdown_series = calculate_drawdown_series(equity_curve, initial_value)
    if drawdown_series.empty:
        return 0.0
    return float(drawdown_series.min())


def calculate_sharpe_ratio(
    trades_pnl: pd.Series, initial_capital: float, annualization_factor: float = 252.0
) -> float:
    """Calculates the annualized Sharpe Ratio.

    Formula: SR = (Mean(R) / Std(R)) × √N, with Rƒ = 0 (implicit).

    Source:
        Sharpe, William F. *"The Sharpe Ratio"*,
        Journal of Portfolio Management, Fall 1994, pp. 49-58.

    Args:
        trades_pnl: Series of realized profit and loss per trade.
        initial_capital: Starting capital used for return calculation.
        annualization_factor: Factor to scale to annual (default 252).

    Returns:
        float: Annualized Sharpe Ratio.
    """
    if len(trades_pnl) < MIN_SAMPLE_SIZE or initial_capital < EPSILON:
        return 0.0

    returns = trades_pnl / initial_capital
    standard_deviation = returns.std(ddof=1)

    if standard_deviation < EPSILON:
        return 0.0

    return float((returns.mean() / standard_deviation) * np.sqrt(annualization_factor))


def calculate_return_on_invested(
    trades_pnl: pd.Series, invested_capital: pd.Series
) -> float:
    """Calculates the overall percentage return on total invested capital.

    Formula: Return % = (Σ(PnL_valid) / Σ(Invested_valid)) × 100

    Args:
        trades_pnl: Series of realized profit and loss per trade.
        invested_capital: Series of total invested capital (Entry Price × Size) per trade.

    Returns:
        float: Total percentage return on invested capital.
    """
    if trades_pnl.empty or invested_capital.empty:
        return 0.0

    clean_invested = pd.to_numeric(invested_capital, errors="coerce").fillna(0.0)
    clean_pnl = pd.to_numeric(trades_pnl, errors="coerce").fillna(0.0)

    valid_mask = clean_invested > EPSILON
    if not valid_mask.any():
        return 0.0

    total_invested = float(clean_invested[valid_mask].sum())
    if total_invested <= EPSILON:
        return 0.0

    total_pnl = float(clean_pnl[valid_mask].sum())
    return float((total_pnl / total_invested) * 100.0)


def calculate_sharpe_ratio_from_roi(
    roi_series: pd.Series, trades_per_year: float = 252.0
) -> float:
    """Calculates the annualized Sharpe Ratio from a series of trade ROI percentages.

    Formula: SR = (Mean(ROI) / Std(ROI)) × √N_trades_per_year

    Args:
        roi_series: Series of individual trade return-on-investment ratios.
        trades_per_year: Estimated or annual trade count scaling factor.

    Returns:
        float: Annualized Sharpe Ratio.
    """
    clean_roi = roi_series.dropna()
    if len(clean_roi) < MIN_SAMPLE_SIZE or trades_per_year <= EPSILON:
        return 0.0

    standard_deviation = float(clean_roi.std(ddof=1))
    if standard_deviation < EPSILON:
        return 0.0

    return float((clean_roi.mean() / standard_deviation) * np.sqrt(trades_per_year))


def calculate_sortino_ratio_from_roi(
    roi_series: pd.Series,
    trades_per_year: float = 252.0,
    target_return: float = 0.0,
) -> float:
    """Calculates the annualized Sortino Ratio from trade ROI and downside deviation.

    Formula: Sortino = ((Mean(ROI) − Target) / Downside_Deviation) × √N_trades_per_year

    Args:
        roi_series: Series of individual trade return-on-investment ratios.
        trades_per_year: Estimated or annual trade count scaling factor.
        target_return: Minimum acceptable return benchmark (default 0.0).

    Returns:
        float: Annualized Sortino Ratio.
    """
    clean_roi = roi_series.dropna()
    if len(clean_roi) < MIN_SAMPLE_SIZE or trades_per_year <= EPSILON:
        return 0.0

    underperformance = np.minimum(0.0, clean_roi.to_numpy() - target_return)
    downside_deviation = float(np.sqrt(np.mean(underperformance**2)))

    if downside_deviation < EPSILON:
        return 0.0

    mean_excess_return = float(clean_roi.mean() - target_return)
    return float((mean_excess_return / downside_deviation) * np.sqrt(trades_per_year))


def calculate_sqn(r_multiples: pd.Series) -> float:
    """Calculates the System Quality Number (SQN).

    Formula: SQN = (Mean(R) / Std(R)) × √n

    Source:
        Tharp, Van K. *Definitive Guide to Position Sizing*,
        IITM, 2008.

    Args:
        r_multiples: Series of R-Multiples (PnL / Initial Risk).

    Returns:
        float: SQN value.
    """
    total_trades = len(r_multiples)
    if total_trades < MIN_SAMPLE_SIZE:
        return 0.0

    mean_r = r_multiples.mean()
    standard_deviation_r = r_multiples.std(ddof=1)

    if standard_deviation_r < EPSILON:
        return 0.0

    return float((mean_r / standard_deviation_r) * np.sqrt(total_trades))


def calculate_kelly_criterion(win_rate: float, risk_reward_ratio: float) -> float:
    """Calculates the Kelly Criterion fraction.

    Formula: f* = W − (1 − W) / b, where b = avg_win / avg_loss.

    Source:
        Kelly, J.L. *"A New Interpretation of Information Rate"*,
        Bell System Technical Journal, Vol. 35(4), 1956, pp. 917-926.

    Args:
        win_rate: Probability of a winning trade (0.0 to 1.0).
        risk_reward_ratio: Average win / Average loss.

    Returns:
        float: Kelly fraction (clamped between 0.0 and 1.0).
    """
    loss_rate = 1.0 - win_rate
    if loss_rate < EPSILON:
        return 1.0

    if risk_reward_ratio < EPSILON:
        return 0.0

    kelly_fraction = win_rate - (loss_rate / risk_reward_ratio)

    return float(max(0.0, min(1.0, kelly_fraction)))
