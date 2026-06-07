"""Core mathematical metrics for trading performance analysis.

This module provides pure, vectorized functions for calculating standard
trading metrics using pandas and numpy. It follows the Functional Core
principle, ensuring referential transparency and ease of testing.
"""

import numpy as np
import pandas as pd

# Constants for maintainability
EPSILON: float = 1e-6


def calculate_win_rate(trades_pnl: pd.Series) -> float:
    """Calculates the percentage of winning trades.

    Formula: (Winning Trades / Total Trades)

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

    Formula: Sum of Profits / Abs(Sum of Losses)

    Args:
        trades_pnl: Series of realized profit and loss per trade.

    Returns:
        float: Profit factor (returns 999.0 if no losses).
    """
    gross_profit = trades_pnl[trades_pnl > EPSILON].sum()
    gross_loss = abs(trades_pnl[trades_pnl < -EPSILON].sum())

    if gross_loss < EPSILON:
        return 999.0 if gross_profit > EPSILON else 0.0

    return float(gross_profit / gross_loss)


def calculate_expectancy(trades_pnl: pd.Series) -> float:
    """Calculates the average profit/loss per trade.

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
        return 999.0  # Return infinity representation if there are no losses

    average_win = winning_trades.mean()
    average_loss = abs(losing_trades.mean())

    return float(average_win / average_loss) if average_loss > EPSILON else 0.0


def calculate_max_drawdown(equity_curve: pd.Series, initial_value: float | None = None) -> float:
    """Calculates the maximum peak-to-trough decline.

    Args:
        equity_curve: Series of portfolio equity values over time.
        initial_value: Optional initial capital. If provided, prepended to the series
                       to ensure the initial peak is correctly registered.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g., -0.15 for 15%).
    """
    if equity_curve.empty:
        return 0.0

    if initial_value is not None:
        equity_curve = pd.concat([pd.Series([initial_value]), equity_curve], ignore_index=True)

    running_maximum = equity_curve.cummax()
    drawdown = (equity_curve - running_maximum) / running_maximum

    return float(drawdown.min())


def calculate_sharpe_ratio(
    trades_pnl: pd.Series, initial_capital: float, annualization_factor: float = 252.0
) -> float:
    """Calculates the annualized Sharpe Ratio.

    Args:
        trades_pnl: Series of realized profit and loss per trade.
        initial_capital: Starting capital used for return calculation.
        annualization_factor: Factor to scale to annual (default 252).

    Returns:
        float: Annualized Sharpe Ratio.
    """
    if len(trades_pnl) < 2 or initial_capital < EPSILON:
        return 0.0

    returns = trades_pnl / initial_capital
    standard_deviation = returns.std(ddof=1)

    if standard_deviation < EPSILON:
        return 0.0

    return float((returns.mean() / standard_deviation) * np.sqrt(annualization_factor))


def calculate_sqn(r_multiples: pd.Series) -> float:
    """Calculates the System Quality Number (SQN).

    Formula: (Mean R / Std R) * Sqrt(Total Trades)

    Args:
        r_multiples: Series of R-Multiples (PnL / Initial Risk).

    Returns:
        float: SQN value.
    """
    total_trades = len(r_multiples)
    if total_trades < 2:
        return 0.0

    mean_r = r_multiples.mean()
    standard_deviation_r = r_multiples.std(ddof=1)

    if standard_deviation_r < EPSILON:
        return 0.0

    return float((mean_r / standard_deviation_r) * np.sqrt(total_trades))


def calculate_kelly_criterion(win_rate: float, risk_reward_ratio: float) -> float:
    """Calculates the Kelly Criterion fraction.

    Formula: WinRate - (LossRate / RewardToRisk)

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


def calculate_ulcer_index(equity_curve: pd.Series) -> float:
    """Calculates the Ulcer Index (Pain Index).

    Measures the depth and duration of drawdowns.

    Args:
        equity_curve: Series of portfolio equity values over time.

    Returns:
        float: Ulcer Index value.
    """
    if equity_curve.empty:
        return 0.0

    running_maximum = equity_curve.cummax()
    drawdown_percentage = ((equity_curve - running_maximum) / running_maximum) * 100.0

    squared_drawdown = drawdown_percentage**2
    return float(np.sqrt(squared_drawdown.mean()))
