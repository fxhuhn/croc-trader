import math

import pandas as pd
import pytest

from app.tools import metrics


def test_calculate_win_rate_basic():
    """Tests win rate calculation with positive and negative net pnl."""
    # Arrange
    net_pnl = pd.Series([100.0, -50.0, 200.0, 0.0, -100.0])

    # Act
    win_rate = metrics.calculate_win_rate(net_pnl)

    # Assert
    # 2 wins (100, 200) out of 5 trades = 0.4
    assert win_rate == pytest.approx(0.4)


def test_calculate_win_rate_empty():
    """Tests win rate calculation for an empty series."""
    # Arrange
    net_pnl = pd.Series([], dtype=float)

    # Act
    win_rate = metrics.calculate_win_rate(net_pnl)

    # Assert
    assert win_rate == 0.0


def test_calculate_profit_factor():
    """Tests profit factor calculation for normal trades."""
    # Arrange
    net_pnl = pd.Series([100.0, -50.0, 200.0, -100.0])

    # Act
    profit_factor = metrics.calculate_profit_factor(net_pnl)

    # Assert
    # Gross Profit = 300, Gross Loss = 150 -> 300 / 150 = 2.0
    assert profit_factor == pytest.approx(2.0)


def test_calculate_profit_factor_no_losses():
    """Tests profit factor when there are no losses."""
    # Arrange
    net_pnl = pd.Series([100.0, 200.0])

    # Act
    profit_factor = metrics.calculate_profit_factor(net_pnl)

    # Assert
    assert math.isinf(profit_factor)


def test_calculate_max_drawdown():
    """Tests maximum drawdown calculation for an equity curve."""
    # Arrange
    equity = pd.Series([100.0, 110.0, 90.0, 120.0, 80.0])

    # Act
    max_dd = metrics.calculate_max_drawdown(equity)

    # Assert
    # Peaks: 100, 110, 110, 120, 120
    # Drawdowns: 0, 0, (90-110)/110 = -0.1818, 0, (80-120)/120 = -0.3333
    assert max_dd == pytest.approx(-0.333333, abs=1e-5)


def test_calculate_sharpe_ratio():
    """Tests Sharpe Ratio calculation."""
    # Arrange
    net_pnl = pd.Series([100.0, 200.0, -50.0, 150.0])
    initial_capital = 10000.0

    # Act
    sharpe = metrics.calculate_sharpe_ratio(net_pnl, initial_capital)

    # Assert
    returns = net_pnl / initial_capital
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)
    expected_sharpe = (mean_return / std_return) * (252**0.5)
    assert sharpe == pytest.approx(expected_sharpe)


def test_calculate_sqn():
    """Tests System Quality Number calculation."""
    # Arrange
    r_multiples = pd.Series([1.0, 2.0, -1.0, 0.5, 1.5])

    # Act
    sqn = metrics.calculate_sqn(r_multiples)

    # Assert
    mean_r = r_multiples.mean()
    std_r = r_multiples.std(ddof=1)
    expected_sqn = (mean_r / std_r) * (len(r_multiples) ** 0.5)
    assert sqn == pytest.approx(expected_sqn)


def test_calculate_kelly_criterion():
    """Tests Kelly Criterion calculation."""
    # Act
    kelly = metrics.calculate_kelly_criterion(0.6, 2.0)

    # Assert
    # K = 0.6 - (0.4 / 2.0) = 0.6 - 0.2 = 0.4
    assert kelly == pytest.approx(0.4)


def test_calculate_max_drawdown_with_initial_value():
    """Tests maximum drawdown calculation with initial value prepended."""
    # Arrange
    equity = pd.Series([95.0, 105.0, 98.0])

    # Act
    max_dd = metrics.calculate_max_drawdown(equity, initial_value=100.0)

    # Assert
    # Extended series: [100.0, 95.0, 105.0, 98.0]
    # Running peaks:   [100.0, 100.0, 105.0, 105.0]
    # Drawdowns:       [0.0, -0.05, 0.0, -0.0666666]
    # Max drawdown:    -0.0666666
    assert max_dd == pytest.approx(-0.0666666, abs=1e-5)
