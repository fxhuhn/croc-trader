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


def test_calculate_drawdown_series():
    """Tests full drawdown series calculation including empty series and initial capital."""
    assert metrics.calculate_drawdown_series(pd.Series([], dtype=float)).empty

    equity = pd.Series([100.0, 110.0, 90.0, 120.0, 80.0])
    dd_series = metrics.calculate_drawdown_series(equity)
    assert len(dd_series) == 5
    assert dd_series.iloc[0] == 0.0
    assert dd_series.iloc[1] == 0.0
    assert dd_series.iloc[2] == pytest.approx(-0.181818, abs=1e-5)
    assert dd_series.iloc[3] == 0.0
    assert dd_series.iloc[4] == pytest.approx(-0.333333, abs=1e-5)


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


def test_calculate_return_on_invested():
    """Tests return on invested capital calculation."""
    # Arrange
    pnl = pd.Series([200.0, -100.0, 400.0])
    invested = pd.Series([1000.0, 2000.0, 2000.0])

    # Act
    return_pct = metrics.calculate_return_on_invested(pnl, invested)

    # Assert
    # Total PnL = 500, Total Invested = 5000 -> 500 / 5000 * 100 = 10.0%
    assert return_pct == pytest.approx(10.0)


def test_calculate_return_on_invested_empty():
    """Tests return on invested capital with empty series or zero investment."""
    assert metrics.calculate_return_on_invested(pd.Series([]), pd.Series([])) == 0.0
    assert (
        metrics.calculate_return_on_invested(pd.Series([100.0]), pd.Series([0.0]))
        == 0.0
    )


def test_calculate_sharpe_ratio_from_roi():
    """Tests Sharpe Ratio from ROI series."""
    # Arrange
    roi_series = pd.Series([0.05, 0.02, -0.01, 0.04])
    trades_per_year = 50.0

    # Act
    sharpe = metrics.calculate_sharpe_ratio_from_roi(roi_series, trades_per_year)

    # Assert
    mean_roi = roi_series.mean()
    std_roi = roi_series.std(ddof=1)
    expected_sharpe = (mean_roi / std_roi) * (trades_per_year**0.5)
    assert sharpe == pytest.approx(expected_sharpe)


def test_calculate_sharpe_ratio_from_roi_insufficient_data():
    """Tests Sharpe Ratio with insufficient data points."""
    assert metrics.calculate_sharpe_ratio_from_roi(pd.Series([0.05])) == 0.0
    assert metrics.calculate_sharpe_ratio_from_roi(pd.Series([])) == 0.0


def test_calculate_sortino_ratio_from_roi():
    """Tests Sortino Ratio from ROI series with downside deviation."""
    # Arrange
    roi_series = pd.Series([0.10, -0.05, 0.08, -0.02])
    trades_per_year = 40.0

    # Act
    sortino = metrics.calculate_sortino_ratio_from_roi(
        roi_series, trades_per_year=trades_per_year, target_return=0.0
    )

    # Assert
    underperformance = [0.0, -0.05, 0.0, -0.02]
    downside_dev = (sum(x**2 for x in underperformance) / 4) ** 0.5
    mean_roi = roi_series.mean()
    expected_sortino = (mean_roi / downside_dev) * (trades_per_year**0.5)
    assert sortino == pytest.approx(expected_sortino)


def test_calculate_sortino_ratio_from_roi_no_downside():
    """Tests Sortino Ratio when there are no negative returns."""
    roi_series = pd.Series([0.05, 0.10, 0.02])
    assert metrics.calculate_sortino_ratio_from_roi(roi_series) == 0.0
