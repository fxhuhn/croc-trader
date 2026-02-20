import pytest
import pandas as pd
from unittest.mock import patch
from app.services.backtester.analytics import MetricsCalculator, BacktestAnalytics
from app.models import BacktestMetrics


@pytest.fixture
def sample_trades_df():
    """Provides a sample trades DataFrame for testing."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "realized_pnl": [1000.0, -500.0, 1500.0],
            "entry_price": [150.0, 300.0, 2800.0],
            "exit_price": [160.0, 295.0, 2815.0],
            "initial_size": [100, 100, 10],
            "entry_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "exit_date": ["2023-01-05", "2023-01-06", "2023-01-07"],
            "strategy": ["DipBuyer", "DipBuyer", "TwoPercent"],
            "initial_stop": [145.0, 305.0, 2750.0],
            "mae": [2.0, 10.0, 50.0],
            "mfe": [15.0, 2.0, 20.0],
        }
    )


def test_metrics_calculator_basic(sample_trades_df):
    """Tests that MetricsCalculator produced expected BacktestMetrics."""
    # Arrange
    calculator = MetricsCalculator()
    initial_capital = 100000.0

    # Act
    metrics = calculator.calculate_trade_metrics(
        trades_dataframe=sample_trades_df, initial_capital=initial_capital
    )

    # Assert
    assert isinstance(metrics, BacktestMetrics)
    assert metrics.total_trades == 3
    assert metrics.net_profit > 0  # 1000 - 500 + 1500 = 2000 (minus costs)
    assert 0.0 < metrics.win_rate < 1.0


@patch("app.services.backtester.analytics.BacktestDataLoader")
def test_backtest_analytics_orchestration(mock_loader_class, sample_trades_df):
    """Tests the orchestration logic in BacktestAnalytics."""
    # Arrange
    mock_loader = mock_loader_class.return_value
    mock_loader.fetch_closed_trades.return_value = sample_trades_df
    mock_loader.calculate_exposure_and_benchmark.return_value = {
        "exposure_pct": 0.5,
        "benchmark_return": 0.05,
    }

    analytics = BacktestAnalytics("backtest.db", "market.db")

    # Act
    results = analytics.run_strategy_analysis()

    # Assert
    assert "DipBuyer" in results
    assert "TwoPercent" in results
    assert results["DipBuyer"].total_trades == 2
    assert results["TwoPercent"].total_trades == 1
    assert results["DipBuyer"].market_exposure_pct == 0.5


def test_transaction_cost_model():
    """Tests the TransactionCostModel logic."""
    from app.services.backtester.analytics import TransactionCostModel

    # Arrange
    cost_model = TransactionCostModel(slippage_bps=10.0)
    trades_df = pd.DataFrame(
        {
            "initial_size": [
                100,
                10,
            ],  # 100 * 0.01 = 1.0 (min 2.0), 10 * 0.01 = 0.1 (min 2.0)
            "entry_price": [100.0, 1000.0],
        }
    )

    # Act
    costs = cost_model.calculate_cost(trades_df)

    # Assert
    # Commissions: (2.0 * 2) * 2 = 8.0  (Wait, 2.0 per order, entry and exit = 4.0 per trade)
    # Trade 1: max(2, 100*0.01)*2 = 4.0. Notional = 10000. Slippage = 10000 * 0.001 * 2 = 20.0. Total = 24.0
    # Trade 2: max(2, 10*0.01)*2 = 4.0. Notional = 10000. Slippage = 10000 * 0.001 * 2 = 20.0. Total = 24.0
    assert costs.iloc[0] == 24.0
    assert costs.iloc[1] == 24.0
