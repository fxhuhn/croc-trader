# filename: test_analytics_math.py
import pytest
import pandas as pd
from app.services.backtester.analytics import (
    safe_divide,
    safe_percentile,
    MonteCarloSimulator,
    TradeQualityAnalyzer,
)


def test_safe_divide_returns_expected_results() -> None:
    """Verifies division handles zero and small denominators safely."""
    # Arrange & Act & Assert
    assert safe_divide(10, 2) == 5.0
    assert safe_divide(10, 0) == 0.0
    assert safe_divide(10, 1e-8) == 0.0
    assert safe_divide(10, 1e-5) == 10 / 1e-5


@pytest.mark.parametrize(
    "input_list, percentile, expected",
    [
        ([], 25, 0.0),
        ([1, 2, 3], 50, 2.0),
        ([10, 20, 30, 40], 75, 32.5),
    ],
)
def test_safe_percentile_calculates_correctly(
    input_list: list[float], percentile: float, expected: float
) -> None:
    """Verifies percentile calculation with empty and valid data."""
    # Act & Assert
    assert safe_percentile(input_list, percentile) == expected


def test_monte_carlo_simulator_kelly_bootstrap_handles_all_losses() -> None:
    """Verifies that all losses result in zero Kelly expectancy."""
    # Arrange
    trades_dataframe = pd.DataFrame(
        {
            "net_pnl": [-100.0, -50.0] * 10,
            "entry_price": [100.0] * 20,
            "initial_size": [1.0] * 20,
        }
    )
    simulator = MonteCarloSimulator(iterations=100)

    # Act
    results = simulator.run_safe_kelly_bootstrap(trades_dataframe)

    # Assert
    assert results["mean"] == 0.0
    assert results["safe"] == 0.0


def test_monte_carlo_simulator_kelly_bootstrap_handles_no_trades() -> None:
    """Verifies that empty input returns zeroed metrics."""
    # Arrange
    trades_dataframe = pd.DataFrame(columns=["net_pnl"])
    simulator = MonteCarloSimulator(iterations=100)

    # Act
    results = simulator.run_safe_kelly_bootstrap(trades_dataframe)

    # Assert
    assert results["mean"] == 0.0


def test_monte_carlo_simulator_kelly_bootstrap_handles_mixed_trades() -> None:
    """Verifies Kelly calculation for a robust sample of wins and losses."""
    # Arrange
    # 30 trades to satisfy minimum requirements in implementation
    pnl_history = [100.0, -50.0] * 15
    trades_dataframe = pd.DataFrame(
        {
            "net_pnl": pnl_history,
            "entry_price": [100.0] * 30,
            "initial_size": [1.0] * 30,
        }
    )
    simulator = MonteCarloSimulator(iterations=100)

    # Act
    results = simulator.run_safe_kelly_bootstrap(trades_dataframe)

    # Assert
    assert results["mean"] > 0.0
    assert 0.0 <= results["safe"] <= 1.0


def test_monte_carlo_simulator_kelly_bootstrap_handles_all_wins() -> None:
    """Verifies that consistent wins result in high Kelly scores."""
    # Arrange
    pnl_history = [100.0] * 25
    trades_dataframe = pd.DataFrame(
        {
            "net_pnl": pnl_history,
            "entry_price": [100.0] * 25,
            "initial_size": [1.0] * 25,
        }
    )
    simulator = MonteCarloSimulator(iterations=100)

    # Act
    results = simulator.run_safe_kelly_bootstrap(trades_dataframe)

    # Assert
    assert results["mean"] >= 0.9


def test_trade_quality_scoring_for_winning_trade() -> None:
    """Verifies high quality score for a profitable target hit."""
    # Arrange
    analyzer = TradeQualityAnalyzer()
    winning_trade = {
        "id": 1,
        "entry_price": 100.0,
        "initial_stop": 95.0,  # Risk = 5
        "initial_size": 1.0,
        "realized_pnl": 10.0,  # R = 2.0
        "exit_reason": "TARGET_HIT",
    }

    # Act
    score_metadata = analyzer.score_trade(winning_trade)

    # Assert
    # Base calculation expectation: Entry(25) + Risk(18) + Context(15) + Exit(30) = 88
    assert score_metadata["total_score"] >= 85
    assert score_metadata["grade"].startswith("A") or score_metadata["grade"] == "B+"


def test_trade_quality_scoring_for_losing_trade() -> None:
    """Verifies low quality score for a stopped out trade."""
    # Arrange
    analyzer = TradeQualityAnalyzer()
    losing_trade = {
        "id": 2,
        "entry_price": 100.0,
        "initial_stop": 95.0,
        "initial_size": 1.0,
        "realized_pnl": -5.0,  # R = -1.0
        "exit_reason": "STOP_LOSS",
    }

    # Act
    score_metadata = analyzer.score_trade(losing_trade)

    # Assert
    assert score_metadata["total_score"] <= 50
    assert score_metadata["grade"] == "F"
