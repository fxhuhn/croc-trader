# filename: test_engine.py
import pytest
from unittest.mock import MagicMock
from app.services.screener.engine import ScreenerEngine, ScreenerConfiguration
from app.services.screener.protocols import StrategyProtocol


@pytest.fixture
def mock_dependencies():
    """Provides mock repositories and data provider."""
    trade_repo = MagicMock()
    signal_repo = MagicMock()
    data_provider = MagicMock()
    return trade_repo, signal_repo, data_provider


@pytest.fixture
def mock_strategy():
    """Provides a mock strategy implementing StrategyProtocol."""
    strategy = MagicMock(spec=StrategyProtocol)
    strategy.name = "mock_strategy"
    strategy.run.return_value = 5
    return strategy


def test_screener_engine_registration(mock_dependencies, mock_strategy) -> None:
    """Verifies that strategies can be registered correctly."""
    # Arrange
    trade_repo, signal_repo, data_provider = mock_dependencies
    engine = ScreenerEngine(trade_repo, signal_repo, data_provider)

    # Act
    engine.register_strategy(mock_strategy)

    # Assert
    assert len(engine.active_strategies) == 1
    assert engine.get_strategy("mock_strategy") == mock_strategy


def test_screener_engine_run_all_executes_strategies(
    mock_dependencies, mock_strategy
) -> None:
    """Verifies that run_all orchestrates strategy execution correctly."""
    # Arrange
    trade_repo, signal_repo, data_provider = mock_dependencies
    engine = ScreenerEngine(
        trade_repo, signal_repo, data_provider, strategies=[mock_strategy]
    )
    data_provider.get_latest_date.return_value = "2026-02-17"

    # Act
    results = engine.run_all(days=0)

    # Assert
    assert results == {"mock_strategy": 5}
    mock_strategy.run.assert_called_once_with(days=0, analysis_date="2026-02-17")
    data_provider.clear_cache.assert_called_once()


def test_screener_engine_handles_strategy_failure(
    mock_dependencies, mock_strategy, caplog
) -> None:
    """Verifies that a failure in one strategy does not crash the engine."""
    # Arrange
    trade_repo, signal_repo, data_provider = mock_dependencies
    failing_strategy = MagicMock(spec=StrategyProtocol)
    failing_strategy.name = "failing_strategy"
    failing_strategy.run.side_effect = RuntimeError("Strategy failed")

    engine = ScreenerEngine(
        trade_repository=trade_repo,
        signal_repository=signal_repo,
        data_provider=data_provider,
        strategies=[failing_strategy, mock_strategy],
    )

    # Act
    results = engine.run_all(days=0)

    # Assert
    assert results["failing_strategy"] == 0
    assert results["mock_strategy"] == 5
    assert "Error executing strategy failing_strategy" in caplog.text


def test_screener_engine_strategy_filter(mock_dependencies, mock_strategy) -> None:
    """Verifies that the strategy filter works correctly."""
    # Arrange
    trade_repo, signal_repo, data_provider = mock_dependencies
    other_strategy = MagicMock(spec=StrategyProtocol)
    other_strategy.name = "other_strategy"

    engine = ScreenerEngine(
        trade_repository=trade_repo,
        signal_repository=signal_repo,
        data_provider=data_provider,
        strategies=[mock_strategy, other_strategy],
    )

    # Act
    results = engine.run_all(strategy_filter="mock_strategy")

    # Assert
    assert "mock_strategy" in results
    assert "other_strategy" not in results
    mock_strategy.run.assert_called_once()
    other_strategy.run.assert_not_called()


def test_screener_engine_init_with_configuration(mock_dependencies) -> None:
    """Verifies that the engine initializes correctly with configuration."""
    # Arrange
    trade_repo, signal_repo, data_provider = mock_dependencies
    config: ScreenerConfiguration = {"strategy_ranking": ["a", "b"]}

    # Act
    engine = ScreenerEngine(
        trade_repository=trade_repo,
        signal_repository=signal_repo,
        data_provider=data_provider,
        configuration=config,
    )

    # Assert
    assert engine.configuration == config
