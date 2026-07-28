from unittest.mock import MagicMock

import pytest

from app.services.portfolio.allocator import PortfolioAllocator
from app.services.portfolio.manager import PortfolioManager

# --- Allocator Tests ---


@pytest.fixture
def allocator():
    return PortfolioAllocator()


@pytest.mark.parametrize(
    "trade_input,expected_size,expected_reason_substr",
    [
        # Happy Path: DipBuyer (Budget 2500)
        (
            {"strategy": "DipBuyer", "entry_price": 100.0, "symbol": "TEST"},
            25,
            "Budget",
        ),
        (
            {"strategy": "DipBuyer", "entry_price": 2000.0, "symbol": "TEST"},
            1,
            "Budget",
        ),
        # Edge Case: DipBuyer Price > Budget
        (
            {"strategy": "DipBuyer", "entry_price": 2501.0, "symbol": "TEST"},
            0,
            "Price > Budget",
        ),
        # Happy Path: HoldTarget (Risk 100)
        (
            {
                "strategy": "HoldTarget",
                "entry_price": 50.0,
                "current_stop_loss": 45.0,
                "symbol": "TEST",
            },
            20,  # Risk/Share = 5. 100/5 = 20.
            "Fixed Risk",
        ),
        # Happy Path: TurnoverTiming (Budget 2500)
        (
            {"strategy": "TurnoverTiming", "entry_price": 50.0, "symbol": "TEST"},
            50,
            "Budget",
        ),
        # Edge Case: Invalid Entry Price
        (
            {"strategy": "DipBuyer", "entry_price": 0.0, "symbol": "TEST"},
            0,
            "Invalid Entry Price",
        ),
        # Edge Case: Only Strategy Name Match (Case Insensitive)
        (
            {"strategy": "dipbuyer", "entry_price": 100.0, "symbol": "TEST"},
            25,
            "Budget",
        ),
        # Edge Case: HoldTarget Invalid SL
        (
            {
                "strategy": "HoldTarget",
                "entry_price": 50.0,
                "current_stop_loss": 55.0,
                "symbol": "TEST",
            },
            0,
            "Invalid Stop Loss",
        ),
        # Edge Case: HoldTarget Risk/Share > Risk Amount (Size < 1)
        # Risk 100. Entry 1000, SL 500. Risk/Share 500. 100/500 = 0.2 -> 0.
        (
            {
                "strategy": "HoldTarget",
                "entry_price": 1000.0,
                "current_stop_loss": 500.0,
                "symbol": "TEST",
            },
            0,
            "Risk/Share > Risk Amount",
        ),
        # Happy Path: TGIM (Budget 10000)
        (
            {"strategy": "TGIM", "entry_price": 500.0, "symbol": "QQQ"},
            20,  # 10000 / 500 = 20
            "Budget",
        ),
        (
            {"strategy": "tgim", "entry_price": 500.0, "symbol": "QQQ"},
            20,
            "Budget",
        ),
        # Happy Path: BridgeScout (Budget 10000)
        (
            {"strategy": "BridgeScout", "entry_price": 500.0, "symbol": "QQQ"},
            20,
            "Budget",
        ),
        (
            {"strategy": "bridge_scout", "entry_price": 500.0, "symbol": "QQQ"},
            20,
            "Budget",
        ),
        # Happy Path: BounceBandit (Budget 10000)
        (
            {"strategy": "BounceBandit", "entry_price": 500.0, "symbol": "QQQ"},
            20,
            "Budget",
        ),
        (
            {"strategy": "bounce_bandit", "entry_price": 500.0, "symbol": "QQQ"},
            20,
            "Budget",
        ),
        # Unknown Strategy
        (
            {"strategy": "RandomStrat", "entry_price": 100.0, "symbol": "TEST"},
            0,
            "Unknown Strategy",
        ),
    ],
)
def test_allocator_logic(allocator, trade_input, expected_size, expected_reason_substr):
    # Act
    result = allocator.allocate(trade_input)

    # Assert
    assert result.size == expected_size
    assert expected_reason_substr in result.reason


# --- Manager Tests ---


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def manager(mock_repo):
    return PortfolioManager(mock_repo)


def test_manager_processes_trades_successfully(manager, mock_repo):
    # Arrange
    trade_1 = {
        "id": "1",
        "symbol": "A",
        "strategy": "DipBuyer",
        "entry_price": 100.0,
        "initial_size": 0,
    }
    trade_2 = {
        "id": "2",
        "symbol": "B",
        "strategy": "HoldTarget",
        "entry_price": 50.0,
        "current_stop_loss": 40.0,
        "initial_size": 0,
    }

    # Mock return
    mock_repo.get_by_status.return_value = [trade_1, trade_2]

    # Act
    count = manager.process_daily_signals()

    # Assert
    assert count == 2
    # Verify updates
    assert mock_repo.update_trade.call_count == 2

    # Check Trade 1 Update (DipBuyer: 2500/100 = 25)
    import json

    args_1, kwargs_1 = mock_repo.update_trade.call_args_list[0]
    assert args_1[0] == "1"
    assert args_1[1]["initial_size"] == 25

    # Verify Context (Budget)
    context_1 = json.loads(args_1[1]["signal_context"])
    assert context_1["budget"] == 2500.0

    # Check Trade 2 Update (HoldTarget: 100/(50-40) = 10)
    args_2, kwargs_2 = mock_repo.update_trade.call_args_list[1]
    assert args_2[0] == "2"
    assert args_2[1]["initial_size"] == 10

    # Verify Context (Risk Amount)
    context_2 = json.loads(args_2[1]["signal_context"])
    assert context_2["risk_amount"] == 100.0


def test_manager_skips_trades_with_size(manager, mock_repo):
    # Arrange: Trade already has size
    trade_1 = {
        "id": "1",
        "symbol": "A",
        "strategy": "DipBuyer",
        "entry_price": 100.0,
        "initial_size": 10,
    }
    mock_repo.get_by_status.return_value = [trade_1]

    # Act
    count = manager.process_daily_signals()

    # Assert
    assert count == 0
    mock_repo.update_trade.assert_not_called()


def test_manager_handles_repo_exception_gracefully(manager, mock_repo):
    # Arrange
    import sqlite3

    mock_repo.get_by_status.side_effect = sqlite3.OperationalError(
        "DB Connection Failed"
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="PortfolioManager: Database unavailable."):
        manager.process_daily_signals()


def test_manager_handles_allocation_failure_gracefully(manager, mock_repo):
    # Arrange: Trade that fails allocation (e.g. price > budget)
    trade_1 = {
        "id": "1",
        "symbol": "A",
        "strategy": "DipBuyer",
        "entry_price": 5000.0,
        "initial_size": 0,
    }
    mock_repo.get_by_status.return_value = [trade_1]

    # Act
    count = manager.process_daily_signals()

    # Assert
    assert count == 0
    mock_repo.update_trade.assert_not_called()


def test_portfolio_config_strategy_budgets():
    from app.config import AppConfig, PortfolioConfig

    config = PortfolioConfig()
    assert config.get_budget("tgim") == 10000.0
    assert config.get_budget("bounce_bandit") == 10000.0
    assert config.get_budget("bridge_scout") == 10000.0
    assert config.get_budget("ndx_momentum") == 10000.0
    assert config.get_budget("dip_buyer") == 2500.0

    # Test YAML dict deserialization
    raw_data = {
        "portfolio": {
            "strategies": {
                "tgim": {"budget": 12000.0},
                "bounce_bandit": {"budget": 15000.0},
                "bridge_scout": {"budget": 11000.0},
            }
        }
    }
    app_cfg = AppConfig.from_dict(raw_data)
    assert app_cfg.portfolio.get_budget("tgim") == 12000.0
    assert app_cfg.portfolio.get_budget("bounce_bandit") == 15000.0
    assert app_cfg.portfolio.get_budget("bridge_scout") == 11000.0
