import pytest
import pandas as pd
import json
from unittest.mock import MagicMock
from app.services.trade_manager.strategies.split_target import SplitTargetStrategy
from app.database.repositories.trade import TradeRepository
from app.types import TradeStatus


@pytest.fixture
def strategy():
    return SplitTargetStrategy()


@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)


def test_check_entry_breakout(strategy, mock_repo):
    trade = {
        "id": 1,
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "initial_size": 10,
        "signal_context": json.dumps({"date": "2024-01-01"}),
    }

    # Gap Up Case
    candle = pd.Series(
        {
            "date": "2024-01-02",
            "open": 101.0,
            "high": 105.0,
            "low": 99.0,
            "close": 104.0,
        }
    )
    result = strategy.check_entry(trade, candle, pd.DataFrame(), mock_repo)

    assert "FILLED" in result
    mock_repo.update_trade.assert_called_once()
    args, _ = mock_repo.update_trade.call_args
    updates = args[1]

    assert updates["status"] == TradeStatus.ACTIVE
    assert updates["entry_price"] == 101.0
    assert updates["current_size"] == 10.0


def test_manage_active_trade_tp1_split(strategy, mock_repo):
    # Setup: Active Trade
    context = {"take_profit_1": 110.0, "take_profit_3": 130.0, "is_phase_2": False}
    trade = {
        "id": 1,
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "current_size": 10.0,
        "realized_pnl": 0.0,
        "signal_context": json.dumps(context),
    }

    # Candle hits TP1 (112) but NOT TP3 (130)
    candle = pd.Series(
        {
            "date": "2024-01-05",
            "open": 105.0,
            "high": 112.0,
            "low": 104.0,
            "close": 111.0,
        }
    )
    df_hist = pd.DataFrame([candle])

    result = strategy.manage_active_trade(trade, df_hist, mock_repo)

    assert "TP1 HIT" in result
    assert "Partial Sell" in result

    # Verify Update: HALF sold (5 remain), SL moved to Entry (100.0)
    args, _ = mock_repo.update_trade.call_args
    updates = args[1]

    assert updates["current_size"] == 5.0
    assert updates["current_stop_loss"] == 100.0
    # (110 - 100) * 5 = 50.0
    assert updates["realized_pnl"] == 50.0


def test_manage_active_trade_gap_over_tp3(strategy, mock_repo):
    """
    Test Gap-Over: Candle hits TP1 AND TP3.
    Should close the entire position (Phase 3 logic).
    """
    context = {"take_profit_1": 110.0, "take_profit_3": 130.0, "is_phase_2": False}
    trade = {
        "id": 1,
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "current_size": 10.0,
        "realized_pnl": 0.0,
        "signal_context": json.dumps(context),
    }

    # Candle gaps over TP1 (110) AND TP3 (130)
    # Open=135 is above both
    candle = pd.Series(
        {
            "date": "2024-01-05",
            "open": 135.0,
            "high": 140.0,
            "low": 134.0,
            "close": 138.0,
        }
    )
    df_hist = pd.DataFrame([candle])

    result = strategy.manage_active_trade(trade, df_hist, mock_repo)

    assert "TARGET_HIT" in result

    # Verify FULL Close
    args, _ = mock_repo.update_trade.call_args
    updates = args[1]

    assert updates["status"] == TradeStatus.CLOSED
    assert updates["exit_price"] == 135.0  # Open price execution
    assert updates["current_size"] == 0
    # (135 - 100) * 10 = 350.0
    assert updates["realized_pnl"] == 350.0


def test_manage_active_trade_tp3_final(strategy, mock_repo):
    # Setup: Phase 2 Trade
    context = {"take_profit_1": 110.0, "take_profit_3": 130.0, "is_phase_2": True}
    trade = {
        "id": 1,
        "symbol": "TEST",
        "entry_price": 100.0,
        "current_stop_loss": 100.0,  # BE
        "current_size": 5.0,
        "realized_pnl": 50.0,
        "signal_context": json.dumps(context),
    }

    # Candle hits TP3
    candle = pd.Series(
        {
            "date": "2024-01-10",
            "open": 120.0,
            "high": 135.0,
            "low": 119.0,
            "close": 132.0,
        }
    )
    df_hist = pd.DataFrame([candle])

    result = strategy.manage_active_trade(trade, df_hist, mock_repo)

    assert "TARGET_HIT" in result

    # Verify Close
    args, _ = mock_repo.update_trade.call_args
    updates = args[1]

    assert updates["status"] == TradeStatus.CLOSED
    assert updates["exit_price"] == 130.0
    # 50 + (130 - 100) * 5 = 50 + 150 = 200
    assert updates["realized_pnl"] == 200.0
