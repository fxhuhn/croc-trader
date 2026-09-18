import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.const import Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.croc_setup import (
    CrocSetupStrategy,
    PriceData,
    enrich_sma_distances,
    evaluate_indicator_condition,
    find_best_rule_match,
)

# --- FIXTURES ---


@pytest.fixture
def mock_trade_repository():
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_signal_repository():
    return MagicMock(spec=SignalRepository)


@pytest.fixture
def mock_provider():
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    return [
        {
            "Signal": "TestSignal",
            "Score": 5.0,
            "RSI": "Oversold (<30)",
            "Dist_SMA_200": "> 10%",
            "Exit": "Hold_TP3",
        },
        {
            "Signal": "TestSignal",
            "Score": 2.0,
            "RSI": "Neutral",
            "Dist_SMA_200": "0 to 3%",
            "Exit": "Standard",
        },
    ]


@pytest.fixture
def strategy(
    mock_trade_repository, mock_signal_repository, mock_provider, sample_rules
):
    # Patch _load_config to return our sample rules directly
    with patch.object(CrocSetupStrategy, "_load_config", return_value=sample_rules):
        # Patch symbol sets to avoid file access
        with patch.object(CrocSetupStrategy, "_get_indices_string", return_value="SPX"):
            strat = CrocSetupStrategy(
                trade_repository=mock_trade_repository,
                data_provider=mock_provider,
                signal_repository=mock_signal_repository,
            )
            yield strat


# --- LOGIC TESTS ---


def test_compute_sma_distances_calculation():
    """Test ((Close - SMA)/SMA)*100 logic."""
    row = {
        "close": 110.0,
        "sma_20": 100.0,  # +10%
        "sma_200": 100.0,  # +10%
    }
    prices = PriceData.from_row(row)
    assert prices is not None
    enriched = enrich_sma_distances(row, prices)

    assert enriched["dist_sma_20"] == 10.0
    assert enriched["dist_sma_200"] == 10.0


def test_compute_sma_distances_safe_failure():
    """Test robustness against missing/bad values via PriceData."""
    row = {"close": "bad", "sma_20": None}
    prices = PriceData.from_row(row)
    assert prices is None


def test_check_condition_rsi():
    """Test RSI string matching."""
    # "Oversold (<30)" -> < 30
    assert evaluate_indicator_condition(25.0, "Oversold (<30)") is True
    assert evaluate_indicator_condition(35.0, "Oversold (<30)") is False

    # "Neutral" -> 45-55
    assert evaluate_indicator_condition(50.0, "Neutral") is True


def test_check_condition_ema():
    """Test SMA/EMA range matching."""
    # "0 to 3%" -> 0 <= val <= 3
    assert evaluate_indicator_condition(1.5, "0 to 3%") is True
    assert evaluate_indicator_condition(5.0, "0 to 3%") is False

    # "< -10%" -> val < -10
    assert evaluate_indicator_condition(-12.0, "< -10%") is True


# --- FLOW TESTS ---


def test_find_best_match(strategy):
    """Test finding the correct rule rule based on criteria."""
    # Case 1: High Score Match
    # RSI=25 (Oversold), Dist=12 (>10%) -> Matches first rule (Score 5.0)
    data_1 = {"signal": "TestSignal", "rsi": 25.0, "dist_sma_200": 12.0}
    match_1 = find_best_rule_match(data_1, strategy.ranking_rules)
    assert match_1 is not None
    assert match_1["Score"] == 5.0

    # Case 2: Low Score Match
    # RSI=50 (Neutral), Dist=2 (0-3%) -> Matches second rule (Score 2.0)
    data_2 = {"signal": "TestSignal", "rsi": 50.0, "dist_sma_200": 2.0}
    match_2 = find_best_rule_match(data_2, strategy.ranking_rules)
    assert match_2 is not None
    assert match_2["Score"] == 2.0

    # Case 3: No Match
    data_3 = {"signal": "TestSignal", "rsi": 90.0}
    match_3 = find_best_rule_match(data_3, strategy.ranking_rules)
    assert match_3 is None


def test_run_execution(strategy, mock_signal_repository, mock_trade_repository):
    """Integration Test: Signal -> Trade"""
    # 1. Mock DB Row
    mock_row = {
        "id": 1,
        "timestamp": "2025-01-01",
        "data": json.dumps(
            {
                "signal": "TestSignal",
                "symbol": "AAPL",
                "close": 110.0,
                "high": 115.0,
                "low": 105.0,
                "volume": 1000,
                "sma_20": 100.0,
                "sma_200": 90.0,
                "rsi": 25.0,
            }
        ),
    }
    mock_signal_repository.get_signals_by_date.return_value = [mock_row]

    # 2. Run
    strategy.run(days=1)

    # 3. Assert Verify Trade Creation
    mock_trade_repository.create_trade.assert_called_once()

    kwargs = mock_trade_repository.create_trade.call_args[1]
    assert kwargs["symbol"] == "AAPL"
    assert (
        kwargs["strategy"] == Strategies.HoldTarget
    )  # From rule exit logic (Hold_TP3)
    assert kwargs["stop_loss"] == 105.0  # Entry(115) - Risk(10)
    assert kwargs["target"] == 145.0  # Entry(115) + 3*Risk(30)


def test_run_execution_skipped_no_bucket(
    strategy, mock_signal_repository, mock_trade_repository
):
    """Test that signals with no index bucket (indices='-') are skipped."""
    # 1. Mock DB Row with valid signal data structure
    mock_row = {
        "id": 1,
        "timestamp": "2025-01-01",
        "symbol": "UNKNOWN_STOCK",
        "data": json.dumps(
            {
                "signal": "TestSignal",
                "symbol": "UNKNOWN_STOCK",
                "close": 110.0,
                "high": 115.0,
                "low": 105.0,
                "volume": 1000,
                "sma_20": 100.0,
                "sma_200": 90.0,
                "rsi": 25.0,
            }
        ),
    }
    mock_signal_repository.get_signals_by_date.return_value = [mock_row]

    # 2. Force _get_indices_string to return "-"
    with patch.object(strategy, "_get_indices_string", return_value="-"):
        hits = strategy.run(days=1)

    # 3. Assertions
    assert hits == 0
    mock_trade_repository.create_trade.assert_not_called()
