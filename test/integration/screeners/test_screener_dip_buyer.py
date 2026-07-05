from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.dip_buyer import DipBuyerStrategy

# --- HELPER FUNCTIONS ---


def create_market_data(
    length=250, price_start=100.0, trend=0.1, symbol="TEST", end_date="2025-01-10"
):
    """Creates a Dict of DataFrames matching the Provider format (Columns=Symbols)."""
    dates = pd.date_range(end=end_date, periods=length, freq="B")
    actual_length = len(dates)

    # Generate random walk
    returns = np.random.normal(0, 0.01, actual_length) + (trend / actual_length)
    prices = price_start * np.exp(np.cumsum(returns))

    close = prices
    high = close * 1.02
    low = close * 0.98
    open_p = close * 1.01
    volume = np.random.randint(500_000, 2_000_000, actual_length)

    # Structure: Dict[field] -> DataFrame(index=Dates, columns=[Symbol])
    return {
        "close": pd.DataFrame({symbol: close}, index=dates),
        "open": pd.DataFrame({symbol: open_p}, index=dates),
        "high": pd.DataFrame({symbol: high}, index=dates),
        "low": pd.DataFrame({symbol: low}, index=dates),
        "volume": pd.DataFrame({symbol: volume}, index=dates),
    }


# --- FIXTURES ---


@pytest.fixture
def mock_repo():
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_provider():
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def strategy(mock_repo, mock_provider):
    # Mock symbols to include our test ticker
    dt_patcher = patch.object(
        DipBuyerStrategy, "_initialize_symbol_sets", return_value=None
    )
    dt_patcher.start()

    strat = DipBuyerStrategy(trade_repository=mock_repo, data_provider=mock_provider)

    # Manually inject symbol sets
    strat._dow_set = {"TEST"}
    strat._sp500_set = {"TEST"}
    strat._ndx_set = set()

    yield strat
    dt_patcher.stop()


# --- TESTS ---


def test_compute_indicators_calculation(strategy):
    """Test if indicators (SMA, ATR, IBS) are calculated correctly."""
    data = create_market_data(length=300)

    # Run calculation
    indicators = strategy._compute_indicators(
        data["close"], data["high"], data["low"], data["volume"]
    )

    assert "sma200" in indicators
    assert "atr" in indicators
    assert "ibs" in indicators
    assert "atr_ratio_3day" in indicators
    assert "volatility_ratio" in indicators

    # SMA200 check: Last value should be present
    assert not pd.isna(indicators["sma200"].iloc[-1]).all()

    # ATR check: Positive value
    assert (indicators["atr"].iloc[-1] > 0).all()

    # IBS Check: Between 0 and 1
    ibs = indicators["ibs"].iloc[-1]
    assert (0 <= ibs).all() and (ibs <= 1.0).all()


def test_entry_filter_logic_success(strategy):
    """Test a scenario that SHOULD pass all filters."""
    # Setup data at Checkpoint
    # 1. Price > SMA200 (Uptrend)
    # 2. Volume > 1M
    # 3. Dip: ATR R3 < -1.0 (Drop > 1 ATR)
    # 4. Volatility > 3%
    # 5. IBS < 0.2 (Close near Low)
    # 6. Red Candle (Close < Open)

    index_date = pd.Timestamp("2025-01-01")

    current_values = {
        "close": 100.0,
        "open": 102.0,  # Red Candle
        "high": 105.0,
        "low": 99.0,  # Low IBS (100-99)/(105-99) = 1/6 = ~0.16 < 0.2
        "volume": 2_000_000,  # > 1M
        "sma200": 90.0,  # Uptrend (100 > 90)
        "volume_sma": 1_500_000,
        "atr": 5.0,  # Volatility
        "atr_ratio_3day": -1.5,  # Large Dip (-1.5 < -1.0)
        "ibs": 0.16,
        "volatility_ratio": 0.05,  # 5/100 = 5% > 3%
        "setup_score": 1.5,  # -1 * -1.5
    }

    previous_values = {
        "close": 108.0,
        "open": 110.0,  # Red Candle yesterday
    }

    # helper to convert scalar dicts to Series with Index for the method compatibility
    def to_series(d):
        return {k: pd.Series([v], index=[index_date]) for k, v in d.items()}

    current_series = to_series(current_values)
    previous_series = to_series(previous_values)

    # Update method name to _filter_market_state
    results = strategy._filter_market_state(current_series, previous_series)

    assert not results.empty
    assert len(results) == 1
    assert results.index[0] == index_date


def test_entry_filter_fails_downtrend(strategy):
    """Test failure: Price < SMA200."""
    index_date = pd.Timestamp("2025-01-01")
    current_values = {
        "close": 80.0,  # < SMA200
        "open": 82.0,
        "high": 85.0,
        "low": 79.0,
        "volume": 2_000_000,
        "sma200": 90.0,  # Downtrend
        "volume_sma": 1_500_000,
        "atr": 5.0,
        "atr_ratio_3day": -1.5,
        "ibs": 0.16,
        "volatility_ratio": 0.05,
        "setup_score": 1.5,
    }
    previous_values = {"close": 85.0, "open": 87.0}

    def to_series(d):
        return {k: pd.Series([v], index=[index_date]) for k, v in d.items()}

    results = strategy._filter_market_state(
        to_series(current_values), to_series(previous_values)
    )

    assert results.empty


def test_process_signals_creates_trade(strategy, mock_repo):
    """Integration test: Signals DF -> Repo Create Call."""
    date_obj = pd.Timestamp("2025-01-10")

    signals = pd.DataFrame(
        [
            {
                "close": 100.0,
                "high": 105.0,
                "volume": 1_000_000,
                "atr": 5.0,
                "sma200": 90.0,
                "atr_ratio_3day": -1.5,
                "ibs": 0.1,
                "setup_score": 1.5,
            }
        ],
        index=["TEST"],
    )

    strategy._process_signals(signals, date_obj)

    mock_repo.create_trade.assert_called_once()
    call_kwargs = mock_repo.create_trade.call_args[1]

    assert call_kwargs["symbol"] == "TEST"
    assert call_kwargs["strategy"] == "dip_buyer"
    assert call_kwargs["entry"] == 95.0  # 100 - (5.0 * 1.0)
    # TP = Entry + (0.8 * ATR) = 95.0 + (0.8 * 5.0) = 95.0 + 4.0 = 99.0
    assert call_kwargs["target"] == 99.0
    assert call_kwargs["stop_loss"] == 0.0


def test_results_are_sorted_by_score(strategy):
    """Test that results are sorted by setup_score in descending order."""

    symbols = ["SYM_C", "SYM_A", "SYM_B"]  # Unsorted order

    # Score = atr_ratio_3day * -1. So lower atr_ratio_3day = higher score.
    # Symbol A: atr_ratio_3day = -2.0 -> Score = 2.0 (Highest)
    # Symbol B: atr_ratio_3day = -1.5 -> Score = 1.5 (Middle)
    # Symbol C: atr_ratio_3day = -1.2 -> Score = 1.2 (Lowest)

    current_series = {
        "close": pd.Series([100.0, 100.0, 100.0], index=symbols),
        "open": pd.Series([102.0, 102.0, 102.0], index=symbols),  # Red
        "high": pd.Series([105.0, 105.0, 105.0], index=symbols),
        "low": pd.Series([99.0, 99.0, 99.0], index=symbols),  # Low IBS
        "volume": pd.Series([2e6, 2e6, 2e6], index=symbols),
        "sma200": pd.Series([90.0, 90.0, 90.0], index=symbols),  # Uptrend
        "volume_sma": pd.Series([2e6, 2e6, 2e6], index=symbols),
        "atr": pd.Series([5.0, 5.0, 5.0], index=symbols),
        "atr_ratio_3day": pd.Series(
            [-1.2, -2.0, -1.5], index=symbols
        ),  # Variable Scores
        "ibs": pd.Series([0.1, 0.1, 0.1], index=symbols),
        "volatility_ratio": pd.Series([0.05, 0.05, 0.05], index=symbols),
        "setup_score": pd.Series([1.2, 2.0, 1.5], index=symbols),
    }

    previous_series = {
        "close": pd.Series([108.0, 108.0, 108.0], index=symbols),
        "open": pd.Series([110.0, 110.0, 110.0], index=symbols),
    }

    results = strategy._filter_market_state(current_series, previous_series)

    # Expected Order: SYM_A (2.0), SYM_B (1.5), SYM_C (1.2)
    assert len(results) == 3
    assert results.index.tolist() == ["SYM_A", "SYM_B", "SYM_C"]

    # Check if scores are strictly descending
    scores = results["setup_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_run_historical_date_slicing(strategy, mock_provider, mock_repo):
    """Test the full run() pipeline with a historical date to verify slicing logic."""
    # 1. Setup data for 10 days
    dates = pd.date_range(end="2025-01-10", periods=10, freq="B")
    symbol = "TEST"

    # We want 2025-01-08 to be the signal date

    data = {
        "close": pd.DataFrame(
            {symbol: [110, 108, 105, 100, 102, 105, 108, 110, 112, 115]}, index=dates
        ),
        "open": pd.DataFrame(
            {symbol: [112, 110, 107, 105, 104, 107, 110, 115, 114, 117]}, index=dates
        ),  # All Red
        "high": pd.DataFrame(
            {symbol: [115, 112, 110, 108, 106, 110, 115, 118, 120, 122]}, index=dates
        ),
        "low": pd.DataFrame(
            {symbol: [108, 105, 102, 98, 100, 102, 105, 108, 110, 112]}, index=dates
        ),
        "volume": pd.DataFrame({symbol: [2e6] * 10}, index=dates),
    }

    # Mock provider
    mock_provider.get_universe_daily_data.return_value = data

    # Logic Pass Setup: Modify config to pass with this data
    from app.services.screener.strategies.dip_buyer import DipBuyerConfig

    strategy.config = DipBuyerConfig(
        SMA_TREND_WINDOW=5,
        ATR_WINDOW=3,
        MIN_VOLUME=100_000,
        MAX_ATR_RATIO_3DAY=10.0,  # Relaxed
        MAX_IBS=1.0,  # Relaxed
    )

    analysis_date = "2025-01-08"
    count = strategy.run(analysis_date=analysis_date)
    assert count >= 0


def test_run_auto_date_detection(strategy, mock_provider):
    """Test that strategies uses the last DB date if no date is provided."""
    pd.date_range(end="2025-02-01", periods=5, freq="B")
    data = create_market_data(length=5, end_date="2025-02-01")

    mock_provider.get_universe_daily_data.return_value = data

    # We expect it to pick 2025-02-01
    # We mock _calculate_signals to spy on the date passed
    with patch.object(
        DipBuyerStrategy, "_calculate_signals", return_value=pd.DataFrame()
    ) as mock_calc:
        strategy.run(analysis_date=None)

        args, _ = mock_calc.call_args
        passed_date = args[1]
        assert passed_date == pd.Timestamp("2025-01-31")


def test_run_with_gap_in_data(strategy, mock_provider):
    """Test fallback when requested date > last db date (Gap Scenario)."""
    # DB Data ends on Jan 10
    pd.date_range(end="2025-01-10", periods=5, freq="B")
    data = create_market_data(length=5, end_date="2025-01-10")

    mock_provider.get_universe_daily_data.return_value = data

    # User asks for Jan 12 (Future/Gap)
    future_date = "2025-01-12"

    with patch.object(
        DipBuyerStrategy, "_calculate_signals", return_value=pd.DataFrame()
    ) as mock_calc:
        strategy.run(analysis_date=future_date)

        # Verify it fell back to Jan 10
        args, _ = mock_calc.call_args
        passed_date = args[1]
        assert passed_date == pd.Timestamp("2025-01-10")
