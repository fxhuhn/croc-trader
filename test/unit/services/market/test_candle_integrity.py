"""Unit tests for candle integrity checking, anomaly detection, and TradingView fallback."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models import CandleIntegrityError, MarketPrice
from app.services.market.quality import MarketQualityService
from app.services.market.updater import MarketDataUpdater


def test_market_price_valid_candle() -> None:
    """Verifies that mathematically valid candles construct without errors."""
    row = {
        "date": "2026-09-23",
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 1000000,
    }
    price = MarketPrice.from_yahoo("AAPL", row)
    assert price.symbol == "AAPL"
    assert price.open == 100.0
    assert price.high == 105.0
    assert price.low == 98.0
    assert price.close == 102.0
    assert price.volume == 1000000


def test_market_price_geometric_invariants() -> None:
    """Verifies that candles violating geometric invariants raise CandleIntegrityError."""
    # 1. Low > Close (the PAYX 23.09 bug: Low 103.25, Close 96.00)
    payx_corrupt = {
        "date": "2026-09-23",
        "open": 114.73,
        "high": 115.11,
        "low": 103.25,
        "close": 96.00,
        "volume": 500000,
    }
    with pytest.raises(CandleIntegrityError, match="Low .* > min"):
        MarketPrice.from_yahoo("PAYX", payx_corrupt)

    # 2. High < Open or Close
    high_too_low = {
        "date": "2026-09-23",
        "open": 100.0,
        "high": 95.0,
        "low": 90.0,
        "close": 98.0,
        "volume": 10000,
    }
    with pytest.raises(CandleIntegrityError, match="High .* < max"):
        MarketPrice.from_yahoo("TEST", high_too_low)

    # 3. High < Low
    inverted = {
        "date": "2026-09-23",
        "open": 50.0,
        "high": 40.0,
        "low": 60.0,
        "close": 50.0,
        "volume": 10000,
    }
    with pytest.raises(CandleIntegrityError, match="High .* < Low"):
        MarketPrice.from_yahoo("TEST", inverted)

    # 4. Non-positive price
    zero_price = {
        "date": "2026-09-23",
        "open": 0.0,
        "high": 10.0,
        "low": 0.0,
        "close": 10.0,
        "volume": 100,
    }
    with pytest.raises(CandleIntegrityError, match="Non-positive price"):
        MarketPrice.from_yahoo("TEST", zero_price)

    # 5. Negative volume
    neg_vol = {
        "date": "2026-09-23",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": -100,
    }
    with pytest.raises(CandleIntegrityError, match="Negative volume"):
        MarketPrice.from_yahoo("TEST", neg_vol)


def test_market_price_anomalous_wick_detection() -> None:
    """Verifies that odd-lot flash-crash spikes are accurately detected."""
    # 1. PAYX on 22.09: Open 114.36, High 115.00, Low 88.85, Close 113.88
    # Lower wick is 25.03$ (~22% drop), body is only 0.48$ -> clear bad print anomaly
    assert MarketPrice.is_anomalous_wick(114.36, 115.00, 88.85, 113.88) is True

    # 2. ABNB on 22.09: Open 161.42, High 162.74, Low 126.30, Close 161.73
    # Lower wick is 35.12$ (~21.7% drop), body is only 0.31$ -> clear bad print anomaly
    assert MarketPrice.is_anomalous_wick(161.42, 162.74, 126.30, 161.73) is True

    # 3. GEN on 22.09: Open 27.20, High 27.40, Low 23.29, Close 27.28
    # Lower wick is 3.91$ (~14.3% drop), body is only 0.08$ -> bad print anomaly
    assert MarketPrice.is_anomalous_wick(27.20, 27.40, 23.29, 27.28) is True

    # 4. Normal trading day: Open 100, High 102, Low 98, Close 101
    assert MarketPrice.is_anomalous_wick(100.0, 102.0, 98.0, 101.0) is False

    # 5. Genuine market selloff day (closes near low, not an odd-lot wick needle):
    # Open 100, High 100, Low 85, Close 85 (15% drop, closed at the low -> body = 15$)
    assert MarketPrice.is_anomalous_wick(100.0, 100.0, 85.0, 85.0) is False


def test_updater_extract_symbol_market_prices_integrity_flag() -> None:
    """Verifies that _extract_symbol_market_prices flags corrupt candles."""
    session_mock = MagicMock()
    updater = MarketDataUpdater(session_mock)

    # Corrupt DataFrame with a PAYX-style impossible candle (Low > Close)
    corrupt_df = pd.DataFrame(
        [
            {
                "date": "2026-09-22",
                "open": 114.0,
                "high": 115.0,
                "low": 113.0,
                "close": 114.5,
                "volume": 1000,
            },
            {
                "date": "2026-09-23",
                "open": 114.73,
                "high": 115.11,
                "low": 103.25,
                "close": 96.00,
                "volume": 1000,
            },
        ]
    ).set_index("date")

    prices, max_date, has_error = updater._extract_symbol_market_prices(
        corrupt_df, "PAYX", ignore_today=False, today_str="2026-09-24"
    )

    assert has_error is True
    # The valid row was parsed, but the corrupt row was flagged
    assert len(prices) == 1
    assert max_date == "2026-09-22"


def test_updater_process_batch_triggers_tradingview_on_integrity_error() -> None:
    """Verifies that _process_batch excludes corrupt Yahoo data and queues TradingView fallback."""
    session_mock = MagicMock()
    updater = MarketDataUpdater(session_mock)

    # Mock provider returning batch DataFrame with one corrupt symbol
    raw_df = pd.DataFrame(
        {
            ("Close", "PAYX"): [96.00],
            ("Open", "PAYX"): [114.73],
            ("High", "PAYX"): [115.11],
            ("Low", "PAYX"): [103.25],  # Low > Close!
            ("Volume", "PAYX"): [1000],
        },
        index=pd.to_datetime(["2026-09-23"]),
    )

    extract_df = pd.DataFrame(
        [
            {
                "date": "2026-09-23",
                "open": 114.73,
                "high": 115.11,
                "low": 103.25,
                "close": 96.00,
                "volume": 1000,
            }
        ]
    ).set_index("date")

    with (
        patch.object(updater.provider, "fetch_batch_raw", return_value=(raw_df, [])),
        patch.object(updater.provider, "extract_symbol_data", return_value=extract_df),
        patch.object(updater.repo, "save_bulk_prices") as mock_save,
        patch.object(
            updater, "_fallback_to_tradingview", return_value=30
        ) as mock_fallback,
    ):
        saved_count = updater._process_batch(
            batch=["PAYX"],
            start_date="2026-09-15",
            full_reload=False,
            provider_mode="auto",
        )

        # Flawed Yahoo prices must NOT be saved to DB directly
        mock_save.assert_not_called()
        # TradingView fallback must be triggered for PAYX
        mock_fallback.assert_called_once_with(["PAYX"], False, ignore_today=False)
        assert saved_count == 30


def test_market_quality_check_and_repair_corrupt_candles() -> None:
    """Verifies that check_and_repair_corrupt_candles triggers targeted TradingView repair."""
    updater_mock = MagicMock()
    repo_mock = MagicMock()
    updater_mock.repo = repo_mock
    telegram_mock = MagicMock()

    service = MarketQualityService(updater_mock, telegram_bot=telegram_mock)

    # Initial query finds corrupt symbols, second query finds 0 after repair
    repo_mock.get_corrupt_candle_symbols.side_effect = [["PAYX", "ABNB"], []]

    repaired = service.check_and_repair_corrupt_candles(lookback_days=5)

    assert repaired == ["ABNB", "PAYX"]
    # Verify updater called with TradingView provider mode
    updater_mock.run_update.assert_called_once_with(
        full_reload=False,
        specific_symbols=["PAYX", "ABNB"],
        provider_mode="tradingview",
    )
    # Verify Telegram notification dispatched
    telegram_mock.send_message.assert_called_once()
    assert "Kerzen-Integritätsreparatur" in telegram_mock.send_message.call_args[0][0]
