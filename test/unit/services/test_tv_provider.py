from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models import MarketPrice
from app.services.market.tv_provider import TradingViewDataProvider


def test_market_price_from_tradingview():
    row = {
        "date": "2026-07-29",
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 5000,
    }
    price = MarketPrice.from_tradingview("BRK-B", row)

    assert price.symbol == "BRK-B"
    assert price.date == "2026-07-29"
    assert price.close == 104.0
    assert price.provider == "tradingview"
    assert price.timeframe == "1D"


def test_market_price_from_tradingview_invalid_close():
    row = {"date": "2026-07-29", "close": -1.0}
    with pytest.raises(ValueError, match="Negative close price"):
        MarketPrice.from_tradingview("AAPL", row)


def test_tv_provider_missing_exchange_warning(caplog):
    provider = TradingViewDataProvider()

    with caplog.at_level("WARNING"):
        records = provider.fetch_symbol_history("NONEXISTENT_SYMBOL_XYZ_123")

    assert records == []
    assert (
        "TradingView returned empty data for NONEXISTENT_SYMBOL_XYZ_123" in caplog.text
    )


@patch("app.services.market.tv_provider.mapper")
def test_tv_provider_fetch_symbol_history_success(mock_mapper):
    mock_mapper.get_exchange.return_value = "NYSE"

    dummy_df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [104.0],
            "volume": [5000],
        },
        index=pd.DatetimeIndex(["2026-07-29"]),
    )

    mock_tv_instance = MagicMock()
    mock_tv_instance.get_hist.return_value = dummy_df

    provider = TradingViewDataProvider()
    provider._tv = mock_tv_instance

    records = provider.fetch_symbol_history("BRK-B", n_bars=10)

    # Check that tv_symbol was passed as BRK.B
    mock_tv_instance.get_hist.assert_called_once()
    _, kwargs = mock_tv_instance.get_hist.call_args
    assert kwargs["symbol"] == "BRK.B"
    assert kwargs["exchange"] == "NYSE"

    assert len(records) == 1
    assert records[0]["symbol"] == "BRK-B"  # Mapped back to standard Yahoo format
    assert records[0]["close"] == 104.0


def test_market_price_from_tradingview_datetime_key():
    """Verifies that MarketPrice parses 'datetime' key when 'date' is absent in TradingView records."""
    row = {
        "datetime": pd.Timestamp("2026-07-20 15:30:00"),
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 5000,
    }
    price = MarketPrice.from_tradingview("BRK-B", row)
    assert price.date == "2026-07-20"


@patch("app.services.market.tv_provider.mapper")
def test_tv_provider_multi_bar_date_preservation(mock_mapper):
    """Verifies that multi-bar fetches preserve distinct dates across all bars."""
    mock_mapper.get_exchange.return_value = "NASDAQ"

    dates = ["2026-07-20", "2026-07-21", "2026-07-22"]
    dummy_df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [5000, 6000, 7000],
        },
        index=pd.DatetimeIndex(dates, name="datetime"),
    )

    mock_tv_instance = MagicMock()
    mock_tv_instance.get_hist.return_value = dummy_df

    provider = TradingViewDataProvider()
    provider._tv = mock_tv_instance

    records = provider.fetch_symbol_history("AAPL", n_bars=3)
    prices = [MarketPrice.from_tradingview("AAPL", r) for r in records]

    parsed_dates = [p.date for p in prices]
    assert parsed_dates == ["2026-07-20", "2026-07-21", "2026-07-22"]
    assert len(set(parsed_dates)) == 3
