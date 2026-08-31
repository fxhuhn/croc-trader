from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models import MarketPrice
from app.services.market.tv_provider import TradingViewDataProvider


def test_market_price_from_tradingview() -> None:
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


def test_market_price_from_tradingview_invalid_close() -> None:
    row = {"date": "2026-07-29", "close": -1.0}
    with pytest.raises(ValueError, match="Negative close price"):
        MarketPrice.from_tradingview("AAPL", row)


@patch("app.services.market.tv_provider.time.sleep")
@patch("app.services.market.tv_provider.TvDatafeed")
def test_tv_provider_missing_exchange_warning(
    mock_tv_cls: MagicMock,
    mock_sleep: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_instance = MagicMock()
    mock_instance.get_hist.return_value = None
    mock_tv_cls.return_value = mock_instance

    provider = TradingViewDataProvider(retry_delay_seconds=0.0)

    with caplog.at_level("WARNING"):
        records = provider.fetch_symbol_history("NONEXISTENT_SYMBOL_XYZ_123")

    assert records == []
    assert (
        "TradingView returned empty data for NONEXISTENT_SYMBOL_XYZ_123" in caplog.text
    )


@patch("app.services.market.tv_provider.mapper")
def test_tv_provider_fetch_symbol_history_success(
    mock_mapper: MagicMock,
) -> None:
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

    records = provider.fetch_symbol_history("BRK-B", number_of_bars=10)

    # Check that tv_symbol was passed as BRK.B
    mock_tv_instance.get_hist.assert_called_once()
    _, kwargs = mock_tv_instance.get_hist.call_args
    assert kwargs["symbol"] == "BRK.B"
    assert kwargs["exchange"] == "NYSE"

    assert len(records) == 1
    assert records[0]["symbol"] == "BRK-B"  # Mapped back to standard Yahoo format
    assert records[0]["close"] == 104.0


def test_market_price_from_tradingview_datetime_key() -> None:
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
def test_tv_provider_multi_bar_date_preservation(mock_mapper: MagicMock) -> None:
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

    records = provider.fetch_symbol_history("AAPL", number_of_bars=3)
    prices = [MarketPrice.from_tradingview("AAPL", r) for r in records]

    parsed_dates = [p.date for p in prices]
    assert parsed_dates == ["2026-07-20", "2026-07-21", "2026-07-22"]
    assert len(set(parsed_dates)) == 3


@patch("app.services.market.tv_provider.mapper")
@patch("app.services.market.tv_provider.time.sleep")
def test_tv_provider_retry_recovers_from_transient_error(
    mock_sleep: MagicMock, mock_mapper: MagicMock
) -> None:
    """Tests that a transient exception on the first attempt is recovered by a retry."""
    mock_mapper.get_exchange.return_value = "NASDAQ"

    dummy_df = pd.DataFrame(
        {
            "open": [150.0],
            "high": [155.0],
            "low": [149.0],
            "close": [154.0],
            "volume": [10000],
        },
        index=pd.DatetimeIndex(["2026-07-29"]),
    )

    mock_tv_1 = MagicMock()
    mock_tv_1.get_hist.side_effect = RuntimeError("Connection timed out")

    mock_tv_2 = MagicMock()
    mock_tv_2.get_hist.return_value = dummy_df

    provider = TradingViewDataProvider(max_retries=2, retry_delay_seconds=0.0)

    with patch.object(provider, "_get_instance", side_effect=[mock_tv_1, mock_tv_2]):
        records = provider.fetch_symbol_history("AAPL", number_of_bars=5)

    assert len(records) == 1
    assert records[0]["symbol"] == "AAPL"
    assert records[0]["close"] == 154.0
    mock_sleep.assert_called_with(0.5)  # Rate limiting delay at end of fetch


@patch("app.services.market.tv_provider.mapper")
@patch("app.services.market.tv_provider.time.sleep")
def test_tv_provider_retry_recovers_from_empty_response(
    mock_sleep: MagicMock, mock_mapper: MagicMock
) -> None:
    """Tests that an empty None response on the first attempt is recovered by a retry."""
    mock_mapper.get_exchange.return_value = "NYSE"

    dummy_df = pd.DataFrame(
        {
            "open": [200.0],
            "high": [205.0],
            "low": [198.0],
            "close": [204.0],
            "volume": [8000],
        },
        index=pd.DatetimeIndex(["2026-07-29"]),
    )

    mock_tv_1 = MagicMock()
    mock_tv_1.get_hist.return_value = None

    mock_tv_2 = MagicMock()
    mock_tv_2.get_hist.return_value = dummy_df

    provider = TradingViewDataProvider(max_retries=2, retry_delay_seconds=0.0)

    with patch.object(provider, "_get_instance", side_effect=[mock_tv_1, mock_tv_2]):
        records = provider.fetch_symbol_history("IBM", number_of_bars=5)

    assert len(records) == 1
    assert records[0]["symbol"] == "IBM"
    assert records[0]["close"] == 204.0


@patch("app.services.market.tv_provider.mapper")
@patch("app.services.market.tv_provider.time.sleep")
def test_tv_provider_retry_exhausted_returns_empty(
    mock_sleep: MagicMock, mock_mapper: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Tests that if all retries fail, empty list is returned and warnings are logged."""
    mock_mapper.get_exchange.return_value = "NASDAQ"

    mock_tv_1 = MagicMock()
    mock_tv_1.get_hist.side_effect = RuntimeError("Remote host lost")

    mock_tv_2 = MagicMock()
    mock_tv_2.get_hist.side_effect = RuntimeError("Remote host lost")

    provider = TradingViewDataProvider(max_retries=2, retry_delay_seconds=0.0)

    with caplog.at_level("WARNING"):
        with patch.object(
            provider, "_get_instance", side_effect=[mock_tv_1, mock_tv_2]
        ):
            records = provider.fetch_symbol_history("FAIL_TICKER", number_of_bars=5)

    assert records == []
    assert "TradingView download error for symbol FAIL_TICKER" in caplog.text
    assert "after 2 attempts" in caplog.text


@patch("app.services.market.tv_provider.mapper")
@patch("app.services.market.tv_provider.time.sleep")
def test_tv_provider_custom_retry_configuration(
    mock_sleep: MagicMock, mock_mapper: MagicMock
) -> None:
    """Tests custom max_retries parameter configuration."""
    mock_mapper.get_exchange.return_value = "NASDAQ"

    dummy_df = pd.DataFrame(
        {
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000],
        },
        index=pd.DatetimeIndex(["2026-07-29"]),
    )

    mock_tv_1 = MagicMock()
    mock_tv_1.get_hist.return_value = None

    mock_tv_2 = MagicMock()
    mock_tv_2.get_hist.side_effect = RuntimeError("Socket timeout")

    mock_tv_3 = MagicMock()
    mock_tv_3.get_hist.return_value = dummy_df

    provider = TradingViewDataProvider(max_retries=3, retry_delay_seconds=0.01)

    with patch.object(
        provider, "_get_instance", side_effect=[mock_tv_1, mock_tv_2, mock_tv_3]
    ):
        records = provider.fetch_symbol_history("PENNY", number_of_bars=5)

    assert len(records) == 1
    assert records[0]["symbol"] == "PENNY"
    assert records[0]["close"] == 10.5
