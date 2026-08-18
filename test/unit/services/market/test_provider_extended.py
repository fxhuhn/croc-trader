"""Unit tests for app/services/market/provider.py YahooDataProvider and require_lock."""

from unittest.mock import patch

import pandas as pd

from app.services.market.provider import YahooDataProvider, _provider_lock, require_lock


def test_require_lock_skip_when_busy() -> None:
    """Tests that require_lock skips execution and returns None when lock is held."""

    @require_lock
    def dummy_func() -> str:
        return "done"

    with _provider_lock:
        res = dummy_func()
        assert res is None


def test_require_lock_success() -> None:
    """Tests normal execution of function decorated with require_lock."""

    @require_lock
    def dummy_func() -> str:
        return "done"

    assert dummy_func() == "done"


def test_fetch_batch_raw_empty_symbols() -> None:
    """Tests fetch_batch_raw with empty list of symbols."""
    provider = YahooDataProvider()
    df, failed = provider.fetch_batch_raw([], "2026-01-01")
    assert df.empty
    assert failed == []


def test_fetch_batch_raw_exception() -> None:
    """Tests fetch_batch_raw handling of yfinance Exception."""
    provider = YahooDataProvider()
    with patch("yfinance.download", side_effect=RuntimeError("API error")):
        df, failed = provider.fetch_batch_raw(["AAPL"], "2026-01-01")
        assert df.empty
        assert failed == ["AAPL"]


def test_fetch_batch_raw_empty_result() -> None:
    """Tests fetch_batch_raw handling when yfinance returns empty DataFrame."""
    provider = YahooDataProvider()
    with patch("yfinance.download", return_value=pd.DataFrame()):
        df, failed = provider.fetch_batch_raw(["AAPL"], "2026-01-01")
        assert df.empty
        assert failed == ["AAPL"]


def test_fetch_batch_raw_single_index() -> None:
    """Tests fetch_batch_raw with SingleIndex DataFrame (single symbol)."""
    provider = YahooDataProvider()
    dummy_df = pd.DataFrame({"close": [150.0]})
    with patch("yfinance.download", return_value=dummy_df) as mock_download:
        df, failed = provider.fetch_batch_raw(["AAPL"], "2026-01-01")
        assert not df.empty
        assert failed == []
        mock_download.assert_called_once()
        _, kwargs = mock_download.call_args
        assert kwargs.get("repair") is True


def test_extract_symbol_data_cases() -> None:
    """Tests extract_symbol_data with empty DF, MultiIndex, and SingleIndex."""
    provider = YahooDataProvider()

    # Empty DF
    assert provider.extract_symbol_data(pd.DataFrame(), "AAPL").empty

    # MultiIndex - symbol present vs missing
    columns = pd.MultiIndex.from_tuples([("AAPL", "close"), ("MSFT", "close")])
    multi_df = pd.DataFrame([[150.0, 300.0]], columns=columns)

    aapl_df = provider.extract_symbol_data(multi_df, "AAPL")
    assert not aapl_df.empty
    assert provider.extract_symbol_data(multi_df, "GOOG").empty

    # SingleIndex with 'close' column
    flat_close_df = pd.DataFrame({"close": [150.0]})
    assert not provider.extract_symbol_data(flat_close_df, "AAPL").empty

    # SingleIndex without 'close' column
    flat_other_df = pd.DataFrame({"volume": [1000]})
    assert provider.extract_symbol_data(flat_other_df, "AAPL").empty
