"""Unit tests for app/services/market/provider.py YahooDataProvider and require_lock."""

import errno
import sqlite3
from unittest.mock import patch

import multitasking
import pandas as pd
import pytest

from app.services.market.provider import (
    ProviderInfrastructureError,
    YahooDataProvider,
    _provider_lock,
    require_lock,
)


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
        assert kwargs.get("threads") is False


@pytest.mark.parametrize(
    "fatal_error",
    [
        RuntimeError("can't start new thread"),
        MemoryError("Unable to allocate 3.13 MiB"),
        sqlite3.OperationalError("disk I/O error"),
        OSError(errno.ENOMEM, "Cannot allocate memory"),
    ],
)
def test_fetch_batch_raw_raises_on_infrastructure_error(
    fatal_error: BaseException,
) -> None:
    """Regression: Systemic OS/thread/memory errors raise ProviderInfrastructureError."""
    provider = YahooDataProvider()
    with (
        patch("yfinance.download", side_effect=fatal_error),
        pytest.raises(ProviderInfrastructureError),
    ):
        provider.fetch_batch_raw(["AAPL", "MSFT"], "2026-01-01")


def test_fetch_batch_raw_timeout_is_not_infrastructure_error() -> None:
    """Network timeouts return failed symbols without raising ProviderInfrastructureError."""
    provider = YahooDataProvider()
    with patch("yfinance.download", side_effect=TimeoutError("Read timed out")):
        dataframe, failed = provider.fetch_batch_raw(["AAPL"], "2026-01-01")
        assert dataframe.empty
        assert failed == ["AAPL"]


def test_fetch_batch_raw_does_not_spawn_multitasking_tasks() -> None:
    """Regression: threads=False leaves multitasking global task list empty."""
    multitasking.get_list_of_tasks().clear()
    provider = YahooDataProvider()

    def fake_single_download(ctx: object, ticker: str, **kwargs: object) -> None:
        index = pd.to_datetime(["2026-08-01"])
        ctx.dfs[ticker] = pd.DataFrame(  # type: ignore[attr-defined]
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000],
            },
            index=index,
        )

    with patch("yfinance.multi._download_one", side_effect=fake_single_download):
        dataframe, failed = provider.fetch_batch_raw(["AAPL", "MSFT"], "2026-08-01")

    assert not dataframe.empty
    assert failed == []
    assert len(multitasking.get_list_of_tasks()) == 0


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
