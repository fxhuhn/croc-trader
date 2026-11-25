import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.market import MarketRepository
from app.database.session import DatabaseSession
from app.services.market.provider import ProviderInfrastructureError, YahooDataProvider
from app.services.market.tv_provider import TradingViewDataProvider
from app.services.market.updater import MarketDataUpdater


def test_yahoo_data_provider_fetch_batch_raw_finds_failures() -> None:
    """Tests that fetch_batch_raw identifies missing symbols from download columns."""
    provider = YahooDataProvider()

    # Create a MultiIndex DataFrame mock with AAPL and MSFT only (PSTG is missing)
    columns = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "Open"),
            ("AAPL", "Close"),
            ("MSFT", "Open"),
            ("MSFT", "Close"),
        ]
    )
    mock_df = pd.DataFrame([[100, 101, 200, 201]], columns=columns)

    with patch("yfinance.download", return_value=mock_df) as mock_download:
        df, failed = provider.fetch_batch_raw(["AAPL", "PSTG", "MSFT"], "2026-06-01")

        assert mock_download.called
        assert not df.empty
        assert failed == ["PSTG"]


def test_market_updater_process_batch_blacklists_failed_symbols_on_full_reload() -> (
    None
):
    """Tests that failures from fetch_batch_raw are blacklisted if full_reload is True."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)

    # Mock the provider, tv_provider, and repository
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = []
    updater.repo = MagicMock(spec=MarketRepository)

    # fetch_batch_raw returns raw_failures = ["PSTG"]
    mock_df = pd.DataFrame()  # Empty, meaning the batch is empty
    updater.provider.fetch_batch_raw.return_value = (mock_df, ["PSTG"])

    updater._process_batch(["PSTG"], "2026-06-01", full_reload=True)

    # Verify that PSTG was blacklisted
    updater.repo.ignore_symbol.assert_called_once_with(
        "PSTG", "No Data (Yahoo & TradingView)"
    )


def test_market_updater_process_batch_does_not_blacklist_on_incremental() -> None:
    """Tests that failures from fetch_batch_raw are NOT blacklisted if full_reload is False."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)

    # Mock the provider, tv_provider, and repository
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = []
    updater.repo = MagicMock(spec=MarketRepository)

    # fetch_batch_raw returns raw_failures = ["PSTG"]
    mock_df = pd.DataFrame()
    updater.provider.fetch_batch_raw.return_value = (mock_df, ["PSTG"])

    updater._process_batch(["PSTG"], "2026-06-01", full_reload=False)

    # Verify that no symbols were blacklisted
    updater.repo.ignore_symbol.assert_not_called()


def test_market_updater_process_batch_identifies_empty_data_as_failure() -> None:
    """Tests that symbols yielding empty DataFrames after cleaning are identified as failures."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)

    # Mock the provider, tv_provider, and repository
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = []
    updater.repo = MagicMock(spec=MarketRepository)

    # We download AAPL and PSTG. yfinance returns a MultiIndex df.
    # AAPL has data. PSTG is present in columns but its Close column is all NaNs.
    columns = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "Close"),
            ("PSTG", "Close"),
        ]
    )
    mock_batch_df = pd.DataFrame([[100.0, None]], columns=columns)

    updater.provider.fetch_batch_raw.return_value = (mock_batch_df, [])

    # Mock extract_symbol_data behavior
    def mock_extract(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if symbol == "AAPL":
            return pd.DataFrame({"close": [100.0]})
        elif symbol == "PSTG":
            return pd.DataFrame({"close": [None]})
        return pd.DataFrame()

    updater.provider.extract_symbol_data.side_effect = mock_extract

    # Run the batch process with full_reload = True
    updater._process_batch(["AAPL", "PSTG"], "2026-06-01", full_reload=True)

    # Verify that PSTG was blacklisted because its close column was all NaN
    updater.repo.ignore_symbol.assert_called_once_with(
        "PSTG", "No Data (Yahoo & TradingView)"
    )


def test_market_updater_run_update_full_reload_clears_ignored_symbols() -> None:
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.repo.get_ignored_symbols.return_value = set()
    updater.repo.get_all_known_symbols.return_value = ["AAPL"]
    updater.trade_repository = MagicMock()
    updater.trade_repository.get_all_traded_symbols.return_value = []
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.provider.fetch_batch_raw.return_value = (pd.DataFrame(), [])
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = []

    with patch("app.services.market.updater.ExchangeSymbol") as mock_exchange:
        mock_exchange.return_value.all = ["AAPL"]
        updater.run_update(full_reload=True, specific_symbols=None)

    updater.repo.clear_ignored_symbols.assert_called_once()


def test_market_updater_provider_mode_tradingview() -> None:
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = [
        {
            "date": "2026-08-01",
            "open": 100,
            "high": 105,
            "low": 99,
            "close": 102,
            "volume": 1000,
        }
    ]

    count = updater._process_batch(
        ["AAPL"], "2026-08-01", full_reload=False, provider_mode="tradingview"
    )

    assert count == 1
    updater.repo.save_bulk_prices.assert_called_once()


def test_market_updater_ignore_today_skips_today_bars() -> None:
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = []

    today_str = datetime.now().strftime("%Y-%m-%d")

    columns = pd.MultiIndex.from_tuples([("AAPL", "Close")])
    mock_batch_df = pd.DataFrame(
        [[150.0]], index=[pd.Timestamp(today_str)], columns=columns
    )

    updater.provider.fetch_batch_raw.return_value = (mock_batch_df, [])

    mock_sym_df = pd.DataFrame({"close": [150.0]}, index=[pd.Timestamp(today_str)])
    updater.provider.extract_symbol_data.return_value = mock_sym_df

    count = updater._process_batch(
        ["AAPL"],
        "2026-08-01",
        full_reload=False,
        provider_mode="yahoo",
        ignore_today=True,
    )

    assert count == 0


def test_market_updater_aborts_without_tv_fallback_on_infrastructure_error() -> None:
    """Regression: Thread/OS exhaustion aborts update without triggering TV storm."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.repo.get_ignored_symbols.return_value = set()
    updater.provider = MagicMock(spec=YahooDataProvider)
    updater.provider.fetch_batch_raw.side_effect = ProviderInfrastructureError(
        "can't start new thread"
    )
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)

    with (
        patch("app.services.market.updater.BATCH_SIZE", 1),
        pytest.raises(ProviderInfrastructureError, match="can't start new thread"),
    ):
        updater.run_update(specific_symbols=["AAPL", "MSFT"])

    updater.tv_provider.fetch_symbol_history.assert_not_called()
    assert updater.provider.fetch_batch_raw.call_count == 1


def test_fallback_to_tradingview_batches_persistence_and_isolates_symbol_error() -> (
    None
):
    """Regression: TV fallback isolates single-symbol errors and writes in one batch."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.repo.delete_superseded_prices.return_value = 2
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)

    def fake_tv_history(
        symbol: str, number_of_bars: int = 15
    ) -> list[dict[str, object]]:
        if symbol == "BROKEN":
            raise RuntimeError("Malformed websocket frame")
        if symbol == "NODATA":
            return []
        return [
            {
                "date": "2026-08-01",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000,
            }
        ]

    updater.tv_provider.fetch_symbol_history.side_effect = fake_tv_history

    saved = updater._fallback_to_tradingview(
        ["AAPL", "BROKEN", "MSFT", "NODATA"],
        full_reload=True,
        ignore_today=False,
    )

    assert saved == 2
    updater.repo.save_bulk_prices.assert_called_once()
    persisted_records = updater.repo.save_bulk_prices.call_args[0][0]
    assert [item.symbol for item in persisted_records] == ["AAPL", "MSFT"]
    updater.repo.delete_superseded_prices.assert_called_once_with(
        [("AAPL", "2026-08-01"), ("MSFT", "2026-08-01")],
        provider="yahoo",
    )
    updater.repo.ignore_symbol.assert_called_once_with(
        "NODATA", "No Data (Yahoo & TradingView)"
    )


def test_fallback_to_tradingview_propagates_sqlite_operational_error() -> None:
    """Regression: SQLite disk I/O error during batch persistence propagates immediately."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.repo.save_bulk_prices.side_effect = sqlite3.OperationalError(
        "disk I/O error"
    )
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    updater.tv_provider.fetch_symbol_history.return_value = [
        {
            "date": "2026-08-01",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 1000,
        }
    ]

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        updater._fallback_to_tradingview(["AAPL"], full_reload=False)


def test_fallback_to_tradingview_filters_today_invalid_rows_and_propagates_memory_error() -> (
    None
):
    """Covers ignore_today filtering, malformed row skipping, and MemoryError propagation."""
    session = MagicMock(spec=DatabaseSession)
    updater = MarketDataUpdater(session)
    updater.repo = MagicMock(spec=MarketRepository)
    updater.tv_provider = MagicMock(spec=TradingViewDataProvider)
    today_str = datetime.now().strftime("%Y-%m-%d")

    updater.tv_provider.fetch_symbol_history.return_value = [
        {
            "date": today_str,
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 1000,
        },
        {
            "date": "2026-08-01",
            "open": -5.0,
            "high": 105.0,
            "low": 99.0,
            "close": 102.0,
            "volume": 1000,
        },
    ]

    saved = updater._fallback_to_tradingview(
        ["AAPL"], full_reload=True, ignore_today=True
    )
    assert saved == 0
    updater.repo.ignore_symbol.assert_called_once_with(
        "AAPL", "No Data (Yahoo & TradingView)"
    )

    updater.tv_provider.fetch_symbol_history.side_effect = MemoryError("OOM")
    with pytest.raises(MemoryError, match="OOM"):
        updater._fallback_to_tradingview(["AAPL"], full_reload=False)
