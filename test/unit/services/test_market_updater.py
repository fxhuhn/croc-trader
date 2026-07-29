from unittest.mock import MagicMock, patch

import pandas as pd

from app.database.repositories.market import MarketRepository
from app.database.session import DatabaseSession
from app.services.market.provider import YahooDataProvider
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
