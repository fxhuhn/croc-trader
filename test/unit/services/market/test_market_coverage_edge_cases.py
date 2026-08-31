"""Unit tests targeting 100% test coverage for app/services/market modules."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.services.market.quality import MarketQualityService
from app.services.market.tv_provider import TradingViewDataProvider
from app.services.market.updater import MarketDataUpdater


@pytest.fixture
def mock_market_repo() -> MagicMock:
    return MagicMock(spec=MarketRepository)


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_updater(mock_market_repo: MagicMock, mock_trade_repo: MagicMock) -> MagicMock:
    updater = MagicMock(spec=MarketDataUpdater)
    updater.repo = mock_market_repo
    updater.trade_repository = mock_trade_repo
    return updater


# --- TradingViewDataProvider Tests ---


@patch("app.services.market.tv_provider.time.sleep")
@patch("app.services.market.tv_provider.TvDatafeed")
def test_tv_provider_download_exception_resets_connection(
    mock_tv_cls: MagicMock, mock_sleep: MagicMock
) -> None:
    """Tests fetch_symbol_history resets _tv to None when an exception occurs."""
    mock_instance = MagicMock()
    mock_instance.get_hist.side_effect = RuntimeError("Connection dropped")
    mock_tv_cls.return_value = mock_instance

    provider = TradingViewDataProvider(retry_delay_seconds=0.0)
    result = provider.fetch_symbol_history("AAPL", number_of_bars=15)
    assert result == []
    assert provider._tv is None


def test_tv_provider_standardize_dataframe_rename_map() -> None:
    """Tests _standardize_dataframe_records handles index/datetime renaming."""
    provider = TradingViewDataProvider()
    df = pd.DataFrame([{"close": 150.0}], index=pd.to_datetime(["2026-02-01"]))
    records = provider._standardize_dataframe_records(df, "AAPL")
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["symbol"] == "AAPL"


# --- MarketQualityService Tests ---


def test_quality_service_repair_candidates(
    mock_market_repo: MagicMock, mock_updater: MagicMock
) -> None:
    """Tests perform_gap_check triggers update for outdated/shallow symbols."""
    service = MarketQualityService(mock_updater)

    mock_market_repo.get_outdated_symbols.return_value = ["AAPL"]
    mock_market_repo.get_symbols_with_missing_history.side_effect = [
        ["MSFT"],  # initial call
        ["MSFT"],  # post-repair check
    ]

    service.perform_gap_check()

    mock_updater.run_update.assert_called_once()
    assert set(mock_updater.run_update.call_args[1]["specific_symbols"]) == {
        "AAPL",
        "MSFT",
    }


def test_quality_service_repair_exception(
    mock_market_repo: MagicMock, mock_updater: MagicMock
) -> None:
    """Tests perform_gap_check handles repository exceptions gracefully."""
    service = MarketQualityService(mock_updater)
    mock_market_repo.get_outdated_symbols.side_effect = RuntimeError("DB error")

    # Should not raise exception
    service.perform_gap_check()


def test_quality_service_check_completeness_active_trade_symbols_error(
    mock_market_repo: MagicMock, mock_updater: MagicMock
) -> None:
    """Tests check_last_trading_day_completeness handles trade repo exception."""
    telegram_bot = MagicMock()
    telegram_bot.send_message.side_effect = Exception("Telegram error")
    service = MarketQualityService(mock_updater, telegram_bot=telegram_bot)

    mock_market_repo.get_outdated_symbols.return_value = ["QQQ"]
    mock_market_repo.get_all_known_symbols.return_value = ["QQQ", "SPY"]
    mock_updater.trade_repository.get_by_status.side_effect = RuntimeError("Repo error")

    # Should log warning, attempt telegram send (which catches exception), and return False
    assert not service.check_last_trading_day_completeness()
    telegram_bot.send_message.assert_called_once()


# --- MarketDataUpdater Tests ---


def test_updater_no_symbols(mock_market_repo: MagicMock) -> None:
    """Tests run_update logs warning and exits when symbols list is empty."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        mock_market_repo.get_ignored_symbols.return_value = set()
        updater = MarketDataUpdater(session)
        with patch.object(updater, "_get_symbols_to_process", return_value=[]):
            updater.run_update()


def test_updater_batch_exception(mock_market_repo: MagicMock) -> None:
    """Tests run_update catches batch processing exceptions."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        updater = MarketDataUpdater(session)
        with patch.object(updater, "_get_symbols_to_process", return_value=["AAPL"]):
            with patch.object(
                updater, "_process_batch", side_effect=RuntimeError("Batch failed")
            ):
                updater.run_update()


def test_updater_process_batch_yahoo_mode_empty(mock_market_repo: MagicMock) -> None:
    """Tests _process_batch in strictly 'yahoo' mode when provider returns empty df."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        updater = MarketDataUpdater(session)
        updater.provider = MagicMock()
        updater.provider.fetch_batch_raw.return_value = (pd.DataFrame(), ["AAPL"])
        count = updater._process_batch(
            ["AAPL"], "2021-01-01", full_reload=True, provider_mode="yahoo"
        )
        assert count == 0
        mock_market_repo.ignore_symbol.assert_called_with(
            "AAPL", "No Data (Full Reload)"
        )


def test_quality_service_check_completeness_empty_symbols(
    mock_market_repo: MagicMock, mock_updater: MagicMock
) -> None:
    """Tests check_last_trading_day_completeness returns True when all_symbols is empty."""
    service = MarketQualityService(mock_updater)
    mock_market_repo.get_outdated_symbols.return_value = []
    mock_market_repo.get_all_known_symbols.return_value = []
    assert service.check_last_trading_day_completeness() is True


def test_updater_signals_session_fallback_and_get_symbols(
    mock_market_repo: MagicMock,
) -> None:
    """Tests MarketDataUpdater fallback when signals_session is None and _get_symbols_to_process with specific."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        mock_market_repo.get_ignored_symbols.return_value = set()
        updater = MarketDataUpdater(session, signals_session=None)
        assert updater.trade_repository is not None

        symbols = updater._get_symbols_to_process(["AAPL", "MSFT"])
        assert set(symbols) == {"AAPL", "MSFT"}


def test_updater_process_batch_yahoo_mode_failures(
    mock_market_repo: MagicMock,
) -> None:
    """Tests _process_batch in yahoo mode with failures and full_reload=True."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        updater = MarketDataUpdater(session)
        updater.provider = MagicMock()

        df_batch = pd.DataFrame([{"close": 150.0}])
        updater.provider.fetch_batch_raw.return_value = (df_batch, ["FAIL_SYM"])
        df_sym = pd.DataFrame([{"close": 150.0}])
        updater.provider.extract_symbol_data.return_value = df_sym

        count = updater._process_batch(
            ["AAPL", "FAIL_SYM"], "2021-01-01", full_reload=True, provider_mode="yahoo"
        )
        assert count == 1
        mock_market_repo.ignore_symbol.assert_called_with(
            "FAIL_SYM", "No Data (Full Reload)"
        )


def test_updater_process_batch_dropna_empty_and_value_error(
    mock_market_repo: MagicMock,
) -> None:
    """Tests _process_batch when df_sym is empty after dropna or MarketPrice raises ValueError."""
    session = MagicMock()
    with patch(
        "app.services.market.updater.MarketRepository", return_value=mock_market_repo
    ):
        updater = MarketDataUpdater(session)
        updater.provider = MagicMock()

        # df with only NaN close
        df_batch = pd.DataFrame([{"close": None}])
        updater.provider.fetch_batch_raw.return_value = (df_batch, [])
        df_sym = pd.DataFrame([{"close": None}])
        updater.provider.extract_symbol_data.return_value = df_sym

        with patch.object(updater, "_fallback_to_tradingview", return_value=0):
            count = updater._process_batch(
                ["AAPL"], "2021-01-01", full_reload=False, provider_mode="auto"
            )
            assert count == 0


def test_updater_fallback_to_tradingview_empty_list() -> None:
    """Tests _fallback_to_tradingview returns 0 when failed_symbols list is empty."""
    session = MagicMock()
    updater = MarketDataUpdater(session)
    assert updater._fallback_to_tradingview([], full_reload=False) == 0
