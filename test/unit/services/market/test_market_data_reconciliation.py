from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from app.services.market.provider import YahooDataProvider
from app.tasks import run_active_positions_market_sync


def test_reconcile_eod_candle_corrects_post_market_drift() -> None:
    """Verifies that reconcile_eod_candle replaces unfinalized post-market close

    with official regularMarketPrice when relative drift exceeds threshold.
    """
    provider = YahooDataProvider()
    symbol_dataframe = pd.DataFrame(
        {
            "open": [211.0, 212.0],
            "high": [215.0, 220.0],
            "low": [209.0, 208.0],
            "close": [211.5, 223.76],  # Unfinalized post-market spike
            "volume": [10000, 15000],
        },
        index=pd.to_datetime(["2026-09-15", "2026-09-16"]),
    )

    mock_fast_info = {"lastPrice": 209.37}
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_ticker_class.return_value = mock_ticker

        reconciled = provider.reconcile_eod_candle(
            "NBIS", symbol_dataframe, drift_threshold=0.002
        )

        assert reconciled.iloc[-1]["close"] == 209.37
        assert reconciled.iloc[0]["close"] == 211.5


def test_reconcile_eod_candle_preserves_consistent_price() -> None:
    """Verifies that reconcile_eod_candle leaves prices unchanged when drift is within threshold."""
    provider = YahooDataProvider()
    symbol_dataframe = pd.DataFrame(
        {
            "close": [209.37, 209.40],
        },
        index=pd.to_datetime(["2026-09-15", "2026-09-16"]),
    )

    mock_fast_info = {"lastPrice": 209.37}
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info
        mock_ticker_class.return_value = mock_ticker

        reconciled = provider.reconcile_eod_candle(
            "NBIS", symbol_dataframe, drift_threshold=0.002
        )

        # Difference is ~0.014% (< 0.2%), so close remains unchanged at 209.40
        assert reconciled.iloc[-1]["close"] == 209.40


def test_reconcile_eod_candle_handles_empty_dataframe() -> None:
    """Verifies that reconcile_eod_candle handles empty DataFrame without error."""
    provider = YahooDataProvider()
    empty_dataframe = pd.DataFrame()

    reconciled = provider.reconcile_eod_candle("NBIS", empty_dataframe)
    assert reconciled.empty


def test_reconcile_eod_candle_handles_exceptions_gracefully() -> None:
    """Verifies that exceptions during fast_info lookup gracefully return original DataFrame."""
    provider = YahooDataProvider()
    symbol_dataframe = pd.DataFrame(
        {"close": [223.76]}, index=pd.to_datetime(["2026-09-16"])
    )

    with patch("yfinance.Ticker", side_effect=RuntimeError("Network error")):
        reconciled = provider.reconcile_eod_candle("NBIS", symbol_dataframe)
        assert reconciled.iloc[-1]["close"] == 223.76


def test_run_active_positions_market_sync_updates_symbols(tmp_path: Path) -> None:
    """Verifies that run_active_positions_market_sync correctly queries active trades and calls updater."""
    fake_stocks_db = tmp_path / "stocks.db"
    fake_signals_db = tmp_path / "signals.db"
    fake_stocks_db.touch()
    fake_signals_db.touch()

    active_trades = [
        {"symbol": "NBIS", "status": "ACTIVE"},
        {"symbol": "AAPL", "status": "CREATED"},
    ]

    with (
        patch("app.tasks.TradeRepository") as mock_trade_repo_class,
        patch("app.tasks.MarketDataUpdater") as mock_updater_class,
    ):
        mock_repo = MagicMock()
        mock_repo.get_active_trades.return_value = active_trades
        mock_trade_repo_class.return_value = mock_repo

        mock_updater = MagicMock()
        mock_updater_class.return_value = mock_updater

        result = run_active_positions_market_sync(fake_stocks_db)

        assert result == ["AAPL", "NBIS"]
        mock_updater.run_update.assert_called_once_with(
            specific_symbols=["AAPL", "NBIS"],
            provider_mode="auto",
            reconcile_eod=True,
        )


def test_run_active_positions_market_sync_empty_when_no_trades(
    tmp_path: Path,
) -> None:
    """Verifies that run_active_positions_market_sync returns empty list when no active trades exist."""
    fake_stocks_db = tmp_path / "stocks.db"
    fake_signals_db = tmp_path / "signals.db"
    fake_stocks_db.touch()
    fake_signals_db.touch()

    with (
        patch("app.tasks.TradeRepository") as mock_trade_repo_class,
        patch("app.tasks.MarketDataUpdater") as mock_updater_class,
    ):
        mock_repo = MagicMock()
        mock_repo.get_active_trades.return_value = []
        mock_trade_repo_class.return_value = mock_repo

        mock_updater = MagicMock()
        mock_updater_class.return_value = mock_updater

        result = run_active_positions_market_sync(fake_stocks_db)

        assert result == []
        mock_updater.run_update.assert_not_called()


def test_configure_scheduler_registers_preflight_job() -> None:
    """Verifies that configure_scheduler registers the pre-flight sync job."""
    from app.services.setup import configure_scheduler

    mock_app = MagicMock()
    mock_app.extensions = {}
    mock_config = MagicMock()
    mock_config.get_db_path.return_value = "data/stocks.db"

    with patch("app.services.setup.BackgroundScheduler") as mock_scheduler_class:
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        configure_scheduler(mock_app, mock_config)

        job_ids = [
            call.kwargs.get("id") for call in mock_scheduler.add_job.call_args_list
        ]
        assert "market_data_preflight_active_positions" in job_ids
