"""Unit tests for the TGIM screener strategy."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.tgim import TGIMStrategy


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    """Fixture providing a mock TradeRepository."""
    repo = MagicMock(spec=TradeRepository)
    repo.exists.return_value = False
    repo.create_trade.return_value = 1
    return repo


@pytest.fixture
def mock_data_provider() -> MagicMock:
    """Fixture providing a mock MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def tgim_strategy(
    mock_trade_repo: MagicMock, mock_data_provider: MagicMock
) -> TGIMStrategy:
    """Fixture providing a TGIMStrategy instance."""
    return TGIMStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
    )


def test_tgim_skips_non_monday(
    tgim_strategy: TGIMStrategy, mock_data_provider: MagicMock
) -> None:
    """Tests that TGIM skips screening if analysis date is not a Monday (e.g. Tuesday)."""
    hits = tgim_strategy.run(days=0, analysis_date="2026-07-21")
    assert hits == 0
    mock_data_provider.get_batch_history.assert_not_called()


def test_tgim_generates_signal_on_valid_monday_setup(
    tgim_strategy: TGIMStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests signal generation when Monday close is lower than Friday and Thursday close."""
    monday = pd.Timestamp("2026-07-20")
    friday = pd.Timestamp("2026-07-17")
    thursday = pd.Timestamp("2026-07-16")

    df_history = pd.DataFrame(
        [
            {
                "date": thursday,
                "open": 500,
                "high": 502,
                "low": 498,
                "close": 500.0,
                "volume": 1000,
            },
            {
                "date": friday,
                "open": 499,
                "high": 501,
                "low": 494,
                "close": 495.0,
                "volume": 1000,
            },
            {
                "date": monday,
                "open": 494,
                "high": 496,
                "low": 489,
                "close": 490.0,
                "volume": 1000,
            },
        ]
    )

    mock_data_provider.get_batch_history.return_value = {"SPY": df_history}

    hits = tgim_strategy.run(days=0, analysis_date="2026-07-20")

    assert hits == 1
    mock_trade_repo.create_trade.assert_called_once()
    call_kwargs = mock_trade_repo.create_trade.call_args.kwargs
    assert call_kwargs["symbol"] == "SPY"
    assert call_kwargs["strategy"] == Strategies.TGIM.value
    assert call_kwargs["entry"] == 490.0


def test_tgim_fails_if_monday_close_is_not_lowest(
    tgim_strategy: TGIMStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests setup failure if Monday close is higher than Friday close."""
    monday = pd.Timestamp("2026-07-20")
    friday = pd.Timestamp("2026-07-17")
    thursday = pd.Timestamp("2026-07-16")

    df_history = pd.DataFrame(
        [
            {"date": thursday, "close": 500.0},
            {"date": friday, "close": 485.0},
            {"date": monday, "close": 490.0},
        ]
    )

    mock_data_provider.get_batch_history.return_value = {"SPY": df_history}

    hits = tgim_strategy.run(days=0, analysis_date="2026-07-20")

    assert hits == 0
    mock_trade_repo.create_trade.assert_not_called()


def test_tgim_skips_if_active_trade_already_exists(
    tgim_strategy: TGIMStrategy,
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
) -> None:
    """Tests that signal is skipped if an active/created trade already exists for SPY."""
    monday = pd.Timestamp("2026-07-20")
    friday = pd.Timestamp("2026-07-17")
    thursday = pd.Timestamp("2026-07-16")

    df_history = pd.DataFrame(
        [
            {"date": thursday, "close": 500.0},
            {"date": friday, "close": 495.0},
            {"date": monday, "close": 490.0},
        ]
    )

    mock_data_provider.get_batch_history.return_value = {"SPY": df_history}
    mock_trade_repo.exists.return_value = True

    hits = tgim_strategy.run(days=0, analysis_date="2026-07-20")

    assert hits == 0
    mock_trade_repo.create_trade.assert_not_called()
