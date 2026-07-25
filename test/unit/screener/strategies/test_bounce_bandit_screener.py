"""Unit tests for the Bounce Bandit screener strategy."""

from unittest.mock import MagicMock

import pandas as pd

from app.const import Strategies
from app.services.screener.strategies.bounce_bandit import BounceBanditStrategy


def test_bounce_bandit_screener_generates_signal_on_valid_setup() -> None:
    """Tests that BounceBanditStrategy creates a signal when all setup conditions match."""
    trade_repository = MagicMock()
    trade_repository.exists.return_value = False
    trade_repository.create_trade.return_value = 101

    data_provider = MagicMock()

    # Construct 250 trading days of history meeting all conditions on final bar
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    close_prices = [400.0 + (i * 0.5) for i in range(247)]
    # Bar t-2: 524.0, Bar t-1: 523.0, Bar t: 520.0 (strictly < min(524.0, 523.0))
    close_prices.extend([524.0, 523.0, 500.0])

    records = [
        {"date": d, "open": p, "high": p + 1.0, "low": p - 1.0, "close": p}
        for d, p in zip(dates, close_prices, strict=True)
    ]
    df_history = pd.DataFrame(records)

    data_provider.get_batch_history.return_value = {"QQQ": df_history}

    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    last_date_str = dates[-1].strftime("%Y-%m-%d")
    hits = strategy.run(analysis_date=last_date_str)

    assert hits == 1
    trade_repository.create_trade.assert_called_once()
    kwargs = trade_repository.create_trade.call_args.kwargs
    assert kwargs["symbol"] == "QQQ"
    assert kwargs["strategy"] == Strategies.BounceBandit.value


def test_bounce_bandit_screener_fails_when_below_sma_200() -> None:
    """Tests that no signal is generated if Close <= SMA(200)."""
    trade_repository = MagicMock()
    data_provider = MagicMock()

    # Construct history where price drops below 200 SMA
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    close_prices = [500.0 - (i * 0.5) for i in range(250)]
    records = [
        {"date": d, "open": p, "high": p + 1.0, "low": p - 1.0, "close": p}
        for d, p in zip(dates, close_prices, strict=True)
    ]
    df_history = pd.DataFrame(records)
    data_provider.get_batch_history.return_value = {"QQQ": df_history}

    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    last_date_str = dates[-1].strftime("%Y-%m-%d")
    hits = strategy.run(analysis_date=last_date_str)

    assert hits == 0
    trade_repository.create_trade.assert_not_called()


def test_bounce_bandit_screener_handles_missing_holiday_data() -> None:
    """Tests that screening skips when target date is a holiday/weekend with missing data."""
    trade_repository = MagicMock()
    data_provider = MagicMock()

    # Latest candle in DB is 2026-07-03, but analysis_date is 2026-07-04 (Holiday)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    records = [
        {"date": d, "open": 500.0, "high": 505.0, "low": 495.0, "close": 500.0}
        for d in dates
    ]
    df_history = pd.DataFrame(records)
    data_provider.get_batch_history.return_value = {"QQQ": df_history}

    strategy = BounceBanditStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
    )

    # Ask for a date that does not match latest candle in df_history
    hits = strategy.run(analysis_date="2026-07-04")

    assert hits == 0
    trade_repository.create_trade.assert_not_called()
