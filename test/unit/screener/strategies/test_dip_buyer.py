# filename: test_dip_buyer.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from app.services.screener.strategies.dip_buyer import DipBuyerStrategy
from app.database.repositories.trade import TradeRepository
from app.database.repositories.market_data_provider import MarketDataProvider
from app.services.telegram import TelegramBot


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_data_provider() -> MagicMock:
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def mock_telegram_bot() -> MagicMock:
    return MagicMock(spec=TelegramBot)


@pytest.fixture
def strategy(
    mock_trade_repo: MagicMock,
    mock_data_provider: MagicMock,
    mock_telegram_bot: MagicMock,
) -> DipBuyerStrategy:
    with patch("app.services.screener.strategies.dip_buyer.ExchangeSymbol") as mock_ex:
        mock_ex.return_value.dow_30 = ["AAPL"]
        mock_ex.return_value.sp_500 = ["MSFT"]
        mock_ex.return_value.nasdaq_100 = ["TSLA"]
        return DipBuyerStrategy(
            trade_repository=mock_trade_repo,
            data_provider=mock_data_provider,
            telegram_bot=mock_telegram_bot,
        )


def test_resolve_target_date_with_fixed_date(strategy: DipBuyerStrategy) -> None:
    """Tests date resolution when a specific date is requested."""
    # Arrange
    dates = pd.to_datetime(["2026-02-16", "2026-02-17", "2026-02-18"])
    closes = pd.DataFrame({"AAPL": [100, 101, 102]}, index=dates)

    # Act
    target = strategy._resolve_target_date(closes, "2026-02-17")

    # Assert
    assert target == pd.Timestamp("2026-02-17")


def test_resolve_target_date_fallback_to_last(strategy: DipBuyerStrategy) -> None:
    """Tests fallback to last available date when no date is provided."""
    dates = pd.to_datetime(["2026-02-16", "2026-02-17"])
    closes = pd.DataFrame({"AAPL": [100, 101]}, index=dates)

    target = strategy._resolve_target_date(closes, None)
    assert target == pd.Timestamp("2026-02-17")


def test_resolve_target_date_holiday_fallback(strategy: DipBuyerStrategy) -> None:
    """Tests fallback to previous trading day if requested date is a holiday."""
    dates = pd.to_datetime(["2026-02-13", "2026-02-16"])  # Feb 14-15 weekend
    closes = pd.DataFrame({"AAPL": [100, 101]}, index=dates)

    # Requesting Sunday Feb 15
    target = strategy._resolve_target_date(closes, "2026-02-15")
    assert target == pd.Timestamp("2026-02-13")


def test_compute_indicators(strategy: DipBuyerStrategy) -> None:
    """Tests technical indicator calculation sequence."""
    # Arrange
    index = pd.date_range("2026-01-01", periods=10, freq="B")
    closes = pd.DataFrame({"AAPL": np.linspace(100, 110, 10)}, index=index)
    highs = closes + 2
    lows = closes - 2
    volumes = pd.DataFrame({"AAPL": [1e6] * 10}, index=index)

    with patch("app.tools.indicators.calculate_sma", return_value=closes.copy()):
        with patch(
            "app.tools.indicators.calculate_atr",
            return_value=pd.DataFrame(1.0, index=index, columns=["AAPL"]),
        ):
            with patch(
                "app.tools.indicators.calculate_ibs",
                return_value=pd.DataFrame(0.5, index=index, columns=["AAPL"]),
            ):
                # Act
                indicators = strategy._compute_indicators(closes, highs, lows, volumes)

                # Assert
                assert "sma200" in indicators
                assert "atr" in indicators
                assert "ibs" in indicators
                assert "atr_ratio_3day" in indicators


def test_filter_market_state_signals(strategy: DipBuyerStrategy) -> None:
    """Tests filtering logic for identifying valid dip signals."""
    # Arrange
    index = ["AAPL", "MSFT"]
    current = {
        "volume_sma": pd.Series([2e6, 5e5], index=index),
        "close": pd.Series([110, 50], index=index),
        "sma200": pd.Series([100, 40], index=index),
        "atr_ratio_3day": pd.Series([-2.0, -0.5], index=index),
        "volatility_ratio": pd.Series([0.05, 0.02], index=index),
        "ibs": pd.Series([0.1, 0.5], index=index),
        "open": pd.Series([115, 55], index=index),
        "high": pd.Series([120, 60], index=index),
        "volume": pd.Series([2e6, 5e5], index=index),
        "atr": pd.Series([5, 1], index=index),
        "setup_score": pd.Series([2.0, 0.5], index=index),
    }
    previous = {
        "close": pd.Series([112, 52], index=index),
        "open": pd.Series([114, 54], index=index),
    }

    # Act
    results = strategy._filter_market_state(current, previous)

    # Assert
    assert "AAPL" in results.index
    assert "MSFT" not in results.index  # Failed volume and IBS


def test_process_signals_creates_trades(
    strategy: DipBuyerStrategy, mock_trade_repo: MagicMock
) -> None:
    """Tests that valid signals result in created trades in the repository."""
    # Arrange
    date_obj = pd.Timestamp("2026-02-18")
    signals = pd.DataFrame(
        {
            "close": [100.0],
            "high": [105.0],
            "volume": [1e6],
            "atr": [5.0],
            "sma200": [90.0],
            "atr_ratio_3day": [-2.5],
            "ibs": [0.1],
            "setup_score": [2.5],
        },
        index=["AAPL"],
    )

    # Act
    count = strategy._process_signals(signals, date_obj)

    # Assert
    assert count == 1
    mock_trade_repo.create_trade.assert_called_once()
    args, kwargs = mock_trade_repo.create_trade.call_args
    assert kwargs["symbol"] == "AAPL"
    assert kwargs["entry"] == 95.0  # Close (100) - ATR (5) * EntryFactor (1)


def test_analyze_single_symbol(
    strategy: DipBuyerStrategy, mock_data_provider: MagicMock
) -> None:
    """Tests the single symbol debug analysis method."""
    # Arrange
    symbol = "AAPL"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="B"),
            "open": [105] * 10,
            "high": [110] * 10,
            "low": [95] * 10,
            "close": [100] * 10,
            "volume": [1e6] * 10,
        }
    )
    mock_data_provider.get_symbol_history.return_value = df

    # Act
    result = strategy.analyze_single_symbol(symbol)

    # Assert
    assert result["symbol"] == symbol
    assert result["data_valid"] is True
    assert "checks" in result
    assert "values" in result


def test_run_empty_data(
    strategy: DipBuyerStrategy, mock_data_provider: MagicMock
) -> None:
    """Tests strategy run behavior when no data is available."""
    # Arrange
    mock_data_provider.get_universe_daily_data.return_value = {}

    # Act
    count = strategy.run()

    # Assert
    assert count == 0
