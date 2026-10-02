# filename: test_turnover_timing.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.const import Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.turnover_timing import (
    TurnoverCandidate,
    TurnoverConfiguration,
    TurnoverTimingStrategy,
)


@pytest.fixture
def mock_trade_repository() -> MagicMock:
    """Fixture for mocked TradeRepository."""
    repository = MagicMock(spec=TradeRepository)
    repository.exists.return_value = False
    return repository


@pytest.fixture
def mock_market_data_provider() -> MagicMock:
    """Fixture for mocked MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def strategy(
    mock_trade_repository: MagicMock, mock_market_data_provider: MagicMock
) -> TurnoverTimingStrategy:
    """Fixture for TurnoverTimingStrategy with mocked dependencies."""
    return TurnoverTimingStrategy(
        trade_repository=mock_trade_repository,
        data_provider=mock_market_data_provider,
        configuration=TurnoverConfiguration(),
    )


@pytest.fixture
def sample_market_data() -> dict[str, pd.DataFrame]:
    """Generates standard market data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=250, freq="B")
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    data = {}
    for column in ["open", "high", "low", "close"]:
        df = pd.DataFrame(
            np.random.uniform(100, 200, size=(250, 5)), index=dates, columns=symbols
        )
        df.columns.name = "symbol"
        data[column] = df

    volume_df = pd.DataFrame(
        np.random.uniform(1000000, 2000000, size=(250, 5)), index=dates, columns=symbols
    )
    volume_df.columns.name = "symbol"
    data["volume"] = volume_df
    return data


@pytest.mark.parametrize(
    "test_date, is_friday_holiday, expected_run",
    [
        ("2026-02-09", False, False),  # Monday
        ("2026-02-10", False, False),  # Tuesday
        ("2026-02-11", False, False),  # Wednesday
        ("2026-02-12", False, False),  # Thursday (Normal)
        ("2026-02-13", False, True),  # Friday (Normal)
        ("2026-02-12", True, True),  # Thursday (Friday is Holiday)
    ],
)
def test_screener_execution_timing_logic(
    strategy: TurnoverTimingStrategy,
    test_date: str,
    is_friday_holiday: bool,
    expected_run: bool,
) -> None:
    """Verifies that the screener only executes on the designated 'End of Week' days."""
    # Arrange
    with patch(
        "app.tools.market_holidays.MarketHolidayChecker.is_holiday"
    ) as mock_is_holiday:

        def side_effect(date_obj):
            # 2026-02-13 is a Friday
            if is_friday_holiday and str(date_obj) == "2026-02-13":
                return True
            return False

        mock_is_holiday.side_effect = side_effect
        strategy.data_provider.get_universe_daily_data.return_value = {}

        # Act
        result = strategy.run(analysis_date=test_date)

        # Assert
        if expected_run:
            strategy.data_provider.get_universe_daily_data.assert_called()
        else:
            assert result == 0
            strategy.data_provider.get_universe_daily_data.assert_not_called()


@patch("app.services.screener.strategies.turnover_timing.ExchangeSymbol")
def test_turnover_strategy_generates_signals_on_valid_setup(
    mock_exchange_symbol: MagicMock,
    strategy: TurnoverTimingStrategy,
    mock_market_data_provider: MagicMock,
    mock_trade_repository: MagicMock,
    sample_market_data: dict[str, pd.DataFrame],
) -> None:
    """Verifies full execution flow and signal generation for Turnover Timing."""
    # Arrange
    setup_date = "2024-05-17"  # Friday
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    mock_loader = MagicMock()
    mock_loader.nasdaq_100 = symbols[:2]
    mock_loader.sp_500 = symbols[2:4]
    mock_loader.russell_1000 = [symbols[4]]
    mock_exchange_symbol.return_value = mock_loader

    mock_market_data_provider.get_universe_daily_data.return_value = sample_market_data

    # Mock Indicators
    mock_sma_df = pd.DataFrame(
        50.0, index=sample_market_data["close"].index, columns=symbols
    )
    mock_atr_df = pd.DataFrame(
        5.0, index=sample_market_data["close"].index, columns=symbols
    )

    with patch("app.tools.indicators.calculate_sma", return_value=mock_sma_df):
        with patch("app.tools.indicators.calculate_atr", return_value=mock_atr_df):
            # Act
            signal_count = strategy.run(analysis_date=setup_date)

            # Assert
            assert signal_count > 0
            assert mock_trade_repository.create_trade.called


def test_compile_target_universe_with_specific_symbols(
    strategy: TurnoverTimingStrategy,
) -> None:
    """Verifies that specific symbols filter the target universe."""
    # Arrange
    index_constituents = {"IDX": ["AAPL", "MSFT", "GOOG"]}
    specific = ["AAPL", "TSLA"]

    # Act
    universe = strategy._compile_target_universe(
        index_constituents, specific_symbols=specific
    )

    # Assert
    assert universe == ["AAPL"]


def test_analyze_single_symbol_success(
    strategy: TurnoverTimingStrategy, mock_market_data_provider: MagicMock
) -> None:
    """Tests the single symbol debug analysis method."""
    # Arrange
    symbol = "AAPL"
    # Create an uptrend: last close 120, SMA150 will be around 100
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=250, freq="B"),
            "open": [100.0] * 250,
            "high": [125.0] * 250,
            "low": [95.0] * 250,
            "close": [100.0] * 249 + [120.0],
            "volume": [1000000] * 250,
        }
    )
    mock_market_data_provider.get_symbol_history.return_value = df

    # Act
    result = strategy.analyze_single_symbol(symbol)

    # Assert
    assert result["symbol"] == symbol
    assert result["data_valid"] is True
    assert result["checks"]["uptrend_sma150"] is True


def test_identify_strategy_candidates_keyerror_handling(
    strategy: TurnoverTimingStrategy,
) -> None:
    """Tests that candidates identification handles KeyError (missing dates/columns) gracefully."""
    # Arrange
    # Missing 'high', 'low', 'volume'
    data = {
        "close": pd.DataFrame({"AAPL": [100.0]}, index=[pd.Timestamp("2024-01-01")])
    }
    setup_date = pd.Timestamp("2024-01-02")

    # Act & Assert
    with pytest.raises(KeyError):
        strategy._identify_strategy_candidates(data, setup_date, {})


def test_run_last_trading_day_mismatch(
    strategy: TurnoverTimingStrategy, mock_market_data_provider: MagicMock
) -> None:
    """Tests that run() returns 0 if the latest data date doesn't match analysis date."""
    # Arrange
    analysis_date = "2026-02-13"  # Friday
    # Data only goes up to Thursday
    dates = pd.to_datetime(["2026-02-12"])
    df = pd.DataFrame({"AAPL": [100.0]}, index=dates)
    mock_market_data_provider.get_universe_daily_data.return_value = {"close": df}

    # Act
    result = strategy.run(analysis_date=analysis_date)

    # Assert
    assert result == 0


def test_is_setup_day_comprehensive_matrix(strategy: TurnoverTimingStrategy) -> None:
    """Verifies that _is_setup_day accurately detects week-end boundaries across all weekdays."""
    # Monday 2026-07-13 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-13")) is False
    # Tuesday 2026-07-14 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-14")) is False
    # Wednesday 2026-07-15 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-15")) is False
    # Normal Thursday 2026-07-16 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-16")) is False
    # Normal Friday 2026-07-17 -> True
    assert strategy._is_setup_day(pd.Timestamp("2026-07-17")) is True
    # Saturday 2026-07-18 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-18")) is False
    # Sunday 2026-07-19 -> False
    assert strategy._is_setup_day(pd.Timestamp("2026-07-19")) is False

    # Thursday before Good Friday 2026-04-03 -> True
    thursday_before_holiday = pd.Timestamp("2026-04-02")
    assert strategy._is_setup_day(thursday_before_holiday) is True


def test_is_setup_day_parity_with_trading_calendar(
    strategy: TurnoverTimingStrategy,
) -> None:
    """Verifies 100% bit-level parity between _is_setup_day and is_last_trading_day_of_week."""
    from app.tools.trading_calendar import is_last_trading_day_of_week

    start_date = pd.Timestamp("2026-01-01")
    for day_offset in range(60):
        current_date = start_date + pd.Timedelta(days=day_offset)
        expected = is_last_trading_day_of_week(
            current_date, holiday_checker=strategy.holiday_checker
        )
        actual = strategy._is_setup_day(current_date)
        assert actual == expected, (
            f"Mismatch on {current_date}: expected {expected}, got {actual}"
        )


def test_factor_strategy_name_resolution_canonical(
    strategy: TurnoverTimingStrategy,
) -> None:
    """TC-TT-01: Verifies that factors 0.5 and 1.0 resolve to canonical strategy names."""
    assert strategy._resolve_strategy_name_for_factor(0.5) == str(
        Strategies.TurnOverTiming_05
    )
    assert strategy._resolve_strategy_name_for_factor(1.0) == str(
        Strategies.TurnOverTiming_10
    )


def test_factor_strategy_name_resolution_custom(
    strategy: TurnoverTimingStrategy,
) -> None:
    """TC-TT-02: Verifies that an unmapped custom factor dynamically formats as f'{strategy.name}_{factor}'."""
    assert strategy._resolve_strategy_name_for_factor(0.75) == "turnover_timing_0.75"
    assert strategy._resolve_strategy_name_for_factor(2.0) == "turnover_timing_2.0"


def test_signal_creation_parity_pre_post(
    strategy: TurnoverTimingStrategy,
    mock_trade_repository: MagicMock,
) -> None:
    """TC-TT-03: Verifies candidate signal creation creates exact trades in repository."""
    candidate: TurnoverCandidate = {
        "symbol": "AAPL",
        "close": 150.0,
        "sma_price": 140.0,
        "sma_turnover": 5000000.0,
        "atr": 4.0,
        "indices": "NDX",
    }
    setup_date = pd.Timestamp("2026-07-17")
    data_frames = {"open": pd.DataFrame({"AAPL": [148.0]}, index=[setup_date])}

    count = strategy._store_turnover_signals_for_candidate(
        candidate=candidate,
        data_frames=data_frames,
        setup_date=setup_date,
        setup_date_str="2026-07-17",
    )

    assert count == 2
    assert mock_trade_repository.create_trade.call_count == 2

    # First call: Factor 0.5 -> Entry = 150 - (0.5 * 4) = 148.0
    call_05 = mock_trade_repository.create_trade.call_args_list[0].kwargs
    assert call_05["symbol"] == "AAPL"
    assert call_05["strategy"] == str(Strategies.TurnOverTiming_05)
    assert call_05["entry"] == 148.0
    assert call_05["context"]["factor"] == 0.5

    # Second call: Factor 1.0 -> Entry = 150 - (1.0 * 4) = 146.0
    call_10 = mock_trade_repository.create_trade.call_args_list[1].kwargs
    assert call_10["symbol"] == "AAPL"
    assert call_10["strategy"] == str(Strategies.TurnOverTiming_10)
    assert call_10["entry"] == 146.0
    assert call_10["context"]["factor"] == 1.0


def test_duplicate_signal_prevention_with_mapping(
    strategy: TurnoverTimingStrategy,
    mock_trade_repository: MagicMock,
) -> None:
    """TC-TT-04: Verifies that existing signals are skipped when exists returns True."""
    candidate: TurnoverCandidate = {
        "symbol": "AAPL",
        "close": 150.0,
        "sma_price": 140.0,
        "sma_turnover": 5000000.0,
        "atr": 4.0,
        "indices": "NDX",
    }
    setup_date = pd.Timestamp("2026-07-17")
    data_frames = {"open": pd.DataFrame({"AAPL": [148.0]}, index=[setup_date])}

    # Simulate TurnOverTiming_05 already exists, TurnOverTiming_10 does not
    def exists_side_effect(symbol: str, strategy_name: str, date: str) -> bool:
        return strategy_name == str(Strategies.TurnOverTiming_05)

    mock_trade_repository.exists.side_effect = exists_side_effect

    count = strategy._store_turnover_signals_for_candidate(
        candidate=candidate,
        data_frames=data_frames,
        setup_date=setup_date,
        setup_date_str="2026-07-17",
    )

    # Only TurnOverTiming_10 should be created
    assert count == 1
    assert mock_trade_repository.create_trade.call_count == 1
    call_kwargs = mock_trade_repository.create_trade.call_args.kwargs
    assert call_kwargs["strategy"] == str(Strategies.TurnOverTiming_10)


def test_telegram_reporting_factor_labels(strategy: TurnoverTimingStrategy) -> None:
    """TC-TT-05: Verifies Telegram reporting creates correctly labeled items for all factors."""
    mock_bot = MagicMock()
    strategy.telegram_bot = mock_bot

    candidates: list[TurnoverCandidate] = [
        {
            "symbol": "MSFT",
            "close": 300.0,
            "sma_price": 280.0,
            "sma_turnover": 8000000.0,
            "atr": 6.0,
            "indices": "NDX, SPX",
        }
    ]
    setup_date = pd.Timestamp("2026-07-17")

    strategy._report_signals_to_telegram(candidates, setup_date)

    mock_bot.send_dataframe.assert_called_once()
    reported_df = mock_bot.send_dataframe.call_args[0][0]
    actions = reported_df["Action"].tolist()
    assert "BUY LMT (0.5 ATR)" in actions
    assert "BUY LMT (1.0 ATR)" in actions
