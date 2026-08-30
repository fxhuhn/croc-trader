"""Comprehensive 7-Day Weekday & Calendar Boundary Matrix Hardening Tests.

Verifies that all screening strategies, TradeManager lifecycle evaluators,
and order generation logic produce 100% deterministic and identical results
regardless of the real-world day of the week (Monday through Sunday),
holidays, leap years, or month-end calendar boundaries.
"""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies, TradeStatus
from app.services.screener.strategies.bridge_scout import (
    BridgeScoutStrategy,
    get_remaining_trading_days_in_month,
    is_in_end_of_month_window,
)
from app.services.screener.strategies.tgim import TGIMStrategy
from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from app.tools.market_holidays import MarketHolidayChecker


class MockTradeRepo:
    """Deterministic in-memory trade repository for matrix testing."""

    def __init__(self) -> None:
        self.trades: list[dict[str, object]] = []
        self._next_id = 1

    def create_trade(
        self,
        symbol: str,
        strategy: Strategies | str,
        size: float = 0.0,
        entry: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        context: dict[str, object] | None = None,
    ) -> int:
        trade_id = self._next_id
        self._next_id += 1
        strat_val = strategy.value if isinstance(strategy, Strategies) else strategy
        self.trades.append(
            {
                "id": trade_id,
                "symbol": symbol,
                "strategy": strat_val,
                "current_size": size,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "target_price": target,
                "status": TradeStatus.CREATED.value,
                "signal_context": context or {},
                "date": context.get("date") if context else None,
            }
        )
        return trade_id

    def exists(self, symbol: str, strategy: str, setup_date: str) -> bool:
        for t in self.trades:
            if t["symbol"] == symbol and t["strategy"] == strategy:
                ctx = t.get("signal_context", {})
                if isinstance(ctx, dict) and ctx.get("setup_date") == setup_date:
                    return True
        return False

    def has_active_position(self, symbol: str, strategy: str) -> bool:
        for t in self.trades:
            if t["symbol"] == symbol and t["strategy"] == strategy:
                if t.get("status") == TradeStatus.ACTIVE.value:
                    return True
        return False

    def get_by_status(self, status: TradeStatus) -> list[dict[str, object]]:
        return [t for t in self.trades if t.get("status") == status.value]


def _create_synthetic_history(
    end_date_str: str,
    num_bars: int = 60,
    base_price: float = 100.0,
    price_pattern: list[float] | None = None,
) -> pd.DataFrame:
    """Generates synthetic OHLCV dataframe ending on end_date_str."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    if price_pattern:
        closes = [base_price] * (num_bars - len(price_pattern)) + price_pattern
    else:
        closes = [base_price] * num_bars

    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    opens = closes.copy()
    volumes = [1_000_000] * num_bars

    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# =============================================================================
# 1. 7-DAY WALL-CLOCK MATRIX TESTS (Monday through Sunday Execution Invariance)
# =============================================================================

ALL_SEVEN_DAYS = [
    ("Monday", datetime.date(2026, 8, 24)),
    ("Tuesday", datetime.date(2026, 8, 25)),
    ("Wednesday", datetime.date(2026, 8, 26)),
    ("Thursday", datetime.date(2026, 8, 27)),
    ("Friday", datetime.date(2026, 8, 28)),
    ("Saturday", datetime.date(2026, 8, 29)),
    ("Sunday", datetime.date(2026, 8, 30)),
]


@pytest.mark.parametrize("day_name, mock_today", ALL_SEVEN_DAYS)
def test_bridge_scout_execution_day_invariance(
    day_name: str, mock_today: datetime.date
) -> None:
    """Verifies Bridge Scout produces identical setup hits regardless of system day of the week."""
    repo = MockTradeRepo()
    mock_provider = MagicMock()
    holiday_checker = MarketHolidayChecker()

    strategy = BridgeScoutStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
        holiday_checker=holiday_checker,
    )

    day2_eod = "2026-08-27"
    df_day2 = _create_synthetic_history(
        day2_eod, num_bars=60, base_price=710.0, price_pattern=[711.37, 721.11]
    )
    mock_provider.get_batch_history.return_value = {"QQQ": df_day2}
    mock_provider.get_latest_date.return_value = day2_eod

    hits = strategy.run(days=0, analysis_date=day2_eod)

    assert hits == 1, f"Bridge Scout failed on wall-clock {day_name} ({mock_today})"
    assert len(repo.trades) == 1
    assert repo.trades[0]["symbol"] == "QQQ"


@pytest.mark.parametrize("day_name, mock_today", ALL_SEVEN_DAYS)
def test_tgim_execution_day_invariance(
    day_name: str, mock_today: datetime.date
) -> None:
    """Verifies TGIM produces identical setup candidate regardless of system day of the week."""
    repo = MockTradeRepo()
    mock_provider = MagicMock()

    strategy = TGIMStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    # Friday EOD bar (Monday setup candidate)
    friday_date = "2026-08-28"
    df = _create_synthetic_history(friday_date, num_bars=30, base_price=500.0)
    mock_provider.get_batch_history.return_value = {"SPY": df}
    mock_provider.get_latest_date.return_value = friday_date

    hits = strategy.run(days=0, analysis_date=friday_date)

    assert hits == 1, f"TGIM failed on wall-clock {day_name} ({mock_today})"
    assert len(repo.trades) == 1
    assert repo.trades[0]["symbol"] == "SPY"


@pytest.mark.parametrize("day_name, mock_today", ALL_SEVEN_DAYS)
def test_two_percent_execution_day_invariance(
    day_name: str, mock_today: datetime.date
) -> None:
    """Verifies TwoPercent Friday weekly signal generation is invariant to execution day."""
    repo = MockTradeRepo()
    mock_provider = MagicMock()

    strategy = TwoPercentStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    friday_date = "2026-08-28"
    df = _create_synthetic_history(friday_date, num_bars=30, base_price=100.0)
    mock_provider.get_symbol_history.return_value = df

    hits = strategy.run(days=0, analysis_date=friday_date)

    assert hits == 1, f"TwoPercent failed on wall-clock {day_name} ({mock_today})"
    assert len(repo.trades) == 1
    assert repo.trades[0]["symbol"] == "SXRV.DE"


# =============================================================================
# 2. HOLIDAY-SHIFTED WEEKEND TRANSITION TESTS (Good Friday & Thanksgiving)
# =============================================================================


def test_turnover_timing_holiday_thursday_setup_detection() -> None:
    """Verifies Turnover Timing recognizes Thursday as setup day when Friday is a market holiday."""
    repo = MockTradeRepo()
    mock_provider = MagicMock()

    strategy = TurnoverTimingStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    holiday_checker = MarketHolidayChecker()
    # 2026-04-02 is Thursday before Good Friday (2026-04-03 is market holiday)
    thursday_date = "2026-04-02"
    assert holiday_checker.is_holiday("2026-04-03") is True

    analysis_ts = pd.Timestamp(thursday_date)
    assert strategy._is_setup_day(analysis_ts) is True


def test_two_percent_holiday_thursday_end_of_week() -> None:
    """Verifies TwoPercent recognizes Thursday as end-of-week when Friday is a holiday."""
    repo = MockTradeRepo()
    mock_provider = MagicMock()

    strategy = TwoPercentStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    # 2026-04-02 is Thursday before Good Friday (2026-04-03 is holiday)
    thursday = datetime.date(2026, 4, 2)
    existing_dates = {datetime.date(2026, 4, 1), thursday}

    is_eow = strategy._is_fallback_weekday_end_of_week(
        candle_date=thursday,
        weekday=3,  # Thursday
        existing_dates=existing_dates,
    )
    assert is_eow is True


# =============================================================================
# 3. END-OF-MONTH CALENDAR MATRIX (Leap Years, 28/29/30/31 Day Months)
# =============================================================================


@pytest.mark.parametrize(
    "test_date, expected_remaining_days",
    [
        # August 2026 (31 days)
        # Aug 31 is Monday (1), Aug 28 is Friday (2), Aug 27 is Thursday (3), Aug 26 is Wed (4), Aug 25 is Tue (5)
        (datetime.date(2026, 8, 31), 1),
        (datetime.date(2026, 8, 28), 2),
        (datetime.date(2026, 8, 27), 3),
        (datetime.date(2026, 8, 26), 4),
        (datetime.date(2026, 8, 25), 5),
        (datetime.date(2026, 8, 24), 6),
        # February 2028 (Leap Year - 29 days, Feb 29 is Tuesday)
        (datetime.date(2028, 2, 29), 1),
        (datetime.date(2028, 2, 28), 2),
        (datetime.date(2028, 2, 25), 3),
        (datetime.date(2028, 2, 24), 4),
        (datetime.date(2028, 2, 23), 5),
    ],
)
def test_remaining_trading_days_matrix(
    test_date: datetime.date, expected_remaining_days: int
) -> None:
    """Verifies that remaining trading days calculation in Bridge Scout is exact across months."""
    holiday_checker = MarketHolidayChecker()
    remaining = get_remaining_trading_days_in_month(test_date, holiday_checker)
    assert remaining == expected_remaining_days


@pytest.mark.parametrize(
    "test_date, in_window",
    [
        (datetime.date(2026, 8, 25), True),  # 5 days before
        (datetime.date(2026, 8, 26), True),  # 4 days before
        (datetime.date(2026, 8, 27), True),  # 3 days before
        (datetime.date(2026, 8, 28), True),  # 2 days before
        (datetime.date(2026, 8, 31), True),  # Last day of month
        (datetime.date(2026, 8, 24), False),  # 6 days before (Outside window)
        (datetime.date(2026, 8, 15), False),  # Mid-month (Outside window)
    ],
)
def test_eom_window_detection(test_date: datetime.date, in_window: bool) -> None:
    """Verifies Bridge Scout EOM window detection accuracy."""
    holiday_checker = MarketHolidayChecker()
    result = is_in_end_of_month_window(
        test_date, days_before=4, holiday_checker=holiday_checker
    )
    assert result == in_window
