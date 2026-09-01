"""Tests for cash-session daily bar aggregation from 30-minute futures bars.

Verifies that the pure aggregation function correctly filters to cash-session
hours, builds daily OHLCV bars, and handles incomplete sessions and edge cases.
"""

from app.models import FuturesPrice
from app.services.market.futures_aggregation import (
    CashSessionDailyBar,
    _is_within_cash_session,
    _parse_bar_time,
    aggregate_cash_session_daily_bars,
)


def _make_bar(
    bar_time: str,
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: int = 1000,
) -> FuturesPrice:
    """Helper to create a FuturesPrice bar for testing."""
    return FuturesPrice(
        symbol="MNQ",
        contract="MNQU2026",
        bar_time=bar_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


class TestParseBarTime:
    """Tests for bar time parsing."""

    def test_iso_format_with_t(self) -> None:
        result = _parse_bar_time("2026-08-29T09:30:00")
        assert result is not None
        assert result.hour == 9
        assert result.minute == 30

    def test_space_separated_format(self) -> None:
        result = _parse_bar_time("2026-08-29 14:00:00")
        assert result is not None
        assert result.hour == 14

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_bar_time("invalid") is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_bar_time("") is None


class TestIsWithinCashSession:
    """Tests for cash-session time filtering."""

    def test_session_open_is_included(self) -> None:
        """15:30 bar is the first cash-session bar (09:30 ET / Kassa-Open)."""
        bar = _parse_bar_time("2026-08-29T15:30:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True

    def test_session_close_bar_is_included(self) -> None:
        """21:30 bar (ends at 22:00) is included as the last cash bar (16:00 ET / Kassa-Close)."""
        bar = _parse_bar_time("2026-08-29T21:30:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True

    def test_premarket_bar_excluded(self) -> None:
        """15:00 bar starts before cash session."""
        bar = _parse_bar_time("2026-08-29T15:00:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is False

    def test_after_hours_bar_excluded(self) -> None:
        """22:00 bar starts after cash session close."""
        bar = _parse_bar_time("2026-08-29T22:00:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is False

    def test_overnight_bar_excluded(self) -> None:
        """02:00 bar is overnight session."""
        bar = _parse_bar_time("2026-08-29T02:00:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is False

    def test_midday_bar_included(self) -> None:
        """18:00 bar is in the middle of cash session."""
        bar = _parse_bar_time("2026-08-29T18:00:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True

    def test_march_dst_transition_week_open(self) -> None:
        """During March transition week (US in EDT, EU in CET): US Open is 14:30 local time."""
        # 14:30 in March 16 (CET +1) = 09:30 EDT (-4) -> US Open
        bar = _parse_bar_time("2026-03-16T14:30:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True

    def test_march_dst_transition_week_after_close(self) -> None:
        """During March transition week: 21:00 local time is 16:00 EDT (after close bar)."""
        bar = _parse_bar_time("2026-03-16T21:00:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is False

    def test_winter_time_open(self) -> None:
        """In winter (both in standard time): 15:30 CET = 09:30 EST (US Open)."""
        bar = _parse_bar_time("2026-01-15T15:30:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True

    def test_winter_time_close_bar(self) -> None:
        """In winter: 21:30 CET = 15:30 EST (US Close bar)."""
        bar = _parse_bar_time("2026-01-15T21:30:00")
        assert bar is not None
        assert _is_within_cash_session(bar) is True


def _build_full_cash_session_bars(
    trading_date: str = "2026-08-29",
) -> list[FuturesPrice]:
    """Builds a complete set of cash-session bars for a trading day.

    13 bars: 15:30, 16:00, 16:30, 17:00, 17:30, 18:00, 18:30,
             19:00, 19:30, 20:00, 20:30, 21:00, 21:30
    """
    bars: list[FuturesPrice] = []
    # Session open bar (15:30)
    bars.append(
        _make_bar(
            f"{trading_date}T15:30:00",
            open_price=29500.0,
            high=29550.0,
            low=29480.0,
            close=29530.0,
            volume=5000,
        )
    )
    # Middle bars (16:00 through 21:00, 11 bars at 30 min intervals)
    hour_prices = [
        (29530.0, 29560.0, 29520.0, 29545.0, 3000),  # 16:00
        (29545.0, 29570.0, 29540.0, 29555.0, 2500),  # 16:30
        (29555.0, 29580.0, 29530.0, 29540.0, 2000),  # 17:00
        (29540.0, 29550.0, 29510.0, 29520.0, 2200),  # 17:30
        (29520.0, 29525.0, 29490.0, 29500.0, 1800),  # 18:00
        (29500.0, 29510.0, 29485.0, 29505.0, 1500),  # 18:30
        (29505.0, 29520.0, 29500.0, 29515.0, 2000),  # 19:00
        (29515.0, 29530.0, 29510.0, 29525.0, 2500),  # 19:30
        (29525.0, 29540.0, 29520.0, 29535.0, 3000),  # 20:00
        (29535.0, 29545.0, 29530.0, 29540.0, 2800),  # 20:30
        (29540.0, 29550.0, 29535.0, 29538.0, 3200),  # 21:00
    ]
    minutes_offsets = [0, 30, 0, 30, 0, 30, 0, 30, 0, 30, 0]
    hours = [16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21]

    for (open_price, high, low, close, volume), hour, minute in zip(
        hour_prices, hours, minutes_offsets, strict=True
    ):
        bars.append(
            _make_bar(
                f"{trading_date}T{hour:02d}:{minute:02d}:00",
                open_price=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )

    # Session close bar (21:30)
    bars.append(
        _make_bar(
            f"{trading_date}T21:30:00",
            open_price=29538.0,
            high=29560.0,
            low=29530.0,
            close=29550.0,
            volume=4500,
        )
    )
    return bars


class TestAggregateCashSessionDailyBars:
    """Tests for the main aggregation function."""

    def test_full_session_produces_daily_bar(self) -> None:
        """A complete cash session produces one daily bar."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)

        assert len(result) == 1
        daily = result[0]
        assert daily.symbol == "MNQ"
        assert daily.contract == "MNQU2026"
        assert daily.date == "2026-08-29"

    def test_open_is_first_bar_open(self) -> None:
        """Daily open comes from the 15:30 bar."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)
        assert result[0].open == 29500.0

    def test_close_is_last_bar_close(self) -> None:
        """Daily close comes from the 21:30 bar."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)
        assert result[0].close == 29550.0

    def test_high_is_session_maximum(self) -> None:
        """Daily high is the maximum high across all cash bars."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)
        # 29580.0 from the 17:00 bar
        assert result[0].high == 29580.0

    def test_low_is_session_minimum(self) -> None:
        """Daily low is the minimum low across all cash bars."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)
        # 29480.0 from the 15:30 bar
        assert result[0].low == 29480.0

    def test_volume_is_session_sum(self) -> None:
        """Daily volume is the sum across all cash bars."""
        bars = _build_full_cash_session_bars()
        result = aggregate_cash_session_daily_bars(bars)
        expected_volume = sum(bar.volume for bar in bars)
        assert result[0].volume == expected_volume

    def test_overnight_bars_excluded(self) -> None:
        """Bars outside cash session are not included in aggregation."""
        cash_bars = _build_full_cash_session_bars()
        overnight_bars = [
            _make_bar("2026-08-29T02:00:00", volume=9999),
            _make_bar("2026-08-29T06:00:00", volume=9999),
            _make_bar("2026-08-29T23:00:00", volume=9999),
        ]
        all_bars = overnight_bars + cash_bars

        result = aggregate_cash_session_daily_bars(all_bars)
        assert len(result) == 1
        # Overnight volume should not be included
        cash_volume = sum(bar.volume for bar in cash_bars)
        assert result[0].volume == cash_volume

    def test_incomplete_session_excluded(self) -> None:
        """Days with fewer than minimum bars are excluded."""
        # Only 5 bars — below MINIMUM_CASH_SESSION_BAR_COUNT
        partial_bars = [_make_bar(f"2026-08-29T{h:02d}:00:00") for h in range(10, 15)]
        result = aggregate_cash_session_daily_bars(partial_bars)
        assert len(result) == 0

    def test_empty_input_returns_empty(self) -> None:
        result = aggregate_cash_session_daily_bars([])
        assert result == []

    def test_multiple_days_sorted_by_date(self) -> None:
        """Multiple trading days produce multiple daily bars in order."""
        day1_bars = _build_full_cash_session_bars("2026-08-28")
        day2_bars = _build_full_cash_session_bars("2026-08-29")
        # Feed in reverse order to test sorting
        all_bars = day2_bars + day1_bars

        result = aggregate_cash_session_daily_bars(all_bars)
        assert len(result) == 2
        assert result[0].date == "2026-08-28"
        assert result[1].date == "2026-08-29"

    def test_to_db_row_format(self) -> None:
        """CashSessionDailyBar.to_db_row() produces correct tuple."""
        bar = CashSessionDailyBar(
            symbol="MNQ",
            contract="MNQU2026",
            date="2026-08-29",
            open=29500.0,
            high=29580.0,
            low=29480.0,
            close=29550.0,
            volume=36000,
        )
        row = bar.to_db_row()
        assert row == (
            "MNQ",
            "MNQU2026",
            "2026-08-29",
            29500.0,
            29580.0,
            29480.0,
            29550.0,
            36000,
            "cash",
        )
