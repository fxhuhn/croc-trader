"""Unit test suite for the Wounded Bull Sprint screener strategy."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies
from app.services.screener.strategies.wounded_bull_sprint import (
    WoundedBullSprintParameters,
    WoundedBullSprintStrategy,
    evaluate_symbol_technicals,
    rank_and_filter_fip,
)
from app.tools.market_holidays import MarketHolidayChecker


def _build_synthetic_history(
    num_bars: int = 250,
    base_price: float = 100.0,
    price_slope: float = 0.2,
    volume: float = 2_000_000.0,
    daily_range: float = 8.0,
    end_pullback_bars: int = 10,
) -> pd.DataFrame:
    """Builds synthetic daily OHLCV bars satisfying default WBS uptrend and volume criteria."""
    dates = pd.date_range(end="2026-07-20", periods=num_bars, freq="B")
    records: list[dict[str, object]] = []

    price = base_price
    for i in range(num_bars):
        # Build uptrend for first (num_bars - end_pullback_bars) bars, then a short pullback
        if i >= num_bars - end_pullback_bars:
            price -= 1.0  # Causes negative ROC_10 and negative ROC_5
        else:
            price += price_slope

        high = price + daily_range / 2.0
        low = price - daily_range / 2.0
        close = price
        open_p = price - 0.2

        records.append(
            {
                "date": dates[i].date(),
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(records)


def test_evaluate_symbol_technicals_happy_path() -> None:
    """Tests evaluate_symbol_technicals returns metrics dict on valid qualifying data."""
    df = _build_synthetic_history(num_bars=250, base_price=100.0, price_slope=0.5)
    params = WoundedBullSprintParameters()

    result = evaluate_symbol_technicals(df, params)

    assert result is not None
    assert result["setup_close"] > 5.0
    assert result["roc_10"] < 0.0
    assert result["atr_pct"] > 2.5
    assert result["setup_score"] == pytest.approx(-1.0 * float(result["roc_5"]))
    assert isinstance(result["up_days"], int)


def test_evaluate_symbol_technicals_insufficient_history() -> None:
    """Tests evaluate_symbol_technicals returns None when history < 226 bars."""
    df = _build_synthetic_history(num_bars=200)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_evaluate_symbol_technicals_below_min_price() -> None:
    """Tests evaluate_symbol_technicals rejects penny stocks (Close <= 5.0)."""
    df = _build_synthetic_history(num_bars=250, base_price=2.0, price_slope=0.01)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_evaluate_symbol_technicals_below_volume_sma() -> None:
    """Tests evaluate_symbol_technicals rejects illiquid stocks (Volume SMA <= 1M)."""
    df = _build_synthetic_history(num_bars=250, volume=500_000.0)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_evaluate_symbol_technicals_below_trend_sma() -> None:
    """Tests evaluate_symbol_technicals rejects stocks trading below SMA(200)."""
    # Create persistent downtrend
    dates = pd.date_range(end="2026-07-20", periods=250, freq="B")
    records = [
        {
            "date": dates[i].date(),
            "open": 200.0 - (i * 0.5),
            "high": 202.0 - (i * 0.5),
            "low": 198.0 - (i * 0.5),
            "close": 200.0 - (i * 0.5),
            "volume": 2_000_000.0,
        }
        for i in range(250)
    ]
    df = pd.DataFrame(records)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_evaluate_symbol_technicals_low_volatility() -> None:
    """Tests evaluate_symbol_technicals rejects assets with ATR(3)/Close <= 2.5%."""
    # Tiny daily range: 0.1 on a 100.0 stock => ATR% approx 0.1% (< 2.5%)
    df = _build_synthetic_history(num_bars=250, base_price=100.0, daily_range=0.1)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_evaluate_symbol_technicals_positive_roc_10() -> None:
    """Tests evaluate_symbol_technicals rejects assets not experiencing a pullback (ROC_10 >= 0)."""
    # end_pullback_bars = 0 => pure uninterrupted uptrend => ROC_10 > 0
    df = _build_synthetic_history(num_bars=250, base_price=100.0, end_pullback_bars=0)
    params = WoundedBullSprintParameters()

    assert evaluate_symbol_technicals(df, params) is None


def test_rank_and_filter_fip() -> None:
    """Tests cross-sectional FIP ranking drops below 51st percentile and sorts by SetupScore."""
    params = WoundedBullSprintParameters(days_pct=51.0)

    # 4 universe stocks with known UpDays
    all_universe_up_days = {
        "AAA": 80,  # Lowest rank (~25%)
        "BBB": 110,  # 50%
        "CCC": 130,  # 75%
        "DDD": 160,  # 100%
    }

    qualifying_technicals = [
        {
            "symbol": "AAA",
            "setup_close": 50.0,
            "setup_score": 10.0,
            "roc_5": -10.0,
            "roc_10": -5.0,
            "atr_pct": 3.0,
            "up_days": 80,
        },
        {
            "symbol": "CCC",
            "setup_close": 120.0,
            "setup_score": 4.0,  # Moderate score
            "roc_5": -4.0,
            "roc_10": -3.0,
            "atr_pct": 3.2,
            "up_days": 130,
        },
        {
            "symbol": "DDD",
            "setup_close": 200.0,
            "setup_score": 8.0,  # High score
            "roc_5": -8.0,
            "roc_10": -4.0,
            "atr_pct": 3.5,
            "up_days": 160,
        },
    ]

    ranked = rank_and_filter_fip(qualifying_technicals, all_universe_up_days, params)

    # AAA should be dropped (FIP rank ~25% < 51%)
    # CCC and DDD should qualify, with DDD first because setup_score 8.0 > 4.0
    symbols = [r.symbol for r in ranked]
    assert symbols == ["DDD", "CCC"]
    assert ranked[0].setup_score == 8.0
    assert ranked[1].setup_score == 4.0
    assert ranked[0].up_days_pct >= 51.0
    assert ranked[1].up_days_pct >= 51.0


def test_wounded_bull_sprint_screener_happy_path() -> None:
    """Tests WBS screener generates signals and persists them to trade repository."""
    trade_repo = MagicMock()
    trade_repo.get_by_status.return_value = []
    trade_repo.exists.return_value = False
    trade_repo.create_trade.return_value = 501

    data_provider = MagicMock()

    # Target date: 2026-07-22 (TradingDayOfMonth = 16, BarsLeft = 7)
    target_date_str = "2026-07-22"

    # Macro data: SPY MTD = -1%, TLT MTD = +2% => SPY < TLT (defensive regime)
    spy_dates = [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)]
    spy_prices = [500.0, 495.0]  # -1.0% MTD
    spy_df = pd.DataFrame({"date": spy_dates, "close": spy_prices})

    tlt_dates = [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)]
    tlt_prices = [90.0, 91.8]  # +2.0% MTD
    tlt_df = pd.DataFrame({"date": tlt_dates, "close": tlt_prices})

    # Symbol data for S&P 100
    aapl_df = _build_synthetic_history(num_bars=250, base_price=150.0, price_slope=0.5)

    def mock_get_batch_history(
        symbols: list[str], days: int, end_date: str
    ) -> dict[str, pd.DataFrame]:
        res: dict[str, pd.DataFrame] = {}
        if "SPY" in symbols:
            res["SPY"] = spy_df
        if "TLT" in symbols:
            res["TLT"] = tlt_df
        if "AAPL" in symbols:
            res["AAPL"] = aapl_df
        return res

    data_provider.get_batch_history.side_effect = mock_get_batch_history

    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )
    # Restrict exchange universe mock to AAPL
    strategy.exchange_symbols._sp_100 = ["AAPL"]

    hits = strategy.run(analysis_date=target_date_str)

    assert hits == 1
    trade_repo.create_trade.assert_called_once()
    kwargs = trade_repo.create_trade.call_args.kwargs
    assert kwargs["symbol"] == "AAPL"
    assert kwargs["strategy"] == Strategies.WoundedBullSprint
    assert kwargs["context"]["TradingDayOfMonth"] == 16
    assert kwargs["context"]["spy_mtd"] == pytest.approx(-1.0, abs=0.1)
    assert kwargs["context"]["tlt_mtd"] == pytest.approx(2.0, abs=0.1)
    assert "setup_score" in kwargs["context"]


def test_wounded_bull_sprint_screener_holiday_guard() -> None:
    """Tests WBS screener immediately returns 0 on non-trading days."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = True  # Holiday

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )

    hits = strategy.run(analysis_date="2026-07-03")
    assert hits == 0
    trade_repo.create_trade.assert_not_called()


def test_wounded_bull_sprint_screener_tdom_guard() -> None:
    """Tests WBS screener aborts early if TradingDayOfMonth <= 14."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )

    # 2026-07-02 is Trading Day 2 (<= 14)
    hits = strategy.run(analysis_date="2026-07-02")
    assert hits == 0
    data_provider.get_batch_history.assert_not_called()


def test_wounded_bull_sprint_screener_bars_left_guard() -> None:
    """Tests WBS screener aborts early if remaining trading days < 3."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )

    # 2026-07-31 is the last trading day of July (td_left = 1 < 3)
    hits = strategy.run(analysis_date="2026-07-31")
    assert hits == 0
    data_provider.get_batch_history.assert_not_called()


def test_wounded_bull_sprint_screener_macro_regime_guard() -> None:
    """Tests WBS screener aborts if SPY MTD >= TLT MTD (non-defensive environment)."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # SPY +5%, TLT +1% => SPY > TLT => Non-defensive
    spy_df = pd.DataFrame(
        {
            "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
            "close": [500.0, 525.0],
        }
    )
    tlt_df = pd.DataFrame(
        {
            "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
            "close": [90.0, 90.9],
        }
    )
    data_provider.get_batch_history.return_value = {"SPY": spy_df, "TLT": tlt_df}

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )
    strategy.exchange_symbols._sp_100 = ["AAPL"]

    hits = strategy.run(analysis_date="2026-07-22")
    assert hits == 0
    trade_repo.create_trade.assert_not_called()


def test_wounded_bull_sprint_screener_capacity_guard() -> None:
    """Tests WBS screener aborts if 10 active/created positions already exist."""
    trade_repo = MagicMock()
    # 10 active WBS positions
    trade_repo.get_by_status.return_value = [
        {"symbol": f"SYM{i}", "strategy": Strategies.WoundedBullSprint.value}
        for i in range(10)
    ]
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    # Macro defensive
    spy_df = pd.DataFrame(
        {
            "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
            "close": [500.0, 490.0],
        }
    )
    tlt_df = pd.DataFrame(
        {
            "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
            "close": [90.0, 92.0],
        }
    )
    data_provider.get_batch_history.return_value = {"SPY": spy_df, "TLT": tlt_df}

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )
    strategy.exchange_symbols._sp_100 = ["AAPL"]

    hits = strategy.run(analysis_date="2026-07-22")
    assert hits == 0
    trade_repo.create_trade.assert_not_called()
