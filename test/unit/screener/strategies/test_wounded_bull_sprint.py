"""Unit test suite for the Wounded Bull Sprint screener strategy."""

import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import Strategies
from app.services.screener.strategies.wounded_bull_sprint import (
    WBSUniverseInput,
    WoundedBullSprintParameters,
    WoundedBullSprintStrategy,
    evaluate_symbol_technicals,
    evaluate_wbs_universe_setups,
    load_wbs_qualifying_setups,
    normalize_trading_date,
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
    end_date: str = "2026-07-22",
) -> pd.DataFrame:
    """Builds synthetic daily OHLCV bars satisfying default WBS uptrend and volume criteria."""
    dates = pd.date_range(end=end_date, periods=num_bars, freq="B")
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
    assert result["setup_score"] == pytest.approx(float(result["atr_pct"]), abs=0.01)
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
    """Tests WBS screener aborts if 5 active/created positions already exist and clears stale cache."""
    trade_repo = MagicMock()
    trade_repo.get_effective_active_trades.return_value = [
        {"symbol": f"SYM{i}", "strategy": Strategies.WoundedBullSprint.value}
        for i in range(5)
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
    strategy.latest_qualifying_setups = {"STALE"}
    strategy.exchange_symbols._sp_100 = ["AAPL"]

    hits = strategy.run(analysis_date="2026-07-22")
    assert hits == 0
    assert strategy.latest_qualifying_setups == set()
    trade_repo.create_trade.assert_not_called()


def test_wounded_bull_sprint_analyze_single_symbol() -> None:
    """Tests analyze_single_symbol returns comprehensive dictionary for MCP / API."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    df = _build_synthetic_history(num_bars=250, base_price=100.0, price_slope=0.5)
    data_provider.get_symbol_history.return_value = df
    data_provider.get_batch_history.return_value = {
        "SPY": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [500.0, 490.0],
            }
        ),
        "TLT": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [90.0, 92.0],
            }
        ),
    }
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )

    analysis = strategy.analyze_single_symbol("AAPL")
    assert analysis["symbol"] == "AAPL"
    assert "technical_filters" in analysis
    assert "macro_regime" in analysis
    assert "calendar" in analysis
    assert "scoring" in analysis
    tech_filters = analysis["technical_filters"]
    assert isinstance(tech_filters, dict)
    assert tech_filters["uptrend_ok"] is True
    assert tech_filters["volatility_ok"] is True
    assert tech_filters["volume_sma_20"] > 0.0
    assert tech_filters["atr_3"] > 0.0
    assert analysis["all_technical_filters_pass"] is True


def test_wounded_bull_sprint_available_slots_factors_pending_exit() -> None:
    """Verifies that _get_available_portfolio_slots uses get_effective_active_trades to allow rotation."""
    trade_repo = MagicMock()
    data_provider = MagicMock()

    # 4 continuing trades, 1 trade with pending_exit (e.g. COP) was excluded by get_effective_active_trades
    trade_repo.get_effective_active_trades.return_value = [
        {"symbol": "UPS", "strategy": "wounded_bull_sprint"},
        {"symbol": "PM", "strategy": "wounded_bull_sprint"},
        {"symbol": "GOOG", "strategy": "wounded_bull_sprint"},
        {"symbol": "GOOGL", "strategy": "wounded_bull_sprint"},
    ]

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
    )

    available_slots = strategy._get_available_portfolio_slots()
    # max_positions (5) - 4 effective active = 1 free slot!
    assert available_slots == 1
    trade_repo.get_effective_active_trades.assert_called_once_with(
        strategy.STRATEGY_IDENTIFIER
    )


def test_normalize_trading_date_variants() -> None:
    """Verifies safe normalization across date, datetime, Timestamp, str, and invalid values."""
    expected = datetime.date(2026, 7, 22)
    assert normalize_trading_date(expected) == expected
    assert normalize_trading_date(datetime.datetime(2026, 7, 22, 15, 30)) == expected
    assert normalize_trading_date(pd.Timestamp("2026-07-22")) == expected
    assert normalize_trading_date("2026-07-22") == expected
    assert normalize_trading_date("2026-07-22 00:00:00") == expected
    assert normalize_trading_date(None) is None
    assert normalize_trading_date("") is None
    assert normalize_trading_date("not-a-date") is None
    assert normalize_trading_date(pd.NaT) is None


def test_evaluate_wbs_universe_setups_four_pillars_and_zero_lookahead() -> None:
    """Verifies full 4-pillar parity, stale-bar rejection, and zero lookahead-bias."""
    target = datetime.date(2026, 7, 22)
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False
    params = WoundedBullSprintParameters()

    spy_defensive = pd.DataFrame(
        {"date": [datetime.date(2026, 6, 30), target], "close": [500.0, 495.0]}
    )
    tlt_defensive = pd.DataFrame(
        {"date": [datetime.date(2026, 6, 30), target], "close": [90.0, 92.0]}
    )
    qualifying_df = _build_synthetic_history(
        num_bars=250, base_price=150.0, price_slope=0.5, end_date="2026-07-22"
    )
    stale_df = _build_synthetic_history(
        num_bars=250, base_price=150.0, price_slope=0.5, end_date="2026-07-20"
    )

    candidates, fip_ranks = evaluate_wbs_universe_setups(
        WBSUniverseInput(
            history_map={"AAPL": qualifying_df, "STALE": stale_df},
            spy_history=spy_defensive,
            tlt_history=tlt_defensive,
            target_date=target,
            holiday_checker=holiday_checker,
        ),
        params,
    )
    assert [c.symbol for c in candidates] == ["AAPL"]
    assert "AAPL" in fip_ranks
    assert "STALE" not in fip_ranks

    # Zero lookahead-bias: appending future bars T+1..T+3 must yield identical result at T
    future_rows = pd.DataFrame(
        [
            {
                "date": datetime.date(2026, 7, 23),
                "open": 50.0,
                "high": 51.0,
                "low": 49.0,
                "close": 50.0,
                "volume": 5_000_000.0,
            }
        ]
    )
    extended_df = pd.concat([qualifying_df, future_rows], ignore_index=True)
    candidates_ext, _ = evaluate_wbs_universe_setups(
        WBSUniverseInput(
            history_map={"AAPL": extended_df},
            spy_history=spy_defensive,
            tlt_history=tlt_defensive,
            target_date=target,
            holiday_checker=holiday_checker,
        ),
        params,
    )
    assert candidates_ext == candidates

    # Risk-on macro (SPY_MTD >= TLT_MTD) rejects all setups
    spy_risk_on = pd.DataFrame(
        {"date": [datetime.date(2026, 6, 30), target], "close": [500.0, 520.0]}
    )
    candidates_risk_on, _ = evaluate_wbs_universe_setups(
        WBSUniverseInput(
            history_map={"AAPL": qualifying_df},
            spy_history=spy_risk_on,
            tlt_history=tlt_defensive,
            target_date=target,
            holiday_checker=holiday_checker,
        ),
        params,
    )
    assert candidates_risk_on == []


def test_wounded_bull_sprint_precheck_candidates_and_full_fip_cache() -> None:
    """Verifies get_precheck_candidates returns ranked items and caches FIP for all universe symbols."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    data_provider.get_latest_date.return_value = "2026-07-22"

    qualifying_df = _build_synthetic_history(
        num_bars=250, base_price=150.0, price_slope=0.5, end_date="2026-07-22"
    )
    non_qualifying_df = _build_synthetic_history(
        num_bars=250,
        base_price=150.0,
        price_slope=-0.1,
        end_pullback_bars=0,
        end_date="2026-07-22",
    )
    data_provider.get_batch_history.return_value = {
        "AAPL": qualifying_df,
        "WEAK": non_qualifying_df,
    }

    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
    )
    strategy.exchange_symbols._sp_100 = ["AAPL", "WEAK"]

    precheck = strategy.get_precheck_candidates(limit=5)
    assert len(precheck) == 1
    assert precheck[0]["symbol"] == "AAPL"
    assert "AAPL" in strategy._daily_universe_fip_cache
    assert "WEAK" in strategy._daily_universe_fip_cache


def test_load_wbs_qualifying_setups_calendar_precheck_and_cache() -> None:
    """Verifies calendar-inactive dates skip DB queries and valid dates are cached."""
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False
    cache: dict[datetime.date, set[str]] = {}

    # TDOM = 2 (<= 14) -> must short-circuit BEFORE calling get_batch_history
    early_month = datetime.date(2026, 7, 2)
    assert (
        load_wbs_qualifying_setups(
            early_month, data_provider, ["AAPL"], holiday_checker, cache
        )
        == set()
    )
    data_provider.get_batch_history.assert_not_called()

    # Active calendar date -> queries once, second call served from cache
    active_date = datetime.date(2026, 7, 22)
    aapl_df = _build_synthetic_history(
        num_bars=250, base_price=150.0, price_slope=0.5, end_date="2026-07-22"
    )
    data_provider.get_batch_history.return_value = {
        "AAPL": aapl_df,
        "SPY": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), active_date],
                "close": [500.0, 495.0],
            }
        ),
        "TLT": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), active_date],
                "close": [90.0, 92.0],
            }
        ),
    }
    first = load_wbs_qualifying_setups(
        active_date, data_provider, ["AAPL"], holiday_checker, cache
    )
    second = load_wbs_qualifying_setups(
        active_date, data_provider, ["AAPL"], holiday_checker, cache
    )
    assert first == {"AAPL"}
    assert second == {"AAPL"}
    data_provider.get_batch_history.assert_called_once()


def test_wbs_analyze_single_symbol_status_branches() -> None:
    """Covers empty history, qualified FIP, rejected low FIP, and non-defensive macro branches."""
    trade_repo = MagicMock()
    data_provider = MagicMock()
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.return_value = False
    strategy = WoundedBullSprintStrategy(
        trade_repository=trade_repo,
        data_provider=data_provider,
        holiday_checker=holiday_checker,
    )

    # 1. Empty history -> INSUFFICIENT_DATA
    data_provider.get_symbol_history.return_value = pd.DataFrame()
    empty_result = strategy.analyze_single_symbol("EMPTY")
    assert empty_result["status"] == "INSUFFICIENT_DATA"

    # 2. Qualified with cached FIP >= 51% and defensive macro
    valid_df = _build_synthetic_history(
        num_bars=250, base_price=150.0, price_slope=0.5, end_date="2026-07-22"
    )
    data_provider.get_symbol_history.return_value = valid_df
    data_provider.get_latest_date.return_value = "2026-07-22"
    data_provider.get_batch_history.return_value = {
        "SPY": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [500.0, 490.0],
            }
        ),
        "TLT": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [90.0, 92.0],
            }
        ),
    }
    strategy._daily_universe_fip_cache = {"AAPL": 75.0}
    qualified = strategy.analyze_single_symbol("AAPL")
    assert qualified["status"] == "QUALIFIED"
    assert qualified["overall_passed"] is True

    # 3. Low FIP rank -> REJECTED
    strategy._daily_universe_fip_cache = {"AAPL": 30.0}
    low_fip = strategy.analyze_single_symbol("AAPL")
    assert low_fip["status"] == "REJECTED"

    # 4. Non-defensive macro with high FIP -> REJECTED
    strategy._daily_universe_fip_cache = {"AAPL": 80.0}
    data_provider.get_batch_history.return_value = {
        "SPY": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [500.0, 520.0],
            }
        ),
        "TLT": pd.DataFrame(
            {
                "date": [datetime.date(2026, 6, 30), datetime.date(2026, 7, 22)],
                "close": [90.0, 91.0],
            }
        ),
    }
    non_def = strategy.analyze_single_symbol("AAPL")
    assert non_def["status"] == "REJECTED"
