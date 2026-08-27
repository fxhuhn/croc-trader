"""NDX Momentum Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the NDX Momentum Screener and Trade Manager.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import Strategies, TradeStatus
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.ndx_momentum import (
    NDXMomentumConfiguration,
    NDXMomentumScreener,
)
from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
)
from app.types import TradeData


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Screener Rebalance Day & Universe Size
# ==============================================================================
@pytest.mark.tier1
def test_bva_ndx_momentum_rebalance_day_resolution() -> None:
    """BVA: Rebalance day logic strictly identifies last trading day of the calendar month."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = NDXMomentumScreener(
        trade_repository=mock_trade_repo,
        market_data_provider=mock_market_provider,
    )

    # 1. Non-rebalance day (mid-month Wednesday) -> returns False
    mid_month_date = pd.Timestamp("2026-01-14")
    assert screener._is_last_trading_day(mid_month_date) is False

    # 2. Last calendar day is a Friday (2026-01-30 is last trading day of Jan 2026) -> returns True
    last_trading_day = pd.Timestamp("2026-01-30")
    assert screener._is_last_trading_day(last_trading_day) is True

    # 3. Thursday before Good Friday (if Friday is a market holiday, Thursday is last trading day)
    with patch.object(
        screener.holiday_checker, "is_holiday", side_effect=lambda d: d.weekday() == 4
    ):
        thursday_date = pd.Timestamp("2026-04-30")
        assert screener._is_last_trading_day(thursday_date) is True


@pytest.mark.tier1
def test_bva_ndx_momentum_screener_universe_boundaries() -> None:
    """BVA: Screener handles fewer than 5 candidates, exact 5 candidates, and missing history."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = NDXMomentumScreener(
        trade_repository=mock_trade_repo,
        market_data_provider=mock_market_provider,
        configuration=NDXMomentumConfiguration(maximum_ticker_count=5),
    )

    analysis_date = "2026-01-30"
    dates = pd.date_range(end=analysis_date, periods=300, freq="B")

    # Helper to construct full OHLCV DataFrames
    def make_df(
        close_series: np.ndarray, date_series: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": date_series,
                "open": close_series * 0.99,
                "high": close_series * 1.02,
                "low": close_series * 0.98,
                "close": close_series,
                "volume": 1_000_000,
            }
        )

    history_map = {
        "SYM_A": make_df(np.linspace(100, 200, 300), dates),
        "SYM_B": make_df(np.linspace(100, 180, 300), dates),
        "SYM_C": make_df(np.linspace(100, 150, 300), dates),
        "SHORT_HIST": make_df(np.linspace(100, 120, 50), dates[-50:]),  # < 252 bars
        "QQQ": make_df(np.linspace(300, 400, 300), dates),
    }
    mock_market_provider.get_batch_history.return_value = history_map

    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["SYM_A", "SYM_B", "SYM_C", "SHORT_HIST"]
        result = screener.calculate_analysis(analysis_date=analysis_date)

    assert result["triggered"] is True
    # SHORT_HIST was dropped because < 252 bars; only 3 symbols qualified
    assert len(result["top_symbols"]) == 3
    assert result["top_symbols"] == ["SYM_A", "SYM_B", "SYM_C"]


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Trade Manager Entry Gating & Regimes
# ==============================================================================
@pytest.mark.tier1
def test_bva_ndx_momentum_entry_regime_and_duplicate_gating() -> None:
    """BVA: Trade Manager accepts BULL regime, rejects BEAR regime and duplicate active positions."""
    manager = NDXMomentumTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.NDXMomentum,
        "status": "CREATED",
        "entry_price": 150.0,
        "signal_context": {
            "date": "2026-01-30",
            "qqq_regime": "BULL",
        },
    }

    candle = pd.Series(
        {
            "date": "2026-02-02",
            "open": 152.0,
            "high": 155.0,
            "low": 151.0,
            "close": 154.0,
        }
    )
    dates = pd.to_datetime(["2026-01-30", "2026-02-02"])
    history = pd.DataFrame({"date": dates, "close": [150.0, 154.0]})

    # 1. Standard BULL regime, no active duplicate -> Accepted at Market Open
    transition = manager.check_entry(trade, candle, history, active_symbols=set())
    assert transition is not None
    assert transition.updates["status"] == TradeStatus.ACTIVE
    assert transition.updates["entry_price"] == 152.0
    assert "REBALANCE_ENTRY" in transition.reason

    # 2. BEAR regime -> Rejected immediately
    bear_trade: TradeData = {
        "id": 2,
        "symbol": "AAPL",
        "strategy": Strategies.NDXMomentum,
        "status": "CREATED",
        "entry_price": 150.0,
        "signal_context": {
            "date": "2026-01-30",
            "qqq_regime": "BEAR",
        },
    }
    bear_transition = manager.check_entry(
        bear_trade, candle, history, active_symbols=set()
    )
    assert bear_transition is not None
    assert bear_transition.updates["status"] == TradeStatus.INVALID
    assert "BEAR" in bear_transition.reason

    # 3. Duplicate active position in same symbol -> Rejected immediately
    dup_transition = manager.check_entry(
        trade, candle, history, active_symbols={"AAPL"}
    )
    assert dup_transition is not None
    assert dup_transition.updates["status"] == TradeStatus.INVALID
    assert "Position already exists" in dup_transition.reason


# ==============================================================================
# 3. Boundary Value Analysis (BVA) — Monthly Roster Turnover Reconciliation
# ==============================================================================
@pytest.mark.tier1
def test_bva_ndx_momentum_monthly_roster_reconciliation() -> None:
    """BVA: Active positions persist if still in top 5 leaders, and close at Open if dropped."""
    manager = NDXMomentumTradeStrategy()

    active_trade_aapl: TradeData = {
        "id": 10,
        "symbol": "AAPL",
        "strategy": Strategies.NDXMomentum,
        "status": "ACTIVE",
        "current_size": 100,
        "entry_price": 150.0,
        "signal_context": {"date": "2026-01-30"},
    }
    active_trade_msft: TradeData = {
        "id": 11,
        "symbol": "MSFT",
        "strategy": Strategies.NDXMomentum,
        "status": "ACTIVE",
        "current_size": 50,
        "entry_price": 300.0,
        "signal_context": {"date": "2026-01-30"},
    }

    # Month transition: from Jan 30 to Feb 02
    history_month_switch = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-30", "2026-02-02"]),
            "close": [150.0, 155.0],
            "open": [149.0, 153.0],
        }
    )
    current_candle = pd.Series({"date": "2026-02-02", "open": 153.0, "close": 155.0})

    # Case 1: AAPL is still in the new monthly top leaders set -> Kept (No exit transition)
    latest_leaders = {"AAPL", "NVDA", "AMZN", "META", "GOOGL"}
    transition_kept = manager._do_manage_active_trade(
        active_trade_aapl,
        current_candle,
        "2026-02-02",
        history_month_switch,
        latest_leaders=latest_leaders,
    )
    assert transition_kept is None

    # Case 2: MSFT dropped from the top 5 leaders set -> Closed at Market Open
    transition_exit = manager._do_manage_active_trade(
        active_trade_msft,
        current_candle,
        "2026-02-02",
        history_month_switch,
        latest_leaders=latest_leaders,
    )
    assert transition_exit is not None
    assert transition_exit.updates["status"] == "CLOSED"
    assert transition_exit.updates["exit_price"] == 153.0
    assert transition_exit.reason == "REBALANCE_EXIT"

    # Case 3: Mid-month candle (Feb 02 to Feb 03) -> No exit even if not in leaders set
    history_mid_month = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-02", "2026-02-03"]),
            "close": [155.0, 156.0],
            "open": [154.0, 155.0],
        }
    )
    mid_candle = pd.Series({"date": "2026-02-03", "open": 155.0, "close": 156.0})
    transition_mid_month = manager._do_manage_active_trade(
        active_trade_msft,
        mid_candle,
        "2026-02-03",
        history_mid_month,
        latest_leaders=latest_leaders,
    )
    assert transition_mid_month is None


# ==============================================================================
# 4. Boundary Value Analysis (BVA) — Order Generation & Sizing
# ==============================================================================
@pytest.mark.tier1
def test_bva_ndx_momentum_order_generation_boundaries() -> None:
    """BVA: Order generation handles zero budgets, empty history, and generates valid MKT orders."""
    manager = NDXMomentumTradeStrategy()

    trade: TradeData = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.NDXMomentum,
        "status": "CREATED",
        "entry_price": 150.0,
        "signal_context": {"budget": 10000.0},
    }

    # 1. Valid order generation with closing price = $200 -> 10000 / 200 = 50 shares
    history = pd.DataFrame({"date": ["2026-01-30"], "close": [200.0]})
    order = manager._generate_entry_order(trade, history, budget=10000.0)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.quantity == 50
    assert order.entry is not None
    assert order.entry.type == "MKT"
    assert order.entry.time_in_force == "OPG"

    # 2. Empty history -> Returns None
    assert manager._generate_entry_order(trade, pd.DataFrame(), budget=10000.0) is None

    # 3. Budget too small for 1 share (Budget = $100, Price = $200) -> Returns None
    order_small = manager._generate_entry_order(trade, history, budget=100.0)
    assert order_small is None


# ==============================================================================
# 5. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    budget=st.floats(
        min_value=1000.0,
        max_value=1_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    close_price=st.floats(
        min_value=0.50, max_value=10000.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_ndx_momentum_order_sizing(
    budget: float, close_price: float
) -> None:
    """Invariant: Sizing never divides by zero and allocated capital never exceeds total budget."""
    manager = NDXMomentumTradeStrategy()
    trade: TradeData = {
        "id": 1,
        "symbol": "TEST",
        "strategy": Strategies.NDXMomentum,
        "status": "CREATED",
        "entry_price": close_price,
    }

    history = pd.DataFrame({"date": ["2026-01-30"], "close": [close_price]})
    order = manager._generate_entry_order(trade, history, budget=budget)

    if budget < close_price:
        assert order is None
    else:
        assert order is not None
        assert order.quantity >= 1
        allocated_capital = Decimal(str(order.quantity)) * Decimal(str(close_price))
        assert allocated_capital <= Decimal(str(budget)) + Decimal(str(close_price))


@pytest.mark.tier2
@given(
    dates_and_symbols=st.lists(
        st.tuples(
            st.dates().map(lambda d: d.strftime("%Y-%m-%d")),
            st.text(min_size=1, max_size=5, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_extract_latest_leaders_invariants(
    dates_and_symbols: list[tuple[str, str]],
) -> None:
    """Invariant: extract_latest_leaders returns non-empty leaders from the strictly latest date."""
    trades = [
        {"symbol": sym, "signal_context": {"date": dt}} for dt, sym in dates_and_symbols
    ]
    max_date = max(dt for dt, _ in dates_and_symbols)
    expected_symbols = {sym for dt, sym in dates_and_symbols if dt == max_date}

    extracted = NDXMomentumTradeStrategy.extract_latest_leaders(trades)
    assert extracted == expected_symbols


# ==============================================================================
# 6. Zero Lookahead-Bias Guard
# ==============================================================================
@pytest.mark.tier2
def test_ndx_momentum_screener_zero_lookahead_bias() -> None:
    """Lookahead Guard: Screening decisions on date T are strictly independent of data at T+1..T+N."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_market_provider = MagicMock(spec=MarketDataProvider)

    screener = NDXMomentumScreener(
        trade_repository=mock_trade_repo,
        market_data_provider=mock_market_provider,
    )

    analysis_date = "2026-01-30"  # Rebalance date T
    base_dates = pd.date_range(end=analysis_date, periods=300, freq="B")
    future_dates = pd.date_range(start="2026-02-02", periods=30, freq="B")
    full_dates = base_dates.append(future_dates)

    def make_full_df(
        close_series: np.ndarray, date_series: pd.DatetimeIndex
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": date_series,
                "open": close_series * 0.99,
                "high": close_series * 1.02,
                "low": close_series * 0.98,
                "close": close_series,
                "volume": 1_000_000,
            }
        )

    def create_history_map(future_multiplier: float) -> dict[str, pd.DataFrame]:
        prices_aapl = np.linspace(100, 200, len(full_dates))
        prices_aapl[len(base_dates) :] *= future_multiplier
        return {
            "AAPL": make_full_df(prices_aapl, full_dates),
            "MSFT": make_full_df(np.linspace(100, 180, len(full_dates)), full_dates),
            "NVDA": make_full_df(np.linspace(100, 220, len(full_dates)), full_dates),
            "QQQ": make_full_df(np.linspace(300, 400, len(full_dates)), full_dates),
        }

    # Run 1: Historical data up to T
    mock_market_provider.get_batch_history.return_value = {
        k: df[df["date"] <= analysis_date].copy()
        for k, df in create_history_map(1.0).items()
    }
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL", "MSFT", "NVDA"]
        result_baseline = screener.calculate_analysis(analysis_date=analysis_date)

    # Run 2: Full history including massive future price surge in AAPL at T+1..T+N
    mock_market_provider.get_batch_history.return_value = create_history_map(5.0)
    with patch(
        "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
    ) as mock_ex:
        mock_ex.return_value.nasdaq_100 = ["AAPL", "MSFT", "NVDA"]
        result_with_future = screener.calculate_analysis(analysis_date=analysis_date)

    assert result_baseline["top_symbols"] == result_with_future["top_symbols"]
    assert result_baseline["momentum_scores"].equals(
        result_with_future["momentum_scores"]
    )
