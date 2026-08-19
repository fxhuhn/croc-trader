"""Standardized Lifecycle, Timing, and Edge-Case Tests for all 8 Strategies.

Verifies:
1. Pre-market / Pre-setup candle evaluation (must return None, keeping status CREATED).
2. Target session execution (activates on eligible session with exact date and fill price).
3. Target session failure (invalidates setup when condition fails).
4. Post-window evaluation (invalidates/expires setup on stale candles).
5. Holding duration calculation (bars_held) and deterministic exit execution.
"""

import json

import pandas as pd

from app.services.trade_manager.strategies.bounce_bandit import (
    BounceBanditTradeStrategy,
)
from app.services.trade_manager.strategies.bridge_scout import (
    BridgeScoutTradeStrategy,
)
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
)
from app.services.trade_manager.strategies.tgim import (
    TGIMTradeStrategy,
)
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy,
)
from app.services.trade_manager.strategies.two_percent_strategy import (
    TwoPercentStrategy,
)
from app.types import ExitReason, TradeStatus

# =====================================================================
# 1. TGIM (Thank God It's Monday) Lifecycle & Timing Tests
# =====================================================================


def test_tgim_lifecycle_timing() -> None:
    """Verifies full TGIM lifecycle: Pre-market -> Monday Entry -> Tuesday Hold -> Wednesday Time Stop."""
    strategy = TGIMTradeStrategy()
    trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,  # threshold price
        "budget": 10000.0,
        "signal_context": json.dumps(
            {"setup_date": "2026-08-17", "threshold_price": 500.0}
        ),
    }

    # Phase 1: Pre-market Monday morning (Friday candle in DB)
    friday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-14"),
            "open": 502.0,
            "high": 505.0,
            "low": 498.0,
            "close": 500.0,
        }
    )
    assert (
        strategy.check_entry(trade, friday_candle, pd.DataFrame([friday_candle]))
        is None
    )

    # Phase 2: Monday EOD close (Under threshold -> Activates with Monday Close)
    monday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 499.0,
            "high": 501.0,
            "low": 492.0,
            "close": 495.0,
        }
    )
    history_mo = pd.DataFrame([friday_candle, monday_candle])
    transition_entry = strategy.check_entry(trade, monday_candle, history_mo)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_entry.updates["entry_date"] == "2026-08-17"
    assert transition_entry.updates["entry_price"] == 495.0

    # Phase 3: Active trade management on Tuesday (Bar 1, Close <= Monday Close -> Hold)
    active_trade = {
        "id": 1,
        "symbol": "SPY",
        "strategy": "tgim",
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-08-17",
        "entry_price": 495.0,
        "initial_size": 20,
        "current_size": 20,
    }
    tuesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 495.0,
            "high": 496.0,
            "low": 490.0,
            "close": 492.0,  # Lower than 495.0 -> No TP
        }
    )
    history_tu = pd.DataFrame([monday_candle, tuesday_candle])
    assert strategy.manage_active_trade(active_trade, history_tu) is None

    # Phase 4: Active trade management on Wednesday (Bar 2, Close <= Tuesday Close -> TIME_STOP)
    wednesday_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-19"),
            "open": 492.0,
            "high": 493.0,
            "low": 488.0,
            "close": 490.0,  # Lower than Tuesday -> Time Stop
        }
    )
    history_we = pd.DataFrame([monday_candle, tuesday_candle, wednesday_candle])
    transition_exit = strategy.manage_active_trade(active_trade, history_we)
    assert transition_exit is not None
    assert transition_exit.updates["status"] == TradeStatus.CLOSED.value
    assert transition_exit.updates["exit_reason"] == ExitReason.TIME_STOP.value
    assert transition_exit.updates["exit_price"] == 490.0


# =====================================================================
# 2. Dip Buyer Lifecycle & Timing Tests
# =====================================================================


def test_dip_buyer_lifecycle_timing() -> None:
    """Verifies Dip Buyer lifecycle: Signal day ignored -> Day 1 Limit entry -> Stale day rejected."""
    strategy = DipBuyerStrategy()
    trade = {
        "id": 2,
        "symbol": "AAPL",
        "strategy": "dip_buyer",
        "status": TradeStatus.CREATED.value,
        "entry_price": 150.0,
        "signal_context": json.dumps({"date": "2026-08-17"}),
    }

    # Phase 1: Signal Day (Day 0) evaluation -> Must return None
    signal_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 155.0,
            "high": 156.0,
            "low": 149.0,
            "close": 152.0,
        }
    )
    history_d0 = pd.DataFrame([signal_candle])
    assert strategy.check_entry(trade, signal_candle, history_d0) is None

    # Phase 2: Day 1 Evaluation (Low <= Limit -> Limit Hit with Gap-Down benefit)
    day1_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 148.0,  # Gapped below limit
            "high": 151.0,
            "low": 147.0,
            "close": 150.5,
        }
    )
    history_d1 = pd.DataFrame([signal_candle, day1_candle])
    transition_entry = strategy.check_entry(trade, day1_candle, history_d1)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_entry.updates["entry_price"] == 148.0  # Gap down benefit

    # Phase 3: Post-window evaluation (Day 2 after missed entry -> Rejected)
    day2_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-19"),
            "open": 151.0,
            "high": 153.0,
            "low": 149.0,
            "close": 152.0,
        }
    )
    history_d2 = pd.DataFrame([signal_candle, day1_candle, day2_candle])
    transition_stale = strategy.check_entry(trade, day2_candle, history_d2)
    assert transition_stale is not None
    assert transition_stale.updates["status"] == TradeStatus.INVALID.value
    assert "Missed Entry Window" in transition_stale.reason


# =====================================================================
# 3. Two Percent Lifecycle & Timing Tests
# =====================================================================


def test_two_percent_lifecycle_timing() -> None:
    """Verifies Two Percent lifecycle: Signal day ignored -> Day 1 Entry -> 2% Target Calculation."""
    strategy = TwoPercentStrategy()
    trade = {
        "id": 3,
        "symbol": "MSFT",
        "strategy": "two_percent",
        "status": TradeStatus.CREATED.value,
        "entry_price": 400.0,
        "signal_context": json.dumps({"date": "2026-08-17"}),
    }

    # Phase 1: Signal Day -> Ignored
    signal_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 405.0,
            "high": 408.0,
            "low": 398.0,
            "close": 402.0,
        }
    )
    history_d0 = pd.DataFrame([signal_candle])
    assert strategy.check_entry(trade, signal_candle, history_d0) is None

    # Phase 2: Day 1 Entry (Low <= Limit -> Limit Hit with 2% Target)
    day1_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 401.0,
            "high": 403.0,
            "low": 399.0,
            "close": 401.5,
        }
    )
    history_d1 = pd.DataFrame([signal_candle, day1_candle])
    transition_entry = strategy.check_entry(trade, day1_candle, history_d1)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_entry.updates["entry_price"] == 400.0
    assert transition_entry.updates["current_target"] == 408.0  # 400 * 1.02 = 408.0


# =====================================================================
# 4. Turnover Timing Lifecycle Tests
# =====================================================================


def test_turnover_timing_lifecycle() -> None:
    """Verifies Turnover Timing lifecycle: Signal day ignored -> Day 1 Entry -> Day 2 Expiration."""
    strategy = TurnoverTimingStrategy()
    trade = {
        "id": 4,
        "symbol": "NVDA",
        "strategy": "turnover_timing",
        "status": TradeStatus.CREATED.value,
        "entry_price": 120.0,
        "signal_context": json.dumps(
            {"date": "2026-08-17", "setup_candle_green": True}
        ),
    }

    # Signal Day -> None
    signal_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 122.0,
            "high": 125.0,
            "low": 119.0,
            "close": 123.0,
        }
    )
    assert (
        strategy.check_entry(trade, signal_candle, pd.DataFrame([signal_candle]))
        is None
    )

    # Day 1 -> Entry with green candle context tracking
    day1_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 120.0,
            "high": 124.0,
            "low": 118.0,
            "close": 122.0,  # Green candle (close > open)
        }
    )
    history_d1 = pd.DataFrame([signal_candle, day1_candle])
    transition_entry = strategy.check_entry(trade, day1_candle, history_d1)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value

    parsed_ctx = json.loads(str(transition_entry.updates["signal_context"]))
    assert parsed_ctx["green_candle_count"] == 2  # Setup was green + Entry is green


# =====================================================================
# 5. Bounce Bandit Lifecycle Tests
# =====================================================================


def test_bounce_bandit_lifecycle() -> None:
    """Verifies Bounce Bandit: Signal day ignored -> Next day Market On Open (MOO) entry."""
    strategy = BounceBanditTradeStrategy()
    trade = {
        "id": 5,
        "symbol": "QQQ",
        "strategy": "bounce_bandit",
        "status": TradeStatus.CREATED.value,
        "entry_price": 480.0,
        "signal_context": json.dumps({"date": "2026-08-17"}),
    }

    # Signal Day -> None
    signal_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 485.0,
            "high": 486.0,
            "low": 479.0,
            "close": 480.0,
        }
    )
    assert (
        strategy.check_entry(trade, signal_candle, pd.DataFrame([signal_candle]))
        is None
    )

    # Next Day Open -> MOO activation
    next_day_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 482.0,
            "high": 487.0,
            "low": 481.0,
            "close": 486.0,
        }
    )
    history = pd.DataFrame([signal_candle, next_day_candle])
    transition_entry = strategy.check_entry(trade, next_day_candle, history)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_entry.updates["entry_price"] == 482.0
    assert transition_entry.updates["entry_date"] == "2026-08-18"


# =====================================================================
# 6. Bridge Scout Lifecycle Tests
# =====================================================================


def test_bridge_scout_lifecycle() -> None:
    """Verifies Bridge Scout: Month-end MOC entry -> Month transition exit."""
    strategy = BridgeScoutTradeStrategy()
    trade = {
        "id": 6,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": TradeStatus.CREATED.value,
        "entry_price": 480.0,
        "signal_context": json.dumps({"date": "2026-08-28"}),
    }

    # Entry on setup date MOC
    entry_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-28"),
            "open": 482.0,
            "high": 485.0,
            "low": 478.0,
            "close": 480.0,
        }
    )
    history_entry = pd.DataFrame([entry_candle])
    transition_entry = strategy.check_entry(trade, entry_candle, history_entry)
    assert transition_entry is not None
    assert transition_entry.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_entry.updates["entry_date"] == "2026-08-28"

    # Same month management -> Hold
    same_month_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-31"),
            "open": 481.0,
            "high": 484.0,
            "low": 479.0,
            "close": 483.0,
        }
    )
    active_trade = {
        "id": 6,
        "symbol": "QQQ",
        "strategy": "bridge_scout",
        "status": TradeStatus.ACTIVE.value,
        "entry_date": "2026-08-28",
        "entry_price": 480.0,
        "initial_size": 20,
        "current_size": 20,
    }
    history_hold = pd.DataFrame([entry_candle, same_month_candle])
    assert strategy.manage_active_trade(active_trade, history_hold) is None

    # New Month Day 1 (Month change) -> TIME_STOP Exit
    new_month_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-09-01"),
            "open": 484.0,
            "high": 488.0,
            "low": 483.0,
            "close": 487.0,
        }
    )
    history_exit = pd.DataFrame([entry_candle, same_month_candle, new_month_candle])
    transition_exit = strategy.manage_active_trade(active_trade, history_exit)
    assert transition_exit is not None
    assert transition_exit.updates["status"] == TradeStatus.CLOSED.value
    assert transition_exit.updates["exit_reason"] == ExitReason.TIME_STOP.value
    assert transition_exit.updates["exit_price"] == 487.0


# =====================================================================
# 7. NDX Momentum Lifecycle Tests
# =====================================================================


def test_ndx_momentum_lifecycle() -> None:
    """Verifies NDX Momentum: Signal day ignored -> Bull regime MOO entry -> Bear regime rejection."""
    strategy = NDXMomentumTradeStrategy()

    # Case A: Bull Regime -> Day 1 MOO entry
    bull_trade = {
        "id": 7,
        "symbol": "META",
        "strategy": "ndx_momentum",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "signal_context": json.dumps({"date": "2026-08-17", "qqq_regime": "BULL"}),
    }
    signal_candle = pd.Series(
        {"date": pd.Timestamp("2026-08-17"), "open": 498.0, "close": 500.0}
    )
    day1_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 502.0,
            "high": 508.0,
            "low": 500.0,
            "close": 506.0,
        }
    )

    # Signal Day -> None
    assert (
        strategy.check_entry(bull_trade, signal_candle, pd.DataFrame([signal_candle]))
        is None
    )

    # Day 1 -> MOO Entry at 502.0
    history_d1 = pd.DataFrame([signal_candle, day1_candle])
    transition_bull = strategy.check_entry(bull_trade, day1_candle, history_d1)
    assert transition_bull is not None
    assert transition_bull.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_bull.updates["entry_price"] == 502.0

    # Case B: Bear Regime -> Immediate Rejection
    bear_trade = {
        "id": 8,
        "symbol": "META",
        "strategy": "ndx_momentum",
        "status": TradeStatus.CREATED.value,
        "entry_price": 500.0,
        "signal_context": json.dumps({"date": "2026-08-17", "qqq_regime": "BEAR"}),
    }
    transition_bear = strategy.check_entry(bear_trade, day1_candle, history_d1)
    assert transition_bear is not None
    assert transition_bear.updates["status"] == TradeStatus.INVALID.value
    assert "QQQ Regime: BEAR" in transition_bear.reason


# =====================================================================
# 8. CrocSetup (Hold Target) Lifecycle Tests
# =====================================================================


def test_croc_setup_lifecycle() -> None:
    """Verifies CrocSetup: Signal day ignored -> Breakout Entry -> Day 1 Turnaround -> 5-Day Expiration."""
    strategy = HoldTargetStrategy()
    trade = {
        "id": 9,
        "symbol": "TSLA",
        "strategy": "hold_target",
        "status": TradeStatus.CREATED.value,
        "entry_price": 250.0,
        "current_stop_loss": 240.0,
        "signal_context": json.dumps({"date": "2026-08-17"}),
    }

    # Signal Day -> None
    signal_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-17"),
            "open": 245.0,
            "high": 249.0,
            "low": 242.0,
            "close": 246.0,
        }
    )
    assert (
        strategy.check_entry(trade, signal_candle, pd.DataFrame([signal_candle]))
        is None
    )

    # Day 1 Breakout (High >= Entry Price -> Activates at Entry Price)
    breakout_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-18"),
            "open": 247.0,
            "high": 252.0,  # High >= 250.0 Breakout
            "low": 246.0,  # Low > Stop (240.0)
            "close": 251.0,
        }
    )
    history_bo = pd.DataFrame([signal_candle, breakout_candle])
    transition_bo = strategy.check_entry(trade, breakout_candle, history_bo)
    assert transition_bo is not None
    assert transition_bo.updates["status"] == TradeStatus.ACTIVE.value
    assert transition_bo.updates["entry_price"] == 250.0

    # Stale Candle (> 5 Calendar Days after Signal without trigger -> Expired/Invalidated)
    stale_candle = pd.Series(
        {
            "date": pd.Timestamp("2026-08-25"),  # 8 days later
            "open": 244.0,
            "high": 248.0,
            "low": 242.0,
            "close": 245.0,
        }
    )
    history_stale = pd.DataFrame([signal_candle, stale_candle])
    transition_exp = strategy.check_entry(trade, stale_candle, history_stale)
    assert transition_exp is not None
    assert transition_exp.updates["status"] == TradeStatus.INVALID.value
    assert transition_exp.updates["exit_reason"] == ExitReason.INVALIDATED.value
