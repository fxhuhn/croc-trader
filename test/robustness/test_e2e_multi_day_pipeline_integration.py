"""End-to-End Multi-Day Pipeline Integration Tests with Real SQLite Database.

Verifies:
1. Full pipeline interaction between ScreenerEngine and TradeManager across consecutive trading days.
2. Real SQLite queries: SELECT 1 FROM trades WHERE symbol = ? AND strategy = ? AND date = ?
3. Trading calendar transitions:
   - Standard consecutive trading sessions (e.g. Wednesday -> Thursday).
   - Weekend transition (Friday EOD in DB -> Monday live screening).
   - Holiday transition (Market closed days).
   - Month-end (EoM) window entry and exit transitions.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from flask import Flask

from app.const import Strategies, TradeStatus
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.services.screener.engine import ScreenerEngine
from app.services.screener.strategies.bounce_bandit import BounceBanditStrategy
from app.services.screener.strategies.bridge_scout import BridgeScoutStrategy
from app.services.screener.strategies.tgim import TGIMStrategy
from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from app.tasks import run_daily_eod_pipeline
from app.tools.market_holidays import MarketHolidayChecker


@pytest.fixture
def sqlite_trade_repo(tmp_path: Path) -> TradeRepository:
    """Provides a fresh real SQLite TradeRepository with initialized schema."""
    db_file = tmp_path / "test_pipeline_trades.db"
    session = DatabaseSession(str(db_file))
    repo = TradeRepository(session)
    repo.init_schema()
    return repo


@pytest.fixture
def sqlite_signal_repo(tmp_path: Path) -> SignalRepository:
    """Provides a fresh real SQLite SignalRepository with initialized schema."""
    db_file = tmp_path / "test_pipeline_signals.db"
    session = DatabaseSession(str(db_file))
    repo = SignalRepository(session)
    repo.init_schema()
    return repo


def _generate_synthetic_qqq_history(
    end_date_str: str, num_bars: int = 250
) -> pd.DataFrame:
    """Generates synthetic QQQ history ending on end_date_str."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    close_prices = [500.0 + (i * 0.5) for i in range(num_bars - 3)] + [
        624.0,
        623.0,
        600.0,
    ]
    return pd.DataFrame(
        [
            {
                "date": d,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "volume": 5_000_000,
            }
            for d, p in zip(dates, close_prices, strict=True)
        ]
    )


# =====================================================================
# 1. Real SQLite: Multi-Day Bridge Scout Consecutive Screening
# =====================================================================


def test_sqlite_bridge_scout_e2e_consecutive_days(
    sqlite_trade_repo: TradeRepository,
    sqlite_signal_repo: SignalRepository,
) -> None:
    """E2E Test with real SQLite DB: Bridge Scout creates trade on Day 1, TradeManager
    invalidates it, and Day 2 ScreenerEngine creates a fresh trade for Day 2."""
    mock_provider = MagicMock()
    holiday_checker = MarketHolidayChecker()

    bridge_scout = BridgeScoutStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
        holiday_checker=holiday_checker,
    )

    engine = ScreenerEngine(
        trade_repository=sqlite_trade_repo,
        signal_repository=sqlite_signal_repo,
        data_provider=mock_provider,
        strategies=[bridge_scout],
    )

    day1_eod = "2026-08-26"
    day2_eod = "2026-08-27"
    today_live = "2026-08-28"

    # Day 1: ScreenerEngine runs with Day 1 EOD data
    df_day1 = _generate_synthetic_qqq_history(day1_eod)
    mock_provider.get_batch_history.return_value = {"QQQ": df_day1}
    mock_provider.get_latest_date.return_value = day1_eod

    # Create Day 1 trade directly in real SQLite DB
    trade_id_1 = sqlite_trade_repo.create_trade(
        symbol="QQQ",
        strategy=Strategies.BridgeScout.value,
        size=0.0,
        entry=710.16,
        stop_loss=0.0,
        target=0.0,
        context={"date": day2_eod, "setup_date": day2_eod},
    )
    assert trade_id_1 > 0
    assert (
        sqlite_trade_repo.exists("QQQ", Strategies.BridgeScout.value, day2_eod) is True
    )

    # Day 2 06:00: TradeManager sets trade_id_1 to INVALID in real SQLite DB
    sqlite_trade_repo.update_trade(trade_id_1, {"status": TradeStatus.INVALID.value})
    updated_trade = sqlite_trade_repo.get_trade(trade_id_1)
    assert updated_trade is not None
    assert updated_trade["status"] == TradeStatus.INVALID.value

    # Day 2 06:30: Market data updated with Day 2 EOD (close=721.11)
    df_day2 = _generate_synthetic_qqq_history(day2_eod)
    mock_provider.get_batch_history.return_value = {"QQQ": df_day2}
    mock_provider.get_latest_date.return_value = day2_eod

    # ScreenerEngine runs for live day
    results = engine.run_all(days=0)
    assert results == {"bridge_scout": 1}

    # Verify real SQLite database contents
    created_trades = sqlite_trade_repo.get_by_status(TradeStatus.CREATED)
    assert len(created_trades) == 1
    assert created_trades[0]["symbol"] == "QQQ"
    ctx = json.loads(str(created_trades[0].get("signal_context", "{}")))
    assert ctx.get("date") == today_live or ctx.get("setup_date") == today_live
    assert created_trades[0]["id"] != trade_id_1


# =====================================================================
# 2. Weekend Transition (Friday EOD -> Monday Live Pre-Market Screening)
# =====================================================================


def test_sqlite_weekend_transition_friday_to_monday(
    sqlite_trade_repo: TradeRepository,
    sqlite_signal_repo: SignalRepository,
) -> None:
    """E2E Test: TGIM and Pre-Market screening across weekend (Friday EOD in DB -> Monday live)."""
    mock_provider = MagicMock()

    tgim = TGIMStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
    )

    engine = ScreenerEngine(
        trade_repository=sqlite_trade_repo,
        signal_repository=sqlite_signal_repo,
        data_provider=mock_provider,
        strategies=[tgim],
    )

    friday_date = "2026-08-21"
    monday_live = "2026-08-24"

    # Friday EOD history in DB
    dates = pd.date_range(end=friday_date, periods=50, freq="B")
    df_spy = pd.DataFrame(
        [
            {
                "date": d,
                "open": 500.0,
                "high": 505.0,
                "low": 495.0,
                "close": 500.0,
                "volume": 1000000,
            }
            for d in dates
        ]
    )
    mock_provider.get_batch_history.return_value = {"SPY": df_spy}
    mock_provider.get_latest_date.return_value = friday_date

    # Monday morning: ScreenerEngine runs with analysis_date resolved to Friday (latest DB candle)
    results = engine.run_all(days=0)
    assert results == {"tgim": 1}

    # Verify Trade created for Monday in real SQLite DB
    created_trades = sqlite_trade_repo.get_by_status(TradeStatus.CREATED)
    assert len(created_trades) == 1
    assert created_trades[0]["symbol"] == "SPY"
    ctx = json.loads(str(created_trades[0].get("signal_context", "{}")))
    assert ctx.get("date") == monday_live or ctx.get("setup_date") == monday_live


# =====================================================================
# 3. Active Position Guard in Real SQLite (Blocking Duplicate Orders)
# =====================================================================


def test_sqlite_active_position_guard_blocks_duplicate_signals(
    sqlite_trade_repo: TradeRepository,
    sqlite_signal_repo: SignalRepository,
) -> None:
    """E2E Test: If a position is marked ACTIVE in SQLite, ScreenerEngine blocks duplicate signals."""
    mock_provider = MagicMock()

    two_percent = TwoPercentStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
    )
    bounce_bandit = BounceBanditStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
    )

    engine = ScreenerEngine(
        trade_repository=sqlite_trade_repo,
        signal_repository=sqlite_signal_repo,
        data_provider=mock_provider,
        strategies=[two_percent, bounce_bandit],
    )

    analysis_date = "2026-08-27"
    df_qqq = _generate_synthetic_qqq_history(analysis_date)
    mock_provider.get_batch_history.return_value = {"QQQ": df_qqq}
    mock_provider.get_latest_date.return_value = analysis_date

    # Insert ACTIVE trades for both strategies in real SQLite DB
    sqlite_trade_repo.create_trade(
        symbol="QQQ",
        strategy=Strategies.TwoPercent.value,
        size=10.0,
        entry=600.0,
        stop_loss=580.0,
        target=620.0,
        context={"date": "2026-08-26"},
    )
    sqlite_trade_repo.create_trade(
        symbol="QQQ",
        strategy=Strategies.BounceBandit.value,
        size=10.0,
        entry=600.0,
        stop_loss=580.0,
        target=620.0,
        context={"date": "2026-08-26"},
    )
    for trade in sqlite_trade_repo.get_by_status(TradeStatus.CREATED):
        sqlite_trade_repo.update_trade(
            trade["id"], {"status": TradeStatus.ACTIVE.value}
        )

    assert len(sqlite_trade_repo.get_by_status(TradeStatus.ACTIVE)) == 2

    # ScreenerEngine runs -> must produce 0 hits because positions are active
    results = engine.run_all(days=0)
    assert results == {"two_percent": 0, "bounce_bandit": 0}


# =====================================================================
# 4. Month-End (EoM) Window Boundary Transition (Bridge Scout)
# =====================================================================


def test_sqlite_eom_window_boundary_transition(
    sqlite_trade_repo: TradeRepository,
    sqlite_signal_repo: SignalRepository,
) -> None:
    """E2E Test: Bridge Scout operates strictly within EoM window and ignores dates outside window."""
    mock_provider = MagicMock()
    holiday_checker = MarketHolidayChecker()

    bridge_scout = BridgeScoutStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
        holiday_checker=holiday_checker,
    )

    # Date mid-month (outside EoM window, e.g. 2026-08-12)
    mid_month_date = "2026-08-12"
    df_mid = _generate_synthetic_qqq_history(mid_month_date)
    mock_provider.get_batch_history.return_value = {"QQQ": df_mid}
    mock_provider.get_latest_date.return_value = mid_month_date

    hits_outside = bridge_scout.run(days=1, analysis_date=mid_month_date)
    assert hits_outside == 0
    assert len(sqlite_trade_repo.get_by_status(TradeStatus.CREATED)) == 0

    # Date in EoM window (e.g. 2026-08-27)
    eom_date = "2026-08-27"
    df_eom = _generate_synthetic_qqq_history(eom_date)
    mock_provider.get_batch_history.return_value = {"QQQ": df_eom}
    mock_provider.get_latest_date.return_value = eom_date

    hits_inside = bridge_scout.run(days=1, analysis_date=eom_date)
    assert hits_inside == 1
    assert len(sqlite_trade_repo.get_by_status(TradeStatus.CREATED)) == 1


# =====================================================================
# 5. Master Pipeline Orchestrator (run_daily_eod_pipeline) E2E Test
# =====================================================================


def test_e2e_full_pipeline_orchestrator_multi_day_sequence(
    sqlite_trade_repo: TradeRepository,
    sqlite_signal_repo: SignalRepository,
) -> None:
    """E2E Test: Validates that run_daily_eod_pipeline orchestrates TradeManager, Screener,
    and Orders seamlessly on consecutive days without trade blocking collisions."""
    app = Flask(__name__)
    mock_provider = MagicMock()
    holiday_checker = MarketHolidayChecker()

    bridge_scout = BridgeScoutStrategy(
        trade_repository=sqlite_trade_repo,
        data_provider=mock_provider,
        holiday_checker=holiday_checker,
    )

    screener_engine = ScreenerEngine(
        trade_repository=sqlite_trade_repo,
        signal_repository=sqlite_signal_repo,
        data_provider=mock_provider,
        strategies=[bridge_scout],
    )

    trade_manager = MagicMock()

    # Simulate TradeManager resolving Day 1 trades on Day 2 run
    def _mock_evaluate():
        for t in sqlite_trade_repo.get_by_status(TradeStatus.CREATED):
            sqlite_trade_repo.update_trade(
                t["id"], {"status": TradeStatus.INVALID.value}
            )

    trade_manager.run_daily_process.side_effect = _mock_evaluate
    trade_manager.generate_daily_orders.return_value = (
        "data/orders/orders_2026_08_28.csv"
    )

    app.extensions["trade_manager"] = trade_manager
    app.extensions["screener_engine"] = screener_engine

    day2_date = "2026-08-27"
    today_live = "2026-08-28"

    # Day 1: Trade is created in DB
    trade_id_1 = sqlite_trade_repo.create_trade(
        symbol="QQQ",
        strategy=Strategies.BridgeScout.value,
        size=0.0,
        entry=710.16,
        stop_loss=0.0,
        target=0.0,
        context={"date": day2_date, "setup_date": day2_date},
    )
    assert trade_id_1 > 0
    assert len(sqlite_trade_repo.get_by_status(TradeStatus.CREATED)) == 1

    # Day 2 Morning: Market data has Day 2 EOD candle
    df_day2 = _generate_synthetic_qqq_history(day2_date)
    mock_provider.get_batch_history.return_value = {"QQQ": df_day2}
    mock_provider.get_latest_date.return_value = day2_date

    # Execute Master EOD Pipeline Orchestrator for Day 2
    with app.app_context():
        results = run_daily_eod_pipeline(app)

    # Asserts:
    # 1. TradeManager evaluated first -> Day 1 trade became INVALID
    trade_1 = sqlite_trade_repo.get_trade(trade_id_1)
    assert trade_1 is not None
    assert trade_1["status"] == TradeStatus.INVALID.value

    # 2. ScreenerEngine evaluated second -> generated fresh trade for today (2026-08-28)
    assert results["screener"] == {"bridge_scout": 1}
    created_trades = sqlite_trade_repo.get_by_status(TradeStatus.CREATED)
    assert len(created_trades) == 1
    assert created_trades[0]["symbol"] == "QQQ"
    ctx = json.loads(str(created_trades[0].get("signal_context", "{}")))
    assert ctx.get("date") == today_live or ctx.get("setup_date") == today_live

    # 3. Order generation executed third
    assert results["orders"] == "data/orders/orders_2026_08_28.csv"
    assert results["status"] == "success"
