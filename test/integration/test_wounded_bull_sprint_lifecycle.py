"""Integration tests for Wounded Bull Sprint TradeManager and Backfill lifecycles."""

import csv
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock

from app.const import ExitReason, Strategies
from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.services.backfill_engine import run_generic_backfill
from app.services.screener.strategies.wounded_bull_sprint import (
    WoundedBullSprintStrategy,
)
from app.services.trade_manager.manager import TradeManager
from app.services.trade_manager.strategies.wounded_bull_sprint import (
    WoundedBullSprintTradeStrategy,
)
from app.tools.market_holidays import MarketHolidayChecker
from app.tools.symbol_filter import SymbolFilter
from app.types import TradeStatus


def _seed_market_prices(
    stocks_db: Path,
    records: list[tuple[str, str, float, float, float, float, int]],
) -> None:
    """Initializes stocks.db schema and inserts daily OHLCV rows."""
    session = DatabaseSession(str(stocks_db))
    repo = MarketRepository(session)
    repo.init_schema()
    rows = [
        (sym, dt, o, h, low, c, vol, "tradingview", "1D")
        for sym, dt, o, h, low, c, vol in records
    ]
    repo.save_bulk_prices(rows)


def _init_signals_db(signals_db: Path) -> TradeRepository:
    """Initializes signals.db schema and returns TradeRepository."""
    session = DatabaseSession(str(signals_db))
    repo = TradeRepository(session)
    repo.init_schema()
    return repo


def test_wbs_thanksgiving_holiday_rerun_e2e(tmp_path: Path) -> None:
    """E2E regression: Holiday re-run on Wednesday candle keeps trade ACTIVE and exports CSV exit."""
    stocks_db = tmp_path / "stocks.db"
    signals_db = tmp_path / "signals.db"
    orders_dir = tmp_path / "orders"

    trade_repo = _init_signals_db(signals_db)
    # Seed Tuesday (setup) and Wednesday (entry + clexit trigger) candles
    _seed_market_prices(
        stocks_db,
        [
            ("UPS", "2026-11-24", 95.0, 97.0, 94.0, 96.0, 3_000_000),
            ("UPS", "2026-11-25", 96.0, 98.0, 95.5, 97.5, 3_000_000),
        ],
    )

    trade_id = trade_repo.create_trade(
        symbol="UPS",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=96.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-11-24", "setup_date": "2026-11-24", "setup_score": 4.2},
    )

    manager = TradeManager(db_path=signals_db, stocks_db_path=stocks_db)
    holiday_checker = MagicMock(spec=MarketHolidayChecker)
    holiday_checker.is_holiday.side_effect = lambda d: d == datetime.date(2026, 11, 26)
    manager.holiday_checker = holiday_checker
    manager.exchange_symbols._sp_100 = ["UPS"]

    # Thursday 06:30 (latest bar = Wed 2026-11-25): activates UPS on Wed Open, sets bars_held=1 & pending_exit=clexit
    manager.run_daily_process(reference_date="2026-11-26")
    trade_after_thu = trade_repo.get_trade(trade_id)
    assert trade_after_thu is not None
    assert trade_after_thu["status"] == TradeStatus.ACTIVE.value
    assert str(trade_after_thu["entry_date"])[:10] == "2026-11-25"
    ctx_thu = json.loads(str(trade_after_thu["signal_context"]))
    assert ctx_thu["bars_held"] == 1
    assert ctx_thu["pending_exit"] == "clexit"
    assert ctx_thu["last_processed_date"] == "2026-11-25"

    # Friday 06:30 after Thanksgiving (latest bar STILL Wed 2026-11-25): MUST stay ACTIVE!
    manager.run_daily_process(reference_date="2026-11-27")
    trade_after_fri_am = trade_repo.get_trade(trade_id)
    assert trade_after_fri_am is not None
    assert trade_after_fri_am["status"] == TradeStatus.ACTIVE.value

    # Friday 07:30 order generation: exports SELL MKT OPG for UPS
    from app.config import settings

    original_folders = dict(settings.app.database.folders)
    settings.app.database.folders["orders"] = str(orders_dir)
    try:
        csv_path = manager.generate_daily_orders(reference_date="2026-11-27")
        assert csv_path is not None
        with open(csv_path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["symbol"] == "UPS"
        assert rows[0]["action"] == "SELL"
        assert rows[0]["order_type"] == "MKT"
        assert rows[0]["tif"] == "OPG"
    finally:
        settings.app.database.folders = original_folders

    # Saturday 06:30 after Friday bar arrives: closes UPS at Friday real Open (100.5)
    _seed_market_prices(
        stocks_db,
        [("UPS", "2026-11-27", 100.5, 102.0, 99.5, 101.0, 3_000_000)],
    )
    manager.run_daily_process(reference_date="2026-11-28")
    trade_closed = trade_repo.get_trade(trade_id)
    assert trade_closed is not None
    assert trade_closed["status"] == TradeStatus.CLOSED.value
    assert str(trade_closed["exit_date"])[:10] == "2026-11-27"
    assert trade_closed["exit_price"] == 100.5
    assert trade_closed["exit_reason"] == ExitReason.TAKE_PROFIT.value
    # signal_context must not have been overwritten with post-close daily updates
    ctx_closed = json.loads(str(trade_closed["signal_context"]))
    assert ctx_closed["last_processed_date"] == "2026-11-25"


def test_wbs_ndx_leader_isolation_in_trade_manager(tmp_path: Path) -> None:
    """Verifies NDX Momentum leaders never suppress a WBS clexit."""
    stocks_db = tmp_path / "stocks.db"
    signals_db = tmp_path / "signals.db"
    trade_repo = _init_signals_db(signals_db)

    _seed_market_prices(
        stocks_db,
        [
            ("NVDA", "2026-07-20", 100.0, 102.0, 99.0, 100.0, 5_000_000),
            ("NVDA", "2026-07-21", 101.0, 106.0, 100.0, 105.0, 5_000_000),
            ("NVDA", "2026-07-22", 107.0, 109.0, 106.0, 108.0, 5_000_000),
        ],
    )

    # Create an NDX Momentum trade that marks NVDA as a leader
    trade_repo.create_trade(
        symbol="NVDA",
        strategy=Strategies.NDXMomentum,
        size=10.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-21", "leaders": ["NVDA"]},
    )

    # Active WBS trade in NVDA
    wbs_id = trade_repo.create_trade(
        symbol="NVDA",
        strategy=Strategies.WoundedBullSprint,
        size=50.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-17"},
    )
    trade_repo.update_trade(
        wbs_id,
        {
            "status": TradeStatus.ACTIVE,
            "entry_date": "2026-07-20",
            "entry_price": 100.0,
            "initial_size": 50.0,
            "current_size": 50.0,
        },
    )

    manager = TradeManager(db_path=signals_db, stocks_db_path=stocks_db)
    manager.exchange_symbols._sp_100 = ["NVDA"]
    manager.run_daily_process(reference_date="2026-07-23")

    closed_wbs = trade_repo.get_trade(wbs_id)
    assert closed_wbs is not None
    assert closed_wbs["status"] == TradeStatus.CLOSED.value
    assert closed_wbs["exit_price"] == 107.0


def test_wbs_score_prioritized_capacity_and_share_class_cap(tmp_path: Path) -> None:
    """Verifies 5-slot cap, setup_score priority, and GOOGL/GOOG share-class handling."""
    stocks_db = tmp_path / "stocks.db"
    signals_db = tmp_path / "signals.db"
    orders_dir = tmp_path / "orders"
    trade_repo = _init_signals_db(signals_db)

    symbols = ["GOOGL", "AAPL", "MSFT", "UPS", "GOOG", "HIGH", "LOW"]
    price_rows: list[tuple[str, str, float, float, float, float, int]] = []
    for sym in symbols:
        price_rows.append((sym, "2026-07-20", 100.0, 102.0, 99.0, 100.0, 3_000_000))
        price_rows.append((sym, "2026-07-21", 100.0, 101.0, 99.0, 100.0, 3_000_000))
    _seed_market_prices(stocks_db, price_rows)

    # 4 active WBS positions (including GOOGL -> counts as 1 slot, blocks GOOG)
    for active_sym in ["GOOGL", "AAPL", "MSFT", "UPS"]:
        tid = trade_repo.create_trade(
            symbol=active_sym,
            strategy=Strategies.WoundedBullSprint,
            size=50.0,
            entry=100.0,
            stop_loss=0.0,
            target=0.0,
            context={"date": "2026-07-17"},
        )
        trade_repo.update_trade(
            tid,
            {
                "status": TradeStatus.ACTIVE,
                "entry_date": "2026-07-20",
                "entry_price": 100.0,
                "initial_size": 50.0,
                "current_size": 50.0,
            },
        )

    # 3 CREATED candidates inserted out of score order
    low_id = trade_repo.create_trade(
        symbol="LOW",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-20", "setup_score": 3.0},
    )
    goog_id = trade_repo.create_trade(
        symbol="GOOG",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-20", "setup_score": 9.0},
    )
    high_id = trade_repo.create_trade(
        symbol="HIGH",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-20", "setup_score": 8.0},
    )

    symbol_filter = SymbolFilter()
    original_mapping = dict(symbol_filter._mapping)
    symbol_filter._mapping = {"GOOGL": ["GOOG"]}
    from app.config import settings

    original_folders = dict(settings.app.database.folders)
    settings.app.database.folders["orders"] = str(orders_dir)
    try:
        manager = TradeManager(db_path=signals_db, stocks_db_path=stocks_db)
        manager.exchange_symbols._sp_100 = symbols

        # Order generation must export ONLY HIGH (not GOOG duplicate, not LOW beyond 5 slots)
        csv_path = manager.generate_daily_orders(reference_date="2026-07-21")
        assert csv_path is not None
        with open(csv_path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        entry_symbols = [r["symbol"] for r in rows if r["action"] == "BUY"]
        assert entry_symbols == ["HIGH"]

        # Daily process must activate HIGH and invalidate GOOG + LOW
        manager.run_daily_process(reference_date="2026-07-22")
        high_trade = trade_repo.get_trade(high_id)
        goog_trade = trade_repo.get_trade(goog_id)
        low_trade = trade_repo.get_trade(low_id)
        assert high_trade is not None
        assert goog_trade is not None
        assert low_trade is not None
        assert high_trade["status"] == TradeStatus.ACTIVE.value
        assert goog_trade["status"] == TradeStatus.INVALID.value
        assert low_trade["status"] == TradeStatus.INVALID.value
    finally:
        symbol_filter._mapping = original_mapping
        settings.app.database.folders = original_folders


def test_wbs_backfill_active_exits_free_slot_before_created_entries(
    tmp_path: Path,
) -> None:
    """Verifies backfill processes ACTIVE exits before CREATED entries and tracks stats['filled']."""
    stocks_db = tmp_path / "stocks.db"
    signals_db = tmp_path / "signals.db"
    trade_repo = _init_signals_db(signals_db)

    # 5 active WBS trades where SYM0 exits via clexit on 2026-07-22 Open, freeing 1 slot for NEW1
    rows: list[tuple[str, str, float, float, float, float, int]] = []
    for sym in ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4", "NEW1", "NEW2"]:
        rows.append((sym, "2026-07-20", 100.0, 102.0, 99.0, 100.0, 3_000_000))
        close_day2 = 105.0 if sym == "SYM0" else 100.0
        rows.append((sym, "2026-07-21", 100.0, 106.0, 99.0, close_day2, 3_000_000))
        rows.append((sym, "2026-07-22", 106.0, 107.0, 105.0, 106.0, 3_000_000))
    _seed_market_prices(stocks_db, rows)

    for sym in ["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"]:
        tid = trade_repo.create_trade(
            symbol=sym,
            strategy=Strategies.WoundedBullSprint,
            size=50.0,
            entry=100.0,
            stop_loss=0.0,
            target=0.0,
            context={"date": "2026-07-17"},
        )
        trade_repo.update_trade(
            tid,
            {
                "status": TradeStatus.ACTIVE,
                "entry_date": "2026-07-20",
                "entry_price": 100.0,
                "initial_size": 50.0,
                "current_size": 50.0,
            },
        )

    new2_id = trade_repo.create_trade(
        symbol="NEW2",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-21", "setup_score": 2.5},
    )
    new1_id = trade_repo.create_trade(
        symbol="NEW1",
        strategy=Strategies.WoundedBullSprint,
        size=0.0,
        entry=100.0,
        stop_loss=0.0,
        target=0.0,
        context={"date": "2026-07-21", "setup_score": 6.0},
    )

    result = run_generic_backfill(
        stocks_session=DatabaseSession(str(stocks_db)),
        signals_session=DatabaseSession(str(signals_db)),
        strategy_name="wounded_bull_sprint",
        screener_class=WoundedBullSprintStrategy,
        strategy_engine_class=WoundedBullSprintTradeStrategy,
        lookback_days=400,
        start_date="2026-07-22",
        end_date="2026-07-22",
        budget=10000.0,
        clear_existing=False,
    )

    assert result["trades_filled"] == 1
    assert result["trades_closed"] == 1
    new1_trade = trade_repo.get_trade(new1_id)
    new2_trade = trade_repo.get_trade(new2_id)
    assert new1_trade is not None
    assert new2_trade is not None
    assert new1_trade["status"] == TradeStatus.ACTIVE.value
    assert new2_trade["status"] == TradeStatus.INVALID.value
