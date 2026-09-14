"""Unit and integration tests for NDX Momentum and Turnover Timing CSV order export.

Verifies:
- Strategies.NDXMomentum and Strategies.TurnOverTiming variants are present in _CSV_SUPPORTED_STRATEGIES.
- Standardized display names "NDXMomentum", "TurnoverTiming", "TurnoverTiming_1.0", "TurnoverTiming_0.5".
- NDX Momentum exports ENTRY and EXIT as MKT OPG (Market On Open).
- Turnover Timing exports ENTRY as LMT DAY and multi-green EXIT as MKT OPG (Next Open Rule).
- Output CSV strictly conforms to the 13-column schema.
"""

import csv
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.config import settings
from app.const import Strategies
from app.services.trade_manager.order_export import (
    _CSV_SUPPORTED_STRATEGIES,
    _STRATEGY_DISPLAY_NAMES,
    get_strategy_display_name,
    write_csv_orders_file,
)
from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
)
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy,
)
from app.types import TradeData, TradeStatus


def test_ndx_and_turnover_in_supported_strategies_and_display_names() -> None:
    """Verifies that NDXMomentum and all TurnoverTiming variants are whitelisted and named."""
    # NDX Momentum
    assert Strategies.NDXMomentum in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.NDXMomentum] == "NDXMomentum"
    assert get_strategy_display_name(Strategies.NDXMomentum) == "NDXMomentum"

    # Turnover Timing
    assert Strategies.TurnOverTiming in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.TurnOverTiming] == "TurnoverTiming"
    assert get_strategy_display_name(Strategies.TurnOverTiming) == "TurnoverTiming"

    assert Strategies.TurnOverTiming_10 in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.TurnOverTiming_10] == "TurnoverTiming_1.0"
    assert (
        get_strategy_display_name(Strategies.TurnOverTiming_10) == "TurnoverTiming_1.0"
    )

    assert Strategies.TurnOverTiming_05 in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.TurnOverTiming_05] == "TurnoverTiming_0.5"
    assert (
        get_strategy_display_name(Strategies.TurnOverTiming_05) == "TurnoverTiming_0.5"
    )


def test_ndx_momentum_e2e_csv_order_export(tmp_path: Path) -> None:
    """Verifies NDX Momentum entry and exit order generation into CSV export (both MKT OPG)."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        strategy = NDXMomentumTradeStrategy()

        # 1. Entry Order (MOO Entry: MKT OPG)
        trade_entry: TradeData = {
            "id": "601",
            "symbol": "NVDA",
            "strategy": Strategies.NDXMomentum.value,
            "status": TradeStatus.CREATED.value,
            "budget": 10000.0,
        }
        df_entry = pd.DataFrame([{"date": pd.Timestamp("2026-08-31"), "close": 500.0}])
        entry_order = strategy.generate_orders(
            trade=trade_entry,
            dataframe_history=df_entry,
            budget=10000.0,
        )
        assert entry_order is not None
        assert entry_order.entry is not None
        assert entry_order.entry.type == "MKT"
        assert entry_order.entry.time_in_force == "OPG"
        assert entry_order.quantity == 20

        # 2. Exit Order on Month Switch (MKT OPG)
        trade_exit: TradeData = {
            "id": "602",
            "symbol": "AMD",
            "strategy": Strategies.NDXMomentum.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 15,
            "entry_date": "2026-07-01",
        }
        df_exit = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-08-28"), "close": 150.0},
                {"date": pd.Timestamp("2026-08-31"), "close": 155.0},
            ]
        )
        exit_order = strategy.generate_orders(
            trade=trade_exit,
            dataframe_history=df_exit,
            budget=10000.0,
            created_symbols={"NVDA", "MSFT"},  # AMD dropped from leaders
            reference_date="2026-09-01",
        )
        assert exit_order is not None
        assert len(exit_order.exits) == 1
        assert exit_order.exits[0].type == "MKT"
        assert exit_order.exits[0].time_in_force == "OPG"
        assert exit_order.quantity == 15

        # 3. CSV Export
        csv_file = write_csv_orders_file(
            orders_data=[(trade_entry, entry_order), (trade_exit, exit_order)],
            date_string="2026-09-01",
            ibkr_account_id="U111222",
            resolve_strategy_fn=lambda name: Strategies.NDXMomentum,
        )
        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))

        assert len(rows) == 2

        # Entry row
        assert rows[0]["trade_group_id"] == "601_NDXMomentum_NVDA"
        assert rows[0]["bracket_role"] == "ENTRY"
        assert rows[0]["order_type"] == "MKT"
        assert rows[0]["tif"] == "OPG"
        assert rows[0]["strategy_name"] == "NDXMomentum"

        # Exit row
        assert rows[1]["trade_group_id"] == "602_NDXMomentum_AMD"
        assert rows[1]["bracket_role"] == "EXIT"
        assert rows[1]["order_type"] == "MKT"
        assert rows[1]["tif"] == "OPG"
        assert rows[1]["strategy_name"] == "NDXMomentum"
    finally:
        settings.app.database.folders = original_folders


def test_turnover_timing_e2e_csv_order_export(tmp_path: Path) -> None:
    """Verifies Turnover Timing entry (LMT DAY) and green sequence exit (MKT OPG) CSV export."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        strategy = TurnoverTimingStrategy()

        # 1. Entry Order (LMT DAY)
        trade_entry: TradeData = {
            "id": "701",
            "symbol": "SPY",
            "strategy": Strategies.TurnOverTiming.value,
            "status": TradeStatus.CREATED.value,
            "entry_price": 400.0,
            "budget": 3000.0,
        }
        df_entry = pd.DataFrame([{"date": pd.Timestamp("2026-09-01"), "close": 405.0}])
        entry_order = strategy.generate_orders(
            trade=trade_entry,
            dataframe_history=df_entry,
            budget=3000.0,
        )
        assert entry_order is not None
        assert entry_order.entry is not None
        assert entry_order.entry.type == "LMT"
        assert entry_order.entry.time_in_force == "DAY"
        assert entry_order.entry.price == Decimal("400.0")
        assert entry_order.quantity == 7  # 3000 / 400 = 7

        # 2. Exit Order after 2 Green Candles (Next Open Rule: MKT OPG)
        trade_exit: TradeData = {
            "id": "702",
            "symbol": "SPY",
            "strategy": Strategies.TurnOverTiming.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 7,
            "entry_price": 400.0,
            "signal_context": '{"setup_date": "2026-09-01", "setup_candle_green": false}',
        }
        df_exit = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-09-01"),
                    "open": 405.0,
                    "close": 401.0,  # red setup
                },
                {
                    "date": pd.Timestamp("2026-09-02"),
                    "open": 400.0,
                    "close": 404.0,  # Green 1
                },
                {
                    "date": pd.Timestamp("2026-09-03"),
                    "open": 404.0,
                    "close": 408.0,  # Green 2
                },
            ]
        )
        exit_order = strategy.generate_orders(
            trade=trade_exit,
            dataframe_history=df_exit,
            budget=3000.0,
        )
        assert exit_order is not None
        assert len(exit_order.exits) == 1
        assert exit_order.exits[0].type == "MKT"
        assert exit_order.exits[0].time_in_force == "OPG"
        assert exit_order.quantity == 7

        # 3. CSV Export
        csv_file = write_csv_orders_file(
            orders_data=[(trade_entry, entry_order), (trade_exit, exit_order)],
            date_string="2026-09-04",
            ibkr_account_id="U333444",
            resolve_strategy_fn=lambda name: Strategies.TurnOverTiming,
        )
        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))

        assert len(rows) == 2

        # Entry row: LMT DAY
        assert rows[0]["trade_group_id"] == "701_TurnoverTiming_SPY"
        assert rows[0]["bracket_role"] == "ENTRY"
        assert rows[0]["order_type"] == "LMT"
        assert rows[0]["target_price"] == "400.00"
        assert rows[0]["tif"] == "DAY"
        assert rows[0]["strategy_name"] == "TurnoverTiming"

        # Exit row: MKT OPG
        assert rows[1]["trade_group_id"] == "702_TurnoverTiming_SPY"
        assert rows[1]["bracket_role"] == "EXIT"
        assert rows[1]["order_type"] == "MKT"
        assert rows[1]["tif"] == "OPG"
        assert rows[1]["strategy_name"] == "TurnoverTiming"
    finally:
        settings.app.database.folders = original_folders
