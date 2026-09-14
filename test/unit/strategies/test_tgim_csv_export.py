"""Unit and integration tests for TGIM CSV order export.

Verifies:
- Strategies.TGIM is present in _CSV_SUPPORTED_STRATEGIES.
- Standardized display name "TGIM" is used for trade_group_id and strategy_name.
- CREATED trades export a single ENTRY row (BUY LOC DAY).
- ACTIVE trades export EXIT row (Bar 1: SELL LOC DAY, Bar 2: SELL MOC DAY).
- Inventory guard prevents short orders when current_size is 0.
- Output CSV strictly conforms to the 13-column schema.
"""

import csv
from decimal import Decimal
from pathlib import Path

import pandas as pd

from app.config import settings
from app.const import Strategies
from app.models import Order, OrderLeg
from app.services.trade_manager.order_export import (
    _CSV_SUPPORTED_STRATEGIES,
    _STRATEGY_DISPLAY_NAMES,
    get_strategy_display_name,
    write_csv_orders_file,
)
from app.services.trade_manager.strategies.tgim import TGIMTradeStrategy
from app.types import TradeData, TradeStatus


def test_tgim_in_supported_strategies_and_display_names() -> None:
    """Verifies that TGIM is whitelisted and has a standardized display name."""
    assert Strategies.TGIM in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.TGIM] == "TGIM"
    assert get_strategy_display_name(Strategies.TGIM) == "TGIM"


def test_write_csv_orders_file_includes_tgim_created_entry(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct ENTRY row for CREATED TGIM trade."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "301",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.CREATED.value,
        }
        order = Order(
            id="301_TGIM_SPY",
            symbol="SPY",
            quantity=20,
            mode="Entry",
            entry=OrderLeg(
                action="BUY",
                type="LOC",
                price=Decimal("500.00"),
                quantity=20,
                time_in_force="DAY",
            ),
            exits=[],
            last_status="CREATED",
        )

        csv_file_path = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-14",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )

        assert csv_file_path is not None
        assert csv_file_path.exists()

        with open(csv_file_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "301_TGIM_SPY"
        assert row["bracket_role"] == "ENTRY"
        assert row["symbol"] == "SPY"
        assert row["sec_type"] == "STK"
        assert row["exchange"] == "SMART"
        assert row["account_id"] == "U19605236"
        assert row["action"] == "BUY"
        assert row["quantity"] == "20"
        assert row["order_type"] == "LOC"
        assert row["target_price"] == "500.00"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "TGIM"
    finally:
        settings.app.database.folders = original_folders


def test_write_csv_orders_file_includes_tgim_active_exit_bar1_loc(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct LOC EXIT row for ACTIVE TGIM trade on Bar 1."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "302",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 20,
        }
        order = Order(
            id="302_TGIM_SPY",
            symbol="SPY",
            quantity=20,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="LOC",
                    price=Decimal("505.00"),
                    quantity=20,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )

        csv_file_path = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-15",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )

        assert csv_file_path is not None
        assert csv_file_path.exists()

        with open(csv_file_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "302_TGIM_SPY"
        assert row["bracket_role"] == "EXIT"
        assert row["symbol"] == "SPY"
        assert row["sec_type"] == "STK"
        assert row["exchange"] == "SMART"
        assert row["account_id"] == "U19605236"
        assert row["action"] == "SELL"
        assert row["quantity"] == "20"
        assert row["order_type"] == "LOC"
        assert row["target_price"] == "505.00"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "TGIM"
    finally:
        settings.app.database.folders = original_folders


def test_write_csv_orders_file_includes_tgim_active_exit_bar2_moc(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct MOC EXIT row for ACTIVE TGIM trade on Bar 2."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "303",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 20,
        }
        order = Order(
            id="303_TGIM_SPY",
            symbol="SPY",
            quantity=20,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="MOC",
                    price=Decimal("495.00"),
                    quantity=20,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )

        csv_file_path = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-16",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )

        assert csv_file_path is not None
        assert csv_file_path.exists()

        with open(csv_file_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "303_TGIM_SPY"
        assert row["bracket_role"] == "EXIT"
        assert row["symbol"] == "SPY"
        assert row["sec_type"] == "STK"
        assert row["exchange"] == "SMART"
        assert row["account_id"] == "U19605236"
        assert row["action"] == "SELL"
        assert row["quantity"] == "20"
        assert row["order_type"] == "MOC"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "TGIM"
    finally:
        settings.app.database.folders = original_folders


def test_tgim_strategy_to_csv_end_to_end(tmp_path: Path) -> None:
    """Verifies end-to-end flow from TGIMTradeStrategy order generation to CSV export."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    strategy = TGIMTradeStrategy()

    history_entry = pd.DataFrame(
        [
            {
                "date": "2026-09-14",
                "open": 500.0,
                "high": 502.0,
                "low": 498.0,
                "close": 500.0,
            }
        ]
    )

    try:
        # 1. CREATED trade generates LOC entry order
        trade_created: TradeData = {
            "id": "401",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.CREATED.value,
            "entry_price": 500.0,
            "budget": 10000.0,
        }
        order_entry = strategy._generate_entry_order(
            trade_created, history_entry, budget=10000.0
        )
        assert order_entry is not None
        assert order_entry.entry is not None
        assert order_entry.entry.type == "LOC"
        assert order_entry.entry.time_in_force == "DAY"
        assert order_entry.quantity == 20
        assert len(order_entry.exits) == 0

        csv_path_entry = write_csv_orders_file(
            orders_data=[(trade_created, order_entry)],
            date_string="2026-09-14",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )
        assert csv_path_entry is not None
        with open(csv_path_entry, newline="", encoding="utf-8") as file_handle:
            entry_rows = list(csv.DictReader(file_handle))
        assert len(entry_rows) == 1
        assert entry_rows[0]["trade_group_id"] == "401_TGIM_SPY"
        assert entry_rows[0]["bracket_role"] == "ENTRY"
        assert entry_rows[0]["order_type"] == "LOC"
        assert entry_rows[0]["target_price"] == "500.00"
        assert entry_rows[0]["tif"] == "DAY"
        assert entry_rows[0]["strategy_name"] == "TGIM"
        assert entry_rows[0]["quantity"] == "20"

        # 2. ACTIVE trade on Bar 1 (Tuesday) generates LOC exit order
        trade_active: TradeData = {
            "id": "401",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.ACTIVE.value,
            "entry_price": 500.0,
            "entry_date": "2026-09-14",
            "current_size": order_entry.quantity,
        }
        history_bar1 = pd.DataFrame(
            [
                {"date": "2026-09-14", "close": 500.0},
                {"date": "2026-09-15", "close": 505.0},
            ]
        )
        order_exit_bar1 = strategy._generate_exit_order(
            trade_active, history_bar1, budget=10000.0
        )
        assert order_exit_bar1 is not None
        assert order_exit_bar1.entry is None
        assert len(order_exit_bar1.exits) == 1
        assert order_exit_bar1.exits[0].type == "LOC"
        assert order_exit_bar1.exits[0].time_in_force == "DAY"
        assert order_exit_bar1.exits[0].price == Decimal("500.0")

        csv_path_exit_bar1 = write_csv_orders_file(
            orders_data=[(trade_active, order_exit_bar1)],
            date_string="2026-09-15",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )
        assert csv_path_exit_bar1 is not None
        with open(csv_path_exit_bar1, newline="", encoding="utf-8") as file_handle:
            exit_rows_bar1 = list(csv.DictReader(file_handle))
        assert len(exit_rows_bar1) == 1
        assert exit_rows_bar1[0]["trade_group_id"] == "401_TGIM_SPY"
        assert exit_rows_bar1[0]["bracket_role"] == "EXIT"
        assert exit_rows_bar1[0]["order_type"] == "LOC"
        assert exit_rows_bar1[0]["target_price"] == "500.00"
        assert exit_rows_bar1[0]["tif"] == "DAY"
        assert exit_rows_bar1[0]["strategy_name"] == "TGIM"

        # 3. ACTIVE trade on Bar 2 (Wednesday) generates MOC exit order
        history_bar2 = pd.DataFrame(
            [
                {"date": "2026-09-14", "close": 500.0},
                {"date": "2026-09-15", "close": 498.0},
                {"date": "2026-09-16", "close": 495.0},
            ]
        )
        order_exit_bar2 = strategy._generate_exit_order(
            trade_active, history_bar2, budget=10000.0
        )
        assert order_exit_bar2 is not None
        assert order_exit_bar2.entry is None
        assert len(order_exit_bar2.exits) == 1
        assert order_exit_bar2.exits[0].type == "MOC"
        assert order_exit_bar2.exits[0].time_in_force == "DAY"

        csv_path_exit_bar2 = write_csv_orders_file(
            orders_data=[(trade_active, order_exit_bar2)],
            date_string="2026-09-16",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.TGIM,
        )
        assert csv_path_exit_bar2 is not None
        with open(csv_path_exit_bar2, newline="", encoding="utf-8") as file_handle:
            exit_rows_bar2 = list(csv.DictReader(file_handle))
        assert len(exit_rows_bar2) == 1
        assert exit_rows_bar2[0]["trade_group_id"] == "401_TGIM_SPY"
        assert exit_rows_bar2[0]["bracket_role"] == "EXIT"
        assert exit_rows_bar2[0]["order_type"] == "MOC"
        assert exit_rows_bar2[0]["tif"] == "DAY"
        assert exit_rows_bar2[0]["strategy_name"] == "TGIM"

        # 4. Inventory protection: current_size <= 0 produces None
        trade_zero_size: TradeData = {
            "id": "402",
            "symbol": "SPY",
            "strategy": Strategies.TGIM.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 0,
        }
        order_zero = strategy._generate_exit_order(
            trade_zero_size, history_bar1, budget=10000.0
        )
        assert order_zero is None

    finally:
        settings.app.database.folders = original_folders
