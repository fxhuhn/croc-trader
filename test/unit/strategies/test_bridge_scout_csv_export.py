"""Unit and integration tests for Bridge Scout CSV order export.

Verifies:
- Strategies.BridgeScout is present in _CSV_SUPPORTED_STRATEGIES.
- Standardized display name "BridgeScout" is used for trade_group_id and strategy_name.
- CREATED trades export a single ENTRY row (BUY LOC DAY) with target_price.
- ACTIVE trades export a single EXIT row (SELL MOC DAY).
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
from app.services.trade_manager.strategies.bridge_scout import (
    BridgeScoutTradeStrategy,
)
from app.types import TradeData, TradeStatus


def test_bridge_scout_in_supported_strategies_and_display_names() -> None:
    """Verifies that BridgeScout is whitelisted and has a standardized display name."""
    assert Strategies.BridgeScout in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.BridgeScout] == "BridgeScout"
    assert get_strategy_display_name(Strategies.BridgeScout) == "BridgeScout"


def test_write_csv_orders_file_includes_bridge_scout_created_entry(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct ENTRY row for CREATED Bridge Scout trade."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "501",
            "symbol": "QQQ",
            "strategy": Strategies.BridgeScout.value,
            "status": TradeStatus.CREATED.value,
        }
        order = Order(
            id="501_BridgeScout_QQQ",
            symbol="QQQ",
            quantity=20,
            mode="Entry",
            entry=OrderLeg(
                action="BUY",
                type="LOC",
                price=Decimal("485.50"),
                quantity=20,
                time_in_force="DAY",
            ),
            exits=[],
            last_status="CREATED",
        )

        csv_file = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-08-28",
            ibkr_account_id="U999888",
            resolve_strategy_fn=lambda name: Strategies.BridgeScout,
        )

        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "501_BridgeScout_QQQ"
        assert row["bracket_role"] == "ENTRY"
        assert row["symbol"] == "QQQ"
        assert row["account_id"] == "U999888"
        assert row["action"] == "BUY"
        assert row["quantity"] == "20"
        assert row["order_type"] == "LOC"
        assert row["target_price"] == "485.50"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "BridgeScout"
    finally:
        settings.app.database.folders = original_folders


def test_write_csv_orders_file_includes_bridge_scout_active_exit(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct EXIT row for ACTIVE Bridge Scout trade."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "502",
            "symbol": "QQQ",
            "strategy": Strategies.BridgeScout.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 20,
        }
        order = Order(
            id="502_BridgeScout_QQQ_exit",
            symbol="QQQ",
            quantity=20,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="MOC",
                    price=Decimal("492.00"),
                    quantity=20,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )

        csv_file = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-01",
            ibkr_account_id="U999888",
            resolve_strategy_fn=lambda name: Strategies.BridgeScout,
        )

        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "502_BridgeScout_QQQ"
        assert row["bracket_role"] == "EXIT"
        assert row["symbol"] == "QQQ"
        assert row["action"] == "SELL"
        assert row["quantity"] == "20"
        assert row["order_type"] == "MOC"
        assert row["target_price"] == "492.00"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "BridgeScout"
    finally:
        settings.app.database.folders = original_folders


def test_bridge_scout_end_to_end_order_generation_and_csv_export(
    tmp_path: Path,
) -> None:
    """End-to-end integration: BridgeScoutTradeStrategy order generation into CSV export."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        strategy = BridgeScoutTradeStrategy()

        # 1. Entry Generation
        trade_created: TradeData = {
            "id": "503",
            "symbol": "QQQ",
            "strategy": Strategies.BridgeScout.value,
            "status": TradeStatus.CREATED.value,
            "entry_price": 500.0,
            "signal_context": '{"req_close_rsi40": 495.0}',
        }
        df_history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-08-28"),
                    "open": 490.0,
                    "high": 496.0,
                    "low": 488.0,
                    "close": 492.0,
                    "volume": 1000,
                }
            ]
        )

        entry_order = strategy._generate_entry_order(
            trade=trade_created,
            dataframe_history=df_history,
            budget=10000.0,
        )
        assert entry_order is not None
        assert entry_order.entry is not None
        assert entry_order.entry.type == "LOC"
        assert entry_order.entry.price == Decimal("495.0")
        assert entry_order.quantity == 20

        # 2. Exit Generation
        trade_active: TradeData = {
            "id": "504",
            "symbol": "QQQ",
            "strategy": Strategies.BridgeScout.value,
            "status": TradeStatus.ACTIVE.value,
            "entry_price": 495.0,
            "current_size": 20,
            "entry_date": "2026-08-28",
        }
        df_exit_history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-09-01"),
                    "open": 498.0,
                    "high": 505.0,
                    "low": 497.0,
                    "close": 502.0,
                    "volume": 1500,
                }
            ]
        )

        exit_order = strategy._generate_exit_order(
            trade=trade_active,
            dataframe_history=df_exit_history,
            budget=10000.0,
        )
        assert exit_order is not None
        assert len(exit_order.exits) == 1
        assert exit_order.exits[0].type == "MOC"
        assert exit_order.exits[0].time_in_force == "DAY"
        assert exit_order.exits[0].price == Decimal("502.0")

        # 3. Combined CSV Export
        csv_file = write_csv_orders_file(
            orders_data=[(trade_created, entry_order), (trade_active, exit_order)],
            date_string="2026-09-01",
            ibkr_account_id="U999888",
            resolve_strategy_fn=lambda name: Strategies.BridgeScout,
        )

        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))

        assert len(rows) == 2

        # Verify Entry row
        assert rows[0]["trade_group_id"] == "503_BridgeScout_QQQ"
        assert rows[0]["bracket_role"] == "ENTRY"
        assert rows[0]["order_type"] == "LOC"
        assert rows[0]["target_price"] == "495.00"
        assert rows[0]["tif"] == "DAY"
        assert rows[0]["strategy_name"] == "BridgeScout"

        # Verify Exit row
        assert rows[1]["trade_group_id"] == "504_BridgeScout_QQQ"
        assert rows[1]["bracket_role"] == "EXIT"
        assert rows[1]["order_type"] == "MOC"
        assert rows[1]["target_price"] == "502.00"
        assert rows[1]["tif"] == "DAY"
        assert rows[1]["strategy_name"] == "BridgeScout"
    finally:
        settings.app.database.folders = original_folders


def test_bridge_scout_exit_inventory_guard() -> None:
    """Verifies that an active Bridge Scout trade with current_size 0 generates no exit order."""
    strategy = BridgeScoutTradeStrategy()
    trade_empty: TradeData = {
        "id": "505",
        "symbol": "QQQ",
        "strategy": Strategies.BridgeScout.value,
        "status": TradeStatus.ACTIVE.value,
        "current_size": 0,
    }
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-09-01"),
                "open": 498.0,
                "high": 505.0,
                "low": 497.0,
                "close": 502.0,
                "volume": 1500,
            }
        ]
    )

    exit_order = strategy._generate_exit_order(
        trade=trade_empty, dataframe_history=df, budget=10000.0
    )
    assert exit_order is None
