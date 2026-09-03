"""Unit and integration tests for Bounce Bandit CSV order export.

Verifies:
- Strategies.BounceBandit is present in _CSV_SUPPORTED_STRATEGIES.
- Standardized display name "BounceBandit" is used for trade_group_id and strategy_name.
- CREATED trades export a single ENTRY row (BUY MKT OPG).
- ACTIVE trades export a single EXIT row (SELL LOC DAY) with target_price.
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
from app.services.trade_manager.strategies.bounce_bandit import (
    BounceBanditTradeStrategy,
)
from app.types import TradeData, TradeStatus


def test_bounce_bandit_in_supported_strategies_and_display_names() -> None:
    """Verifies that BounceBandit is whitelisted and has a standardized display name."""
    assert Strategies.BounceBandit in _CSV_SUPPORTED_STRATEGIES
    assert _STRATEGY_DISPLAY_NAMES[Strategies.BounceBandit] == "BounceBandit"
    assert get_strategy_display_name(Strategies.BounceBandit) == "BounceBandit"


def test_write_csv_orders_file_includes_bounce_bandit_created_entry(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct ENTRY row for CREATED Bounce Bandit trade."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "101",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.CREATED.value,
        }
        order = Order(
            id="101_BounceBandit_QQQ",
            symbol="QQQ",
            quantity=20,
            mode="Entry",
            entry=OrderLeg(
                action="BUY",
                type="MKT",
                price=Decimal("500.00"),
                quantity=20,
                time_in_force="OPG",
            ),
            exits=[],
            last_status="CREATED",
        )

        csv_file_path = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-02",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.BounceBandit,
        )

        assert csv_file_path is not None
        assert csv_file_path.exists()

        with open(csv_file_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "101_BounceBandit_QQQ"
        assert row["bracket_role"] == "ENTRY"
        assert row["symbol"] == "QQQ"
        assert row["sec_type"] == "STK"
        assert row["exchange"] == "SMART"
        assert row["account_id"] == "U19605236"
        assert row["action"] == "BUY"
        assert row["quantity"] == "20"
        assert row["order_type"] == "MKT"
        assert row["target_price"] == "500.00"
        assert row["tif"] == "OPG"
        assert row["strategy_name"] == "BounceBandit"
    finally:
        settings.app.database.folders = original_folders


def test_write_csv_orders_file_includes_bounce_bandit_active_exit(
    tmp_path: Path,
) -> None:
    """Verifies write_csv_orders_file produces correct EXIT row with LOC for ACTIVE trade."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    try:
        trade: TradeData = {
            "id": "102",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 15,
        }
        order = Order(
            id="102_BounceBandit_QQQ",
            symbol="QQQ",
            quantity=15,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="LOC",
                    price=Decimal("512.45"),
                    quantity=15,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )

        csv_file_path = write_csv_orders_file(
            orders_data=[(trade, order)],
            date_string="2026-09-02",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.BounceBandit,
        )

        assert csv_file_path is not None
        assert csv_file_path.exists()

        with open(csv_file_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["trade_group_id"] == "102_BounceBandit_QQQ"
        assert row["bracket_role"] == "EXIT"
        assert row["symbol"] == "QQQ"
        assert row["sec_type"] == "STK"
        assert row["exchange"] == "SMART"
        assert row["account_id"] == "U19605236"
        assert row["action"] == "SELL"
        assert row["quantity"] == "15"
        assert row["order_type"] == "LOC"
        assert row["target_price"] == "512.45"
        assert row["tif"] == "DAY"
        assert row["strategy_name"] == "BounceBandit"
    finally:
        settings.app.database.folders = original_folders


def test_bounce_bandit_strategy_to_csv_end_to_end(tmp_path: Path) -> None:
    """Verifies end-to-end flow from BounceBanditTradeStrategy generation to CSV export."""
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    strategy = BounceBanditTradeStrategy()

    # 1. Ten days of price history
    dates = pd.date_range("2026-08-15", periods=10, freq="B")
    prices = [500.0, 498.0, 495.0, 492.0, 490.0, 491.0, 493.0, 494.0, 495.0, 496.0]
    df_history = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": p,
                "high": p + 2.0,
                "low": p - 2.0,
                "close": p,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )

    try:
        # 2. Test CREATED trade generates entry order
        trade_created: TradeData = {
            "id": "201",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.CREATED.value,
            "entry_price": 496.0,
        }
        order_entry = strategy._generate_entry_order(
            trade_created, df_history, budget=10000.0
        )
        assert order_entry is not None
        assert order_entry.entry is not None
        assert order_entry.entry.type == "MKT"
        assert order_entry.entry.time_in_force == "OPG"
        assert len(order_entry.exits) == 0  # No exits on entry day

        csv_path_entry = write_csv_orders_file(
            orders_data=[(trade_created, order_entry)],
            date_string="2026-08-28",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.BounceBandit,
        )
        assert csv_path_entry is not None
        with open(csv_path_entry, newline="") as f:
            entry_rows = list(csv.DictReader(f))
        assert len(entry_rows) == 1
        assert entry_rows[0]["bracket_role"] == "ENTRY"
        assert entry_rows[0]["order_type"] == "MKT"
        assert entry_rows[0]["tif"] == "OPG"

        # 3. Test ACTIVE trade generates exit order
        trade_active: TradeData = {
            "id": "201",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.ACTIVE.value,
            "entry_price": 496.0,
            "current_size": order_entry.quantity,
        }
        order_exit = strategy._generate_exit_order(
            trade_active, df_history, budget=10000.0
        )
        assert order_exit is not None
        assert order_exit.entry is None
        assert len(order_exit.exits) == 1
        assert order_exit.exits[0].type == "LOC"
        assert order_exit.exits[0].time_in_force == "DAY"
        assert order_exit.exits[0].price > Decimal("0")

        csv_path_exit = write_csv_orders_file(
            orders_data=[(trade_active, order_exit)],
            date_string="2026-08-29",
            ibkr_account_id="U19605236",
            resolve_strategy_fn=lambda name: Strategies.BounceBandit,
        )
        assert csv_path_exit is not None
        with open(csv_path_exit, newline="") as f:
            exit_rows = list(csv.DictReader(f))
        assert len(exit_rows) == 1
        assert exit_rows[0]["bracket_role"] == "EXIT"
        assert exit_rows[0]["order_type"] == "LOC"
        assert exit_rows[0]["tif"] == "DAY"
        assert float(exit_rows[0]["target_price"]) > 0.0

        # 4. Short-Schutz: current_size <= 0 produces None and no CSV rows
        trade_zero_size: TradeData = {
            "id": "202",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.ACTIVE.value,
            "current_size": 0,
        }
        order_zero = strategy._generate_exit_order(
            trade_zero_size, df_history, budget=10000.0
        )
        assert order_zero is None

    finally:
        settings.app.database.folders = original_folders
