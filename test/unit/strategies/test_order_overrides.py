"""Unit tests for order generation configuration overrides in order export."""

import csv
from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.const import Strategies
from app.models import Order, OrderLeg
from app.services.trade_manager.order_export import (
    map_order_to_csv_rows,
    write_csv_orders_file,
)


@pytest.fixture
def restore_overrides() -> Any:
    """Fixture to ensure settings.app.order_overrides is restored after each test."""
    original_overrides = getattr(settings.app, "order_overrides", {})
    yield
    settings.app.order_overrides = original_overrides


def test_map_order_to_csv_rows_no_overrides(restore_overrides: Any) -> None:
    """Verifies that map_order_to_csv_rows uses default values when no overrides match."""
    # Arrange
    settings.app.order_overrides = {}
    order = Order(
        id="test_order_1",
        symbol="AAPL",
        quantity=10,
        mode="BRACKET",
        entry=OrderLeg(
            action="BUY",
            type="LMT",
            price=150.0,
            quantity=10,
            time_in_force="DAY",
        ),
        exits=[],
        last_status="CREATED",
    )
    trade = {"id": 1, "symbol": "AAPL", "strategy": "two_percent"}

    # Act
    rows = map_order_to_csv_rows(
        trade=trade,
        order=order,
        trade_group_id="1_TwoPercent_AAPL",
        strategy_display_name="TwoPercent",
        ibkr_account_id="U12345",
    )

    # Assert
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["sec_type"] == "STK"
    assert rows[0]["exchange"] == "SMART"
    assert rows[0]["currency"] == ""


def test_map_order_to_csv_rows_with_overrides(restore_overrides: Any) -> None:
    """Verifies that map_order_to_csv_rows applies custom symbol-level overrides."""
    # Arrange
    settings.app.order_overrides = {
        "SXRV.DE": {
            "target_symbol": "SXRV",
            "exchange": "IBIS",
            "currency": "EUR",
        }
    }
    order = Order(
        id="test_order_2",
        symbol="SXRV.DE",
        quantity=5,
        mode="BRACKET",
        entry=OrderLeg(
            action="BUY",
            type="LMT",
            price=1400.0,
            quantity=5,
            time_in_force="DAY",
        ),
        exits=[],
        last_status="CREATED",
    )
    trade = {"id": 2, "symbol": "SXRV.DE", "strategy": "two_percent"}

    # Act
    rows = map_order_to_csv_rows(
        trade=trade,
        order=order,
        trade_group_id="2_TwoPercent_SXRV",
        strategy_display_name="TwoPercent",
        ibkr_account_id="U12345",
    )

    # Assert
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SXRV"
    assert rows[0]["sec_type"] == "STK"
    assert rows[0]["exchange"] == "IBIS"
    assert rows[0]["currency"] == "EUR"


def test_write_csv_orders_file_applies_overrides(
    tmp_path: Path, restore_overrides: Any
) -> None:
    """Verifies that write_csv_orders_file applies overrides and outputs correct CSV structure."""
    # Arrange
    settings.app.order_overrides = {
        "SXRV.DE": {
            "target_symbol": "SXRV",
            "exchange": "IBIS",
            "currency": "EUR",
        }
    }

    order = Order(
        id="test_order_3",
        symbol="SXRV.DE",
        quantity=1,
        mode="BRACKET",
        entry=OrderLeg(
            action="BUY",
            type="LMT",
            price=1450.0,
            quantity=1,
            time_in_force="DAY",
        ),
        exits=[],
        last_status="CREATED",
    )
    trade = {"id": 3, "symbol": "SXRV.DE", "strategy": "two_percent"}
    orders_data = [(trade, order)]

    # Mock output directory to tmp_path/orders using path mock or patch settings.app.database.folders
    original_folders = settings.app.database.folders.copy()
    settings.app.database.folders["orders"] = str(tmp_path / "orders")

    # Setup a dummy resolver function
    def resolve_strategy_fn(strategy_name: str) -> Strategies:
        return Strategies.TwoPercent

    try:
        # Act
        csv_file_path = write_csv_orders_file(
            orders_data=orders_data,
            date_string="2026-06-30",
            ibkr_account_id="U12345",
            resolve_strategy_fn=resolve_strategy_fn,
        )

        # Assert
        assert csv_file_path is not None
        assert csv_file_path.exists()

        # Read and parse CSV file
        with open(csv_file_path, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        # Trade group ID should use overridden symbol
        assert row["trade_group_id"] == "3_TwoPercent_SXRV"
        assert row["symbol"] == "SXRV"
        assert row["exchange"] == "IBIS"
        assert row["currency"] == "EUR"
        assert row["sec_type"] == "STK"
    finally:
        # Restore folders
        settings.app.database.folders = original_folders
