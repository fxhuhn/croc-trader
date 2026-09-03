"""Characterization (pinning) tests for order_export.py.

These tests secure the exact baseline behavior before refactoring, covering:
1. Multi-leg bracket orders with Entry, SL (STP) and TP (LMT/LOC).
2. Quantity fallback when OrderLeg.quantity is None.
3. Whitelist filtering (unsupported strategies like HoldTarget return None).
4. Mixed batches (supported strategies written, unsupported omitted).
5. Display name fallback for non-mapped strategies.
6. Empty override fallback.
7. Golden-Master-Snapshot: Character-by-character verification of the generated CSV.
"""

import csv
from decimal import Decimal
from pathlib import Path

from app.config import settings
from app.const import Strategies
from app.models import Order, OrderLeg
from app.services.trade_manager.order_export import (
    _get_override_for_symbol,
    get_strategy_display_name,
    map_order_to_csv_rows,
    write_csv_orders_file,
)
from app.types import TradeData, TradeStatus


def test_characterization_bracket_order_with_sl_and_tp() -> None:
    """Verifies that 3-leg bracket orders create ENTRY, SL, and TP rows."""
    order = Order(
        id="order_bracket_1",
        symbol="QQQ",
        quantity=10,
        mode="BRACKET",
        entry=OrderLeg(
            action="BUY",
            type="LMT",
            price=Decimal("500.00"),
            quantity=10,
            time_in_force="DAY",
        ),
        exits=[
            OrderLeg(
                action="SELL",
                type="STP",
                price=Decimal("490.00"),
                quantity=10,
                time_in_force="DAY",
            ),
            OrderLeg(
                action="SELL",
                type="LMT",
                price=Decimal("520.00"),
                quantity=10,
                time_in_force="DAY",
            ),
        ],
        last_status="CREATED",
    )
    trade: TradeData = {
        "id": "1",
        "symbol": "QQQ",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
    }

    rows = map_order_to_csv_rows(
        trade=trade,
        order=order,
        trade_group_id="1_TwoPercent_QQQ",
        strategy_display_name="TwoPercent",
        ibkr_account_id="U123456",
    )

    assert len(rows) == 3
    # 1. Entry Leg
    assert rows[0]["bracket_role"] == "ENTRY"
    assert rows[0]["action"] == "BUY"
    assert rows[0]["order_type"] == "LMT"
    assert rows[0]["target_price"] == "500.00"

    # 2. Stop Loss Leg (type == STP)
    assert rows[1]["bracket_role"] == "SL"
    assert rows[1]["action"] == "SELL"
    assert rows[1]["order_type"] == "STP"
    assert rows[1]["target_price"] == "490.00"

    # 3. Take Profit Leg (type != STP)
    assert rows[2]["bracket_role"] == "TP"
    assert rows[2]["action"] == "SELL"
    assert rows[2]["order_type"] == "LMT"
    assert rows[2]["target_price"] == "520.00"


def test_characterization_quantity_fallback_when_leg_quantity_is_none() -> None:
    """Verifies that OrderLeg.quantity=None falls back to Order.quantity."""
    order = Order(
        id="order_fallback_qty",
        symbol="QQQ",
        quantity=35,
        mode="BRACKET",
        entry=OrderLeg(
            action="BUY",
            type="MKT",
            price=Decimal("500.00"),
            quantity=None,  # Fallback to 35
            time_in_force="OPG",
        ),
        exits=[
            OrderLeg(
                action="SELL",
                type="LOC",
                price=Decimal("510.00"),
                quantity=None,  # Fallback to 35
                time_in_force="DAY",
            )
        ],
        last_status="CREATED",
    )
    trade: TradeData = {
        "id": "2",
        "symbol": "QQQ",
        "strategy": Strategies.BounceBandit.value,
        "status": TradeStatus.CREATED.value,
    }

    rows = map_order_to_csv_rows(
        trade=trade,
        order=order,
        trade_group_id="2_BounceBandit_QQQ",
        strategy_display_name="BounceBandit",
        ibkr_account_id="U123456",
    )

    assert len(rows) == 2
    assert rows[0]["quantity"] == 35
    assert rows[1]["quantity"] == 35


def test_characterization_unsupported_strategies_return_none() -> None:
    """Verifies that orders from unsupported strategies (HoldTarget, SplitTarget) return None."""
    order = Order(
        id="order_hold_1",
        symbol="SPY",
        quantity=10,
        mode="Exit",
        entry=None,
        exits=[
            OrderLeg(
                action="SELL",
                type="MKT",
                price=Decimal("550.00"),
                quantity=10,
                time_in_force="DAY",
            )
        ],
        last_status="ACTIVE",
    )
    trade_hold: TradeData = {
        "id": "3",
        "symbol": "SPY",
        "strategy": Strategies.HoldTarget.value,
        "status": TradeStatus.ACTIVE.value,
    }

    result = write_csv_orders_file(
        orders_data=[(trade_hold, order)],
        date_string="2026-09-03",
        ibkr_account_id="U123456",
        resolve_strategy_fn=lambda name: Strategies.HoldTarget,
    )

    assert result is None


def test_characterization_mixed_batch_filters_unsupported_strategies(
    tmp_path: Path,
) -> None:
    """Verifies that a mixed batch exports supported strategies and drops unsupported ones."""
    original_overrides = getattr(settings.app, "order_overrides", {})
    try:
        settings.app.order_overrides = {}

        order_supported = Order(
            id="order_twopercent",
            symbol="AAPL",
            quantity=10,
            mode="Entry",
            entry=OrderLeg(
                action="BUY",
                type="LMT",
                price=Decimal("220.00"),
                quantity=10,
                time_in_force="DAY",
            ),
            exits=[],
            last_status="CREATED",
        )
        trade_supported: TradeData = {
            "id": "10",
            "symbol": "AAPL",
            "strategy": Strategies.TwoPercent.value,
            "status": TradeStatus.CREATED.value,
        }

        order_unsupported = Order(
            id="order_split",
            symbol="MSFT",
            quantity=5,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="MKT",
                    price=Decimal("410.00"),
                    quantity=5,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )
        trade_unsupported: TradeData = {
            "id": "11",
            "symbol": "MSFT",
            "strategy": Strategies.SplitTarget.value,
            "status": TradeStatus.ACTIVE.value,
        }

        def resolve_fn(strat_name: str) -> Strategies | None:
            if strat_name == Strategies.TwoPercent.value:
                return Strategies.TwoPercent
            if strat_name == Strategies.SplitTarget.value:
                return Strategies.SplitTarget
            return None

        csv_file = write_csv_orders_file(
            orders_data=[
                (trade_supported, order_supported),
                (trade_unsupported, order_unsupported),
            ],
            date_string="2026-09-03",
            ibkr_account_id="U123456",
            resolve_strategy_fn=resolve_fn,
        )

        assert csv_file is not None
        assert csv_file.exists()

        with open(csv_file, newline="") as f:
            rows = list(csv.DictReader(f))

        # Only 1 order (TwoPercent AAPL) should be exported
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAPL"
        assert rows[0]["strategy_name"] == "TwoPercent"
    finally:
        settings.app.order_overrides = original_overrides


def test_characterization_empty_orders_returns_none() -> None:
    """Verifies that empty orders_data list returns None."""
    result = write_csv_orders_file(
        orders_data=[],
        date_string="2026-09-03",
        ibkr_account_id="U123456",
        resolve_strategy_fn=lambda name: None,
    )
    assert result is None


def test_characterization_get_override_invalid_structure() -> None:
    """Verifies _get_override_for_symbol handles malformed order_overrides gracefully."""
    original_overrides = getattr(settings.app, "order_overrides", {})
    try:
        # Non-dict top level
        settings.app.order_overrides = "invalid_string"  # type: ignore[assignment]
        assert _get_override_for_symbol("AAPL") == {}

        # Non-dict symbol value
        settings.app.order_overrides = {"AAPL": "not_a_dict"}  # type: ignore[dict-item]
        assert _get_override_for_symbol("AAPL") == {}
    finally:
        settings.app.order_overrides = original_overrides


def test_characterization_display_name_fallback() -> None:
    """Verifies get_strategy_display_name falls back to enum.value if not in mapping."""
    assert get_strategy_display_name(Strategies.TGIM) == "tgim"
    assert get_strategy_display_name(Strategies.BridgeScout) == "bridge_scout"


def test_characterization_order_with_no_legs_returns_none() -> None:
    """Verifies that an order with no legs produces no rows and returns None."""
    order = Order(
        id="order_empty_legs",
        symbol="QQQ",
        quantity=10,
        mode="Exit",
        entry=None,
        exits=[],
        last_status="ACTIVE",
    )
    trade: TradeData = {
        "id": "99",
        "symbol": "QQQ",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.ACTIVE.value,
    }

    result = write_csv_orders_file(
        orders_data=[(trade, order)],
        date_string="2026-09-03",
        ibkr_account_id="U123456",
        resolve_strategy_fn=lambda name: Strategies.TwoPercent,
    )
    assert result is None


def test_write_csv_orders_file_custom_output_directory(tmp_path: Path) -> None:
    """Verifies that write_csv_orders_file respects an explicitly passed output_directory."""
    custom_dir = tmp_path / "custom_orders"
    order = Order(
        id="order_custom_dir",
        symbol="QQQ",
        quantity=10,
        mode="Entry",
        entry=OrderLeg(
            action="BUY",
            type="MKT",
            price=Decimal("500.00"),
            quantity=10,
            time_in_force="OPG",
        ),
        exits=[],
        last_status="CREATED",
    )
    trade: TradeData = {
        "id": "100",
        "symbol": "QQQ",
        "strategy": Strategies.BounceBandit.value,
        "status": TradeStatus.CREATED.value,
    }

    csv_file = write_csv_orders_file(
        orders_data=[(trade, order)],
        date_string="2026-09-03",
        ibkr_account_id="U123456",
        resolve_strategy_fn=lambda name: Strategies.BounceBandit,
        output_directory=custom_dir,
    )

    assert csv_file is not None
    assert csv_file.exists()
    assert csv_file.parent == custom_dir


def test_characterization_golden_master_snapshot() -> None:
    """Golden Master Test: Verifies character-by-character output equality for a canonical multi-strategy batch."""
    original_overrides = getattr(settings.app, "order_overrides", {})
    settings.app.order_overrides = {
        "SXRV.DE": {
            "target_symbol": "SXRV",
            "exchange": "IBIS",
            "currency": "EUR",
        }
    }

    try:
        # 1. TwoPercent order with symbol override and multi-leg (ENTRY + SL + TP)
        order_1 = Order(
            id="order_1",
            symbol="SXRV.DE",
            quantity=10,
            mode="BRACKET",
            entry=OrderLeg(
                action="BUY",
                type="LMT",
                price=Decimal("1450.50"),
                quantity=10,
                time_in_force="DAY",
            ),
            exits=[
                OrderLeg(
                    action="SELL",
                    type="STP",
                    price=Decimal("1400.00"),
                    quantity=10,
                    time_in_force="DAY",
                ),
                OrderLeg(
                    action="SELL",
                    type="LMT",
                    price=Decimal("1500.00"),
                    quantity=10,
                    time_in_force="DAY",
                ),
            ],
            last_status="CREATED",
        )
        trade_1: TradeData = {
            "id": "501",
            "symbol": "SXRV.DE",
            "strategy": Strategies.TwoPercent.value,
            "status": TradeStatus.CREATED.value,
        }

        # 2. BounceBandit order: single EXIT leg (LOC)
        order_2 = Order(
            id="order_2",
            symbol="QQQ",
            quantity=20,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="LOC",
                    price=Decimal("512.75"),
                    quantity=20,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )
        trade_2: TradeData = {
            "id": "502",
            "symbol": "QQQ",
            "strategy": Strategies.BounceBandit.value,
            "status": TradeStatus.ACTIVE.value,
        }

        # 3. Unsupported strategy: HoldTarget (must be excluded)
        order_3 = Order(
            id="order_3",
            symbol="SPY",
            quantity=15,
            mode="Exit",
            entry=None,
            exits=[
                OrderLeg(
                    action="SELL",
                    type="MKT",
                    price=Decimal("550.00"),
                    quantity=15,
                    time_in_force="DAY",
                )
            ],
            last_status="ACTIVE",
        )
        trade_3: TradeData = {
            "id": "503",
            "symbol": "SPY",
            "strategy": Strategies.HoldTarget.value,
            "status": TradeStatus.ACTIVE.value,
        }

        def resolve_fn(strat_name: str) -> Strategies | None:
            mapping = {
                Strategies.TwoPercent.value: Strategies.TwoPercent,
                Strategies.BounceBandit.value: Strategies.BounceBandit,
                Strategies.HoldTarget.value: Strategies.HoldTarget,
            }
            return mapping.get(strat_name)

        csv_file = write_csv_orders_file(
            orders_data=[
                (trade_1, order_1),
                (trade_2, order_2),
                (trade_3, order_3),
            ],
            date_string="2026-09-03",
            ibkr_account_id="U999888",
            resolve_strategy_fn=resolve_fn,
        )

        assert csv_file is not None
        assert csv_file.exists()

        content = csv_file.read_text(encoding="utf-8")

        expected_csv = (
            "trade_group_id,bracket_role,symbol,sec_type,exchange,account_id,action,quantity,order_type,target_price,tif,strategy_name,currency\n"
            "501_TwoPercent_SXRV,ENTRY,SXRV,STK,IBIS,U999888,BUY,10,LMT,1450.50,DAY,TwoPercent,EUR\n"
            "501_TwoPercent_SXRV,SL,SXRV,STK,IBIS,U999888,SELL,10,STP,1400.00,DAY,TwoPercent,EUR\n"
            "501_TwoPercent_SXRV,TP,SXRV,STK,IBIS,U999888,SELL,10,LMT,1500.00,DAY,TwoPercent,EUR\n"
            "502_BounceBandit_QQQ,EXIT,QQQ,STK,SMART,U999888,SELL,20,LOC,512.75,DAY,BounceBandit,\n"
        )

        # Exact character-by-character comparison (Golden Master)
        assert content == expected_csv

    finally:
        settings.app.order_overrides = original_overrides
