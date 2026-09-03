"""Pinning and characterization test suite for TradeManager (manager.py).

Secures baseline behavior and error branches before refactoring:
1. Strategy name resolution (direct, alias, fuzzy regex keywords, invalid).
2. Database fail-closed error propagation across all worker methods.
3. Market recency verification, candle sync, and deferral.
4. Signal context JSON handling and malformed JSON resilience.
5. Single position strategy entry suppression.
6. Order generation fallbacks and date resolution routines.
"""

import datetime
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.const import Strategies
from app.models import Order, OrderLeg
from app.services.trade_manager.manager import (
    _HARDCODED_HISTORY_FALLBACK_DATE,
    TradeManager,
    _extract_anchor_date,
    _resolve_history_start_date,
)
from app.services.trade_manager.types import TradeTransition
from app.types import TradeStatus


@pytest.fixture
def manager(tmp_path: Path) -> TradeManager:
    """Fixture providing TradeManager with mocked repositories."""
    with (
        patch("app.services.trade_manager.manager.DatabaseSession"),
        patch("app.services.trade_manager.manager.TradeRepository"),
        patch("app.services.trade_manager.manager.MarketRepository"),
    ):
        tm = TradeManager(
            db_path=tmp_path / "signals.db",
            stocks_db_path=tmp_path / "stocks.db",
            ibkr_account_id="U1234567",
        )
        tm.trade_repository = MagicMock()
        tm.market_repository = MagicMock()
        return tm


def _mock_trade_repo(manager: TradeManager) -> MagicMock:
    return cast(MagicMock, manager.trade_repository)


def _mock_market_repo(manager: TradeManager) -> MagicMock:
    return cast(MagicMock, manager.market_repository)


# ---------------------------------------------------------------------------
# 1. Strategy Name & Handler Resolution
# ---------------------------------------------------------------------------


def test_resolve_strategy_name_all_branches(manager: TradeManager) -> None:
    """Verifies all resolution branches in _resolve_strategy_name."""
    # Empty / None
    assert manager._resolve_strategy_name("") is None

    # Canonical enum string
    assert manager._resolve_strategy_name("two_percent") == Strategies.TwoPercent

    # Alias matching
    assert manager._resolve_strategy_name("twopercent") == Strategies.TwoPercent

    # Fuzzy regex keyword matching
    assert (
        manager._resolve_strategy_name("custom_turnover_v2")
        == Strategies.TurnOverTiming
    )
    assert manager._resolve_strategy_name("dip_hunter") == Strategies.DipBuyer
    assert manager._resolve_strategy_name("hold_forever") == Strategies.HoldTarget
    assert manager._resolve_strategy_name("tp3_target") == Strategies.HoldTarget
    assert manager._resolve_strategy_name("split_run") == Strategies.SplitTarget
    assert manager._resolve_strategy_name("two_percent_flow") == Strategies.TwoPercent

    # Unknown strategy string
    assert manager._resolve_strategy_name("completely_unknown_xyz") is None


def test_get_strategy_unknown_returns_none(manager: TradeManager) -> None:
    """Verifies that unknown strategy returns None without exception."""
    strategy = manager._get_strategy("non_existent_strat")
    assert strategy is None


# ---------------------------------------------------------------------------
# 2. Market Data Update & Recency Sync
# ---------------------------------------------------------------------------


def test_attempt_targeted_market_update_exception_handled(
    manager: TradeManager,
) -> None:
    """Verifies that exceptions during targeted market updates are logged and swallowed."""
    with patch(
        "app.services.trade_manager.manager.MarketDataUpdater"
    ) as mock_updater_cls:
        mock_updater = MagicMock()
        mock_updater.run_update.side_effect = RuntimeError("Network timeout")
        mock_updater_cls.return_value = mock_updater

        # Must not raise
        manager._attempt_targeted_market_update("AAPL")


def test_verify_and_sync_recency_deferral(manager: TradeManager) -> None:
    """Verifies candle check: missing candle -> attempt sync -> deferral if still missing."""
    history = pd.DataFrame(
        [
            {
                "date": "2026-08-28",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
            }
        ]
    )

    manager.holiday_checker = MagicMock()
    with patch(
        "app.services.trade_manager.manager.get_last_completed_trading_day",
        return_value=datetime.date(2026, 9, 1),
    ):
        # targeted update returns still outdated history
        _mock_market_repo(manager).get_symbol_history_raw.return_value = history

        is_valid, returned_df = manager._verify_and_sync_recency(
            symbol="AAPL",
            history_dataframe=history,
            start_date="2026-08-01",
            reference_date="2026-09-02",
        )

        assert is_valid is False
        assert len(returned_df) == 1


def test_verify_and_sync_recency_success_after_sync(
    manager: TradeManager,
) -> None:
    """Verifies that if targeted update provides the candle, recency check succeeds."""
    stale_history = pd.DataFrame(
        [
            {
                "date": "2026-08-28",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
            }
        ]
    )
    fresh_history = pd.DataFrame(
        [
            {
                "date": "2026-09-01",
                "open": 104.0,
                "high": 106.0,
                "low": 103.0,
                "close": 105.0,
            }
        ]
    )

    manager.holiday_checker = MagicMock()
    with patch(
        "app.services.trade_manager.manager.get_last_completed_trading_day",
        return_value=datetime.date(2026, 9, 1),
    ):
        _mock_market_repo(manager).get_symbol_history_raw.return_value = fresh_history

        is_valid, returned_df = manager._verify_and_sync_recency(
            symbol="AAPL",
            history_dataframe=stale_history,
            start_date="2026-08-01",
            reference_date=None,  # Tests datetime.datetime.now() fallback
        )

        assert is_valid is True
        assert returned_df.iloc[-1]["date"] == "2026-09-01"


# ---------------------------------------------------------------------------
# 3. Fail-Closed Error Handling (sqlite3.DatabaseError -> RuntimeError)
# ---------------------------------------------------------------------------


def test_process_active_trade_database_error_raises_runtime_error(
    manager: TradeManager,
) -> None:
    """Validates that a DB error during active trade processing raises RuntimeError."""
    trade = {
        "id": 1,
        "symbol": "AAPL",
        "strategy": Strategies.DipBuyer.value,
        "status": TradeStatus.ACTIVE.value,
    }
    _mock_market_repo(
        manager
    ).get_symbol_history_raw.side_effect = sqlite3.OperationalError("disk I/O error")

    with pytest.raises(RuntimeError, match="Database error processing active trade"):
        manager._process_active_trade(trade)


def test_process_created_trade_database_error_raises_runtime_error(
    manager: TradeManager,
) -> None:
    """Validates that a DB error during created trade processing raises RuntimeError."""
    trade = {
        "id": 2,
        "symbol": "MSFT",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
    }
    _mock_market_repo(
        manager
    ).get_symbol_history_raw.side_effect = sqlite3.DatabaseError(
        "database disk image is malformed"
    )

    with pytest.raises(RuntimeError, match="Database error processing created trade"):
        manager._process_created_trade(trade)


def test_collect_exit_orders_database_error_raises_runtime_error(
    manager: TradeManager,
) -> None:
    """Validates DB error during exit order collection raises RuntimeError."""
    active_trades = [
        {
            "id": 3,
            "symbol": "QQQ",
            "strategy": Strategies.TwoPercent.value,
            "status": TradeStatus.ACTIVE.value,
        }
    ]
    with patch.object(
        manager,
        "_generate_order_for_trade",
        side_effect=sqlite3.OperationalError("database locked"),
    ):
        with pytest.raises(RuntimeError, match="Database error generating exit order"):
            manager._collect_exit_orders_for_active_trades(
                active_trades, created_symbols=set()
            )


def test_collect_entry_orders_database_error_raises_runtime_error(
    manager: TradeManager,
) -> None:
    """Validates DB error during entry order collection raises RuntimeError."""
    created_trades = [
        {
            "id": 4,
            "symbol": "NVDA",
            "strategy": Strategies.TwoPercent.value,
            "status": TradeStatus.CREATED.value,
        }
    ]
    with patch.object(
        manager,
        "_generate_order_for_trade",
        side_effect=sqlite3.OperationalError("database locked"),
    ):
        with pytest.raises(RuntimeError, match="Database error generating order"):
            manager._collect_entry_orders_for_created_trades(
                created_trades,
                created_symbols=set(),
                active_symbols_by_strategy={},
            )


# ---------------------------------------------------------------------------
# 4. Data-Level Error Handling (Warning & Non-Crashing)
# ---------------------------------------------------------------------------


def test_process_active_trade_data_error_swallowed(
    manager: TradeManager,
) -> None:
    """Validates ValueError/KeyError during active trade processing is logged as warning."""
    trade = {
        "id": 5,
        "symbol": "AAPL",
        "strategy": Strategies.DipBuyer.value,
        "status": TradeStatus.ACTIVE.value,
    }
    history = pd.DataFrame([{"date": "2026-09-01", "close": 150.0}])
    _mock_market_repo(manager).get_symbol_history_raw.return_value = history

    strategy_mock = MagicMock()
    strategy_mock.manage_active_trade.side_effect = ValueError("Corrupt calculation")
    manager.strategies[Strategies.DipBuyer] = strategy_mock

    # Must not raise
    manager._process_active_trade(trade)


def test_process_created_trade_data_error_swallowed(
    manager: TradeManager,
) -> None:
    """Validates KeyError/TypeError during created trade processing is logged as warning."""
    trade = {
        "id": 6,
        "symbol": "MSFT",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
    }
    history = pd.DataFrame([{"date": "2026-09-01", "close": 200.0}])
    _mock_market_repo(manager).get_symbol_history_raw.return_value = history

    strategy_mock = MagicMock()
    strategy_mock.check_entry.side_effect = KeyError("missing_column")
    manager.strategies[Strategies.TwoPercent] = strategy_mock

    # Must not raise
    manager._process_created_trade(trade)


# ---------------------------------------------------------------------------
# 5. Signal Context JSON Handling & Malformed String Resilience
# ---------------------------------------------------------------------------


def test_apply_active_trade_updates_malformed_json(
    manager: TradeManager,
) -> None:
    """Verifies that malformed JSON in trade.signal_context does not crash updates."""
    trade = {
        "id": 10,
        "symbol": "AAPL",
        "signal_context": "{broken_json_not_valid",
    }
    history = pd.DataFrame([{"date": "2026-09-01", "close": 150.0}])
    strategy_mock = MagicMock()
    strategy_mock.get_daily_updates.return_value = {"updated_stop": 145.0}

    # Should log warning but still update current_price
    manager._apply_active_trade_updates(
        trade=trade,
        symbol="AAPL",
        history_dataframe=history,
        strategy=strategy_mock,
        transition=None,
    )
    _mock_trade_repo(manager).update_trade.assert_called_once_with(
        10, {"current_price": 150.0}
    )


def test_process_created_trade_activates_with_daily_updates(
    manager: TradeManager,
) -> None:
    """Verifies transition to ACTIVE applies daily_updates into signal_context JSON."""
    trade = {
        "id": 20,
        "symbol": "TSLA",
        "strategy": Strategies.TwoPercent.value,
        "status": TradeStatus.CREATED.value,
        "signal_context": json.dumps({"origin": "scan"}),
    }
    history = pd.DataFrame([{"date": "2026-09-01", "close": 250.0}])
    _mock_market_repo(manager).get_symbol_history_raw.return_value = history

    strategy_mock = MagicMock()
    strategy_mock.check_entry.return_value = TradeTransition(
        updates={"status": TradeStatus.ACTIVE},
        reason="Entry Hit",
        message="Triggered entry",
    )
    strategy_mock.get_daily_updates.return_value = {"atr": 5.2}
    manager.strategies[Strategies.TwoPercent] = strategy_mock

    manager._process_created_trade(trade)

    _mock_trade_repo(manager).update_trade.assert_called_once()
    args, kwargs = _mock_trade_repo(manager).update_trade.call_args
    assert args[0] == 20
    saved_updates = args[1]
    saved_context = json.loads(saved_updates["signal_context"])
    assert saved_context["origin"] == "scan"
    assert saved_context["atr"] == 5.2


# ---------------------------------------------------------------------------
# 6. Single Position Strategy Entry Suppression
# ---------------------------------------------------------------------------


def test_collect_entry_orders_skips_active_single_position(
    manager: TradeManager,
) -> None:
    """Verifies that created trades are skipped if the symbol is active in a single-position strategy."""
    created_trades = [
        {
            "id": 30,
            "symbol": "SPY",
            "strategy": Strategies.NDXMomentum.value,
        }
    ]
    active_symbols_by_strategy = {
        Strategies.NDXMomentum: {"SPY"},
    }

    entry_orders = manager._collect_entry_orders_for_created_trades(
        created_trades=created_trades,
        created_symbols={"SPY"},
        active_symbols_by_strategy=active_symbols_by_strategy,
    )
    assert len(entry_orders) == 0


# ---------------------------------------------------------------------------
# 7. Date Resolution & History Lookback Functions
# ---------------------------------------------------------------------------


def test_extract_anchor_date_precedence() -> None:
    """Verifies anchor date extraction precedence (entry_date > signal_context > created_at)."""
    # 1. entry_date
    assert _extract_anchor_date({"entry_date": "2026-05-10 12:00:00"}) == "2026-05-10"

    # 2. signal_context JSON
    assert (
        _extract_anchor_date({"signal_context": json.dumps({"date": "2026-05-09"})})
        == "2026-05-09"
    )

    # 3. signal_context dict directly
    assert (
        _extract_anchor_date({"signal_context": {"date": "2026-05-08"}}) == "2026-05-08"
    )

    # 4. signal_context malformed JSON -> fallback to created_at
    assert (
        _extract_anchor_date(
            {
                "signal_context": "{bad_json",
                "created_at": "2026-05-07 09:00:00",
            }
        )
        == "2026-05-07"
    )

    # 5. Empty trade dict
    assert _extract_anchor_date({}) is None


def test_resolve_history_start_date_calculations() -> None:
    """Verifies lookback calculation from anchor date."""
    # No anchor -> fallback constant
    assert _resolve_history_start_date({}) == _HARDCODED_HISTORY_FALLBACK_DATE

    # 0 lookback days -> returns anchor date directly
    trade: dict[str, object] = {"entry_date": "2026-06-15"}
    assert _resolve_history_start_date(trade, lookback_days=0) == "2026-06-15"

    # Positive lookback days -> subtracted
    assert _resolve_history_start_date(trade, lookback_days=10) == "2026-06-05"

    # Invalid timestamp format -> returns raw anchor date
    invalid_date_trade: dict[str, object] = {"entry_date": "not_a_valid_date"}
    assert (
        _resolve_history_start_date(invalid_date_trade, lookback_days=30)
        == "not_a_valid_date"
    )


# ---------------------------------------------------------------------------
# 8. Order Generation Fallbacks
# ---------------------------------------------------------------------------


def test_generate_daily_orders_returns_none_when_empty(
    manager: TradeManager,
) -> None:
    """Verifies that generate_daily_orders returns None when no orders are produced."""
    _mock_trade_repo(manager).get_by_status.return_value = []
    result = manager.generate_daily_orders()
    assert result is None


def test_generate_order_for_trade_unregistered_strategy_returns_none(
    manager: TradeManager,
) -> None:
    """Verifies that an unhandled strategy returns None gracefully."""
    trade = {"id": 99, "symbol": "XYZ", "strategy": "unknown_future_strat"}
    result = manager._generate_order_for_trade(trade)
    assert result is None


def test_trade_manager_init_default_account_id_warning(tmp_path: Path) -> None:
    """Verifies default account id warning when IBKR_ACCOUNT_ID is not configured."""
    with (
        patch("app.services.trade_manager.manager.DatabaseSession"),
        patch("app.services.trade_manager.manager.TradeRepository"),
        patch("app.services.trade_manager.manager.MarketRepository"),
        patch.dict("os.environ", {}, clear=True),
    ):
        tm = TradeManager(
            db_path=tmp_path / "signals.db",
            stocks_db_path=tmp_path / "stocks.db",
            ibkr_account_id=None,
        )
        assert tm._ibkr_account_id == "YOUR_IBKR_ACCOUNT"


def test_process_active_trade_no_strategy_or_empty_history(
    manager: TradeManager,
) -> None:
    """Verifies early exit when strategy is missing or history is empty."""
    # Unknown strategy
    manager._process_active_trade(
        {"id": 1, "symbol": "AAPL", "strategy": "unknown_strat"}
    )
    _mock_trade_repo(manager).update_trade.assert_not_called()

    # Empty history
    _mock_market_repo(manager).get_symbol_history_raw.return_value = pd.DataFrame()
    manager._process_active_trade(
        {"id": 2, "symbol": "AAPL", "strategy": Strategies.DipBuyer.value}
    )
    _mock_trade_repo(manager).update_trade.assert_not_called()


def test_process_created_trade_no_strategy_or_empty_history(
    manager: TradeManager,
) -> None:
    """Verifies early exit for created trade when strategy missing or history empty."""
    # Unknown strategy
    manager._process_created_trade(
        {"id": 3, "symbol": "MSFT", "strategy": "unknown_strat"}
    )
    _mock_trade_repo(manager).update_trade.assert_not_called()

    # Empty history
    _mock_market_repo(manager).get_symbol_history_raw.return_value = pd.DataFrame()
    manager._process_created_trade(
        {"id": 4, "symbol": "MSFT", "strategy": Strategies.TwoPercent.value}
    )
    _mock_trade_repo(manager).update_trade.assert_not_called()


def test_apply_active_trade_updates_with_transition(
    manager: TradeManager,
) -> None:
    """Verifies apply updates when transition is present."""
    history = pd.DataFrame([{"date": "2026-09-01", "close": 155.0}])
    transition = TradeTransition(
        updates={"status": TradeStatus.CLOSED, "exit_price": 155.0},
        reason="Target Reached",
        message="Closed at target",
    )
    strategy_mock = MagicMock()
    strategy_mock.get_daily_updates.return_value = None

    manager._apply_active_trade_updates(
        trade={"id": 55, "symbol": "AAPL"},
        symbol="AAPL",
        history_dataframe=history,
        strategy=strategy_mock,
        transition=transition,
    )
    _mock_trade_repo(manager).update_trade.assert_called_once_with(
        55,
        {"current_price": 155.0, "status": TradeStatus.CLOSED, "exit_price": 155.0},
        reason="Target Reached",
    )


def test_collect_exit_orders_skips_unresolvable_strategy(
    manager: TradeManager,
) -> None:
    """Verifies active trade with unresolvable strategy is skipped."""
    active_trades = [{"id": 1, "symbol": "XYZ", "strategy": "unknown_xyz"}]
    orders, blocked = manager._collect_exit_orders_for_active_trades(
        active_trades, created_symbols=set()
    )
    assert len(orders) == 0


def test_collect_entry_orders_success(manager: TradeManager) -> None:
    """Verifies created trade successfully appends generated order."""
    created_trades = [
        {"id": 1, "symbol": "AAPL", "strategy": Strategies.DipBuyer.value}
    ]
    mock_order = Order(
        id="ord_1",
        symbol="AAPL",
        quantity=10,
        mode="Entry",
        entry=OrderLeg(action="BUY", type="LMT", price=Decimal("150.00")),
        exits=[],
    )
    with patch.object(manager, "_generate_order_for_trade", return_value=mock_order):
        orders = manager._collect_entry_orders_for_created_trades(
            created_trades,
            created_symbols=set(),
            active_symbols_by_strategy={},
        )
        assert len(orders) == 1
        assert orders[0][1] == mock_order


def test_generate_daily_orders_when_write_csv_returns_none(
    manager: TradeManager,
) -> None:
    """Verifies generate_daily_orders returns None when write_csv_orders_file returns None."""
    _mock_trade_repo(manager).get_by_status.return_value = [
        {"id": 1, "symbol": "AAPL", "strategy": Strategies.DipBuyer.value}
    ]
    mock_order = Order(
        id="ord_1",
        symbol="AAPL",
        quantity=10,
        mode="Entry",
        entry=OrderLeg(action="BUY", type="LMT", price=Decimal("150.00")),
        exits=[],
    )
    with (
        patch.object(manager, "_generate_order_for_trade", return_value=mock_order),
        patch(
            "app.services.trade_manager.manager.write_csv_orders_file",
            return_value=None,
        ),
    ):
        result = manager.generate_daily_orders()
        assert result is None
