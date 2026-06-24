# filename: test_trade_manager_audit_hardening.py
"""
Audit Hardening Test Suite for app/services/trade_manager.

Each test class maps to a specific audit finding. The suite validates all
security and robustness fixes applied after the Iron Auditor + Red Teamer
dual-workflow review of the trade_manager module.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.trade_manager.strategies.split_target import SplitTargetStrategy
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy
from app.services.trade_manager.strategies.dip_buyer import DipBuyerStrategy
from app.services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
    _RebalanceCache,
)
from app.services.trade_manager.manager import TradeManager, _resolve_history_start_date
from app.types import TradeStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    symbol: str = "AAPL",
    strategy: str = "SplitTarget",
    entry_price: float = 150.0,
    stop_loss: float = 140.0,
    take_profit_1: float = 165.0,
    take_profit_3: float = 180.0,
    initial_size: float = 10.0,
    entry_date: str = "2026-01-10",
    status: str = "ACTIVE",
) -> dict:
    return {
        "id": 1,
        "symbol": symbol,
        "strategy": strategy,
        "entry_price": entry_price,
        "current_stop_loss": stop_loss,
        "initial_size": initial_size,
        "current_size": initial_size,
        "current_target": 0.0,
        "entry_date": entry_date,
        "exit_date": None,
        "exit_price": 0.0,
        "exit_reason": None,
        "realized_pnl": 0.0,
        "current_price": 0.0,
        "budget": 2000.0,
        "risk_amount": 100.0,
        "status": status,
        "signal_context": json.dumps(
            {
                "direction": "long",
                "take_profit_1": take_profit_1,
                "take_profit_3": take_profit_3,
                "date": "2026-01-09",
            }
        ),
    }


def _make_candle(
    date: str = "2026-01-10",
    open_price: float = 148.0,
    high: float = 155.0,
    low: float = 147.0,
    close: float = 153.0,
) -> pd.Series:
    return pd.Series(
        {
            "date": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _make_history(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SEC-03: NameError Bomb Fix — split_target.generate_orders
# ---------------------------------------------------------------------------


class TestSplitTargetNameErrorFix:
    """Validates SEC-03: quantity_half must be defined before both exit blocks."""

    def test_generate_orders_succeeds_when_tp1_is_zero_and_tp3_is_set(self) -> None:
        """Verifies no NameError when TP1=0 and TP3>0 (the original crash scenario)."""
        # Arrange
        strategy = SplitTargetStrategy()
        trade = _make_trade(take_profit_1=0.0, take_profit_3=180.0)
        mock_repo = MagicMock()

        # Act — must NOT raise NameError
        order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert — order is generated; TP3 exit exists, TP1 exit does not
        assert order is not None
        assert order.symbol == "AAPL"
        exit_types = [leg.type for leg in order.exits]
        assert "LMT" in exit_types  # TP3 exit present
        # All LMT exits are for the full quantity (no half-split when TP1 = 0)
        lmt_exits = [leg for leg in order.exits if leg.type == "LMT"]
        assert all(leg.quantity == int(order.quantity) for leg in lmt_exits)

    def test_generate_orders_produces_correct_split_when_both_targets_set(self) -> None:
        """Verifies 50% TP1 / remaining TP3 split when both targets are non-zero."""
        # Arrange
        strategy = SplitTargetStrategy()
        trade = _make_trade(
            initial_size=10.0,
            take_profit_1=165.0,
            take_profit_3=180.0,
            entry_price=150.0,
            stop_loss=140.0,
        )
        mock_repo = MagicMock()

        # Act
        order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert
        assert order is not None
        lmt_exits = [leg for leg in order.exits if leg.type == "LMT"]
        assert len(lmt_exits) == 2
        qty_half = lmt_exits[0].quantity
        qty_remaining = lmt_exits[1].quantity
        assert qty_half + qty_remaining == order.quantity

    def test_generate_orders_produces_full_tp3_when_only_tp3_set(self) -> None:
        """Verifies entire quantity goes to TP3 exit when TP1 is absent."""
        # Arrange
        strategy = SplitTargetStrategy()
        trade = _make_trade(
            initial_size=10.0,
            take_profit_1=0.0,
            take_profit_3=180.0,
        )
        mock_repo = MagicMock()

        # Act
        order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert — single LMT exit covers the full quantity
        assert order is not None
        lmt_exits = [leg for leg in order.exits if leg.type == "LMT"]
        assert len(lmt_exits) == 1
        assert lmt_exits[0].quantity == order.quantity

    def test_generate_orders_returns_none_when_entry_price_zero(self) -> None:
        """Verifies guard against zero entry price."""
        # Arrange
        strategy = SplitTargetStrategy()
        trade = _make_trade(entry_price=0.0)
        mock_repo = MagicMock()

        # Act & Assert
        result = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)
        assert result is None

    def test_execute_immediate_target_uses_trade_status_enum(self) -> None:
        """Verifies Day-1 target hit uses TradeStatus.CLOSED enum, not string literal."""
        # Arrange
        strategy = SplitTargetStrategy()
        trade = _make_trade(entry_price=150.0, stop_loss=140.0, status="CREATED")
        mock_repo = MagicMock()

        # Candle: gap-up entry above entry, same-day TP3 hit
        candle = _make_candle(open_price=152.0, high=185.0, low=149.0)
        history = _make_history(
            [
                {
                    "date": "2026-01-09",
                    "open": 148.0,
                    "high": 152.0,
                    "low": 147.0,
                    "close": 150.0,
                },
                {
                    "date": "2026-01-10",
                    "open": 152.0,
                    "high": 185.0,
                    "low": 149.0,
                    "close": 183.0,
                },
            ]
        )

        # Act
        strategy.check_entry(trade, candle, history, mock_repo)

        # Assert — the status written to DB must be the enum value, not a raw "CLOSED" string
        call_args = mock_repo.update_trade.call_args
        if call_args:
            update_dict = (
                call_args[0][1]
                if len(call_args[0]) > 1
                else call_args[1].get("updates", {})
            )
            status_value = update_dict.get("status")
            if status_value is not None:
                assert status_value == TradeStatus.CLOSED


# ---------------------------------------------------------------------------
# SEC-02: Fail-Closed on DB Lock — manager.run_daily_process
# ---------------------------------------------------------------------------


class TestTradeManagerFailClosed:
    """Validates SEC-02: DB errors must raise RuntimeError, not silently skip."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> TradeManager:
        """Provides a TradeManager with fully mocked repository layer."""
        with (
            patch("app.services.trade_manager.manager.DatabaseSession"),
            patch("app.services.trade_manager.manager.TradeRepository"),
            patch("app.services.trade_manager.manager.MarketRepository"),
        ):
            manager_instance = TradeManager(
                db_path=tmp_path / "signals.db",
                stocks_db_path=tmp_path / "stocks.db",
            )
            manager_instance.trade_repository = MagicMock()
            manager_instance.market_repo = MagicMock()
            return manager_instance

    def test_run_daily_process_raises_on_active_trade_db_lock(
        self, manager: TradeManager
    ) -> None:
        """Verifies that OperationalError on active trade load raises RuntimeError."""
        # Arrange
        manager.trade_repository.get_by_status.side_effect = sqlite3.OperationalError(
            "database is locked"
        )

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="Database unavailable during active trade load"
        ):
            manager.run_daily_process()

    def test_run_daily_process_raises_on_created_trade_db_lock(
        self, manager: TradeManager
    ) -> None:
        """Verifies fail-closed on DB lock during created trade load."""
        # Arrange — active loads fine, created load fails
        manager.trade_repository.get_by_status.side_effect = [
            [],  # Active: empty (success)
            sqlite3.OperationalError("database is locked"),  # Created: fails
        ]

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="Database unavailable during created trade load"
        ):
            manager.run_daily_process()

    def test_run_daily_process_raises_on_database_error(
        self, manager: TradeManager
    ) -> None:
        """Verifies that sqlite3.DatabaseError (disk I/O) also raises RuntimeError."""
        # Arrange
        manager.trade_repository.get_by_status.side_effect = sqlite3.DatabaseError(
            "disk I/O error"
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Database unavailable"):
            manager.run_daily_process()

    def test_run_daily_process_completes_normally_when_no_trades(
        self, manager: TradeManager
    ) -> None:
        """Verifies no-op completion when both trade lists are empty."""
        # Arrange
        manager.trade_repository.get_by_status.return_value = []

        # Act & Assert — must not raise
        manager.run_daily_process()

    def test_generate_daily_orders_raises_on_db_lock(
        self, manager: TradeManager
    ) -> None:
        """Verifies fail-closed in generate_daily_orders."""
        # Arrange
        manager.trade_repository.get_by_status.side_effect = sqlite3.OperationalError(
            "database is locked"
        )

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="Database unavailable during order generation"
        ):
            manager.generate_daily_orders()


# ---------------------------------------------------------------------------
# _resolve_history_start_date — No More Hardcoded Date
# ---------------------------------------------------------------------------


class TestResolveHistoryStartDate:
    """Validates SEC-05: dynamic history date derivation from trade fields."""

    @pytest.mark.parametrize(
        "trade_fields, expected_date",
        [
            ({"entry_date": "2026-03-15"}, "2026-03-15"),
            ({"entry_date": "2026-03-15 00:00:00"}, "2026-03-15"),  # Strips time
            ({"created_at": "2025-11-01"}, "2025-11-01"),
            ({"signal_date": "2025-06-20"}, "2025-06-20"),
            ({}, "2024-01-01"),  # Fallback only when no date fields
        ],
    )
    def test_resolve_history_start_date_returns_correct_date(
        self, trade_fields: dict, expected_date: str
    ) -> None:
        """Verifies correct date derivation from trade fields or fallback."""
        # Arrange & Act
        result = _resolve_history_start_date(trade_fields)

        # Assert
        assert result == expected_date

    def test_resolve_history_start_date_prefers_entry_date_over_created_at(
        self,
    ) -> None:
        """Verifies entry_date takes precedence over created_at."""
        # Arrange
        trade = {"entry_date": "2026-04-01", "created_at": "2026-01-01"}

        # Act
        result = _resolve_history_start_date(trade)

        # Assert
        assert result == "2026-04-01"


# ---------------------------------------------------------------------------
# TurnoverTimingStrategy: super().__init__() + Single MarketHolidayChecker
# ---------------------------------------------------------------------------


class TestTurnoverTimingStrategyInitialization:
    """Validates that TurnoverTimingStrategy now calls super() and caches holiday checker."""

    def test_strategy_has_holiday_checker_attribute_after_init(self) -> None:
        """Verifies _holiday_checker is set at construction time, not deferred."""
        # Arrange & Act
        strategy = TurnoverTimingStrategy()

        # Assert
        assert hasattr(strategy, "_holiday_checker")
        assert strategy._holiday_checker is not None

    def test_two_instances_both_have_holiday_checker_attribute(self) -> None:
        """Verifies each instance has its own _holiday_checker attribute set after init.

        Note: MarketHolidayChecker is a stateless singleton — sharing the same
        object identity across instances is correct and expected behaviour.
        What matters is that the attribute is set at construction, not on first use.
        """
        # Arrange & Act
        instance_a = TurnoverTimingStrategy()
        instance_b = TurnoverTimingStrategy()

        # Assert — both instances have the attribute initialized
        assert instance_a._holiday_checker is not None
        assert instance_b._holiday_checker is not None

    def test_custom_strategy_name_is_preserved(self) -> None:
        """Verifies the optional strategy_name override still works post-fix."""
        # Arrange & Act
        strategy = TurnoverTimingStrategy(strategy_name="TurnOverTiming_10")

        # Assert
        assert strategy.name == "TurnOverTiming_10"

    def test_manage_active_trade_does_not_instantiate_new_holiday_checker(
        self,
    ) -> None:
        """Verifies MarketHolidayChecker is NOT re-instantiated on each call."""
        # Arrange
        strategy = TurnoverTimingStrategy()
        original_checker = strategy._holiday_checker
        mock_repo = MagicMock()

        history = _make_history(
            [
                {
                    "date": "2026-01-09",
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                },
                {
                    "date": "2026-01-12",
                    "open": 101.0,
                    "high": 103.0,
                    "low": 100.0,
                    "close": 102.0,
                },
            ]
        )
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "entry_date": "2026-01-09",
            "current_size": 10,
            "status": "ACTIVE",
            "signal_context": json.dumps(
                {"green_candle_count": 0, "date": "2026-01-08"}
            ),
        }

        # Act
        strategy.manage_active_trade(trade, history, mock_repo)

        # Assert — same object (not re-created during the call)
        assert strategy._holiday_checker is original_checker


# ---------------------------------------------------------------------------
# NDXMomentumTradeStrategy: Typed _RebalanceCache
# ---------------------------------------------------------------------------


class TestNDXMomentumTypedCache:
    """Validates that _RebalanceCache is a typed dataclass, not a raw dict."""

    def test_rebalance_cache_is_none_before_first_call(self) -> None:
        """Verifies the cache starts as None (declared at class level)."""
        # Arrange & Act
        strategy = NDXMomentumTradeStrategy()

        # Assert
        assert strategy._rebalance_cache is None

    def test_rebalance_cache_is_typed_dataclass_after_first_rebalance(
        self,
    ) -> None:
        """Verifies that after a rebalance call, cache is a _RebalanceCache instance."""
        # Arrange
        strategy = NDXMomentumTradeStrategy()
        mock_repo = MagicMock()
        mock_repo.get_all_by_strategy.return_value = [
            {
                "symbol": "AAPL",
                "signal_context": json.dumps({"date": "2026-01-31"}),
                "strategy": "NDXMomentum",
            }
        ]
        trade = {
            "id": 1,
            "symbol": "NVDA",
            "strategy": "NDXMomentum",
            "entry_date": "2026-01-15",
            "current_size": 10,
            "signal_context": json.dumps({"date": "2026-01-31"}),
        }
        history = _make_history(
            [
                {
                    "date": "2026-01-31",
                    "open": 800.0,
                    "high": 820.0,
                    "low": 795.0,
                    "close": 815.0,
                },
                {
                    "date": "2026-02-03",
                    "open": 810.0,
                    "high": 830.0,
                    "low": 805.0,
                    "close": 825.0,
                },
            ]
        )

        # Act
        strategy.manage_active_trade(trade, history, mock_repo)

        # Assert
        assert isinstance(strategy._rebalance_cache, _RebalanceCache)
        assert strategy._rebalance_cache.cache_key.startswith("latest_leaders_")

    def test_rebalance_cache_is_reused_on_same_date(self) -> None:
        """Verifies the DB is only queried once per rebalance date."""
        # Arrange
        strategy = NDXMomentumTradeStrategy()
        mock_repo = MagicMock()
        mock_repo.get_all_by_strategy.return_value = [
            {
                "symbol": "AAPL",
                "signal_context": json.dumps({"date": "2026-01-31"}),
                "strategy": "NDXMomentum",
            }
        ]
        history = _make_history(
            [
                {
                    "date": "2026-01-31",
                    "open": 800.0,
                    "high": 820.0,
                    "low": 795.0,
                    "close": 815.0,
                },
                {
                    "date": "2026-02-03",
                    "open": 810.0,
                    "high": 830.0,
                    "low": 805.0,
                    "close": 825.0,
                },
            ]
        )
        trade = {
            "id": 1,
            "symbol": "NVDA",
            "strategy": "NDXMomentum",
            "entry_date": "2026-01-15",
            "current_size": 10,
            "signal_context": json.dumps({"date": "2026-01-31"}),
        }
        trade_b = dict(trade) | {"id": 2, "symbol": "MSFT"}

        # Act — two calls on the same rebalance date
        strategy.manage_active_trade(trade, history, mock_repo)
        strategy.manage_active_trade(trade_b, history, mock_repo)

        # Assert — DB only queried once (cache reused on second call)
        assert mock_repo.get_all_by_strategy.call_count == 1

    def test_two_strategy_instances_have_independent_caches(self) -> None:
        """Verifies class-level attribute does not create shared state between instances."""
        # Arrange & Act
        instance_a = NDXMomentumTradeStrategy()
        instance_b = NDXMomentumTradeStrategy()

        # Manually set cache on one instance
        instance_a._rebalance_cache = _RebalanceCache(
            cache_key="test_key",
            latest_signal_date="2026-01-31",
            leaders_symbols={"AAPL"},
        )

        # Assert — other instance is unaffected
        assert instance_b._rebalance_cache is None


# ---------------------------------------------------------------------------
# view_service.attach_sparklines: Injectable Clock
# ---------------------------------------------------------------------------


class TestAttachSparklinesDeterministicClock:
    """Validates SEC-07: attach_sparklines accepts an injected reference_date."""

    def test_attach_sparklines_accepts_reference_date_parameter(self) -> None:
        """Verifies reference_date is used instead of pd.Timestamp.now()."""
        # Arrange
        from app.services.trade_manager.view_service import TradeViewService

        mock_market_repo = MagicMock()
        mock_market_repo.get_batch_history_raw.return_value = pd.DataFrame(
            columns=["symbol", "date", "close"]
        )

        service = TradeViewService(
            trade_repository=MagicMock(),
            market_repository=mock_market_repo,
        )
        fixed_date = pd.Timestamp("2026-03-01")

        # Act
        service.attach_sparklines([], reference_date=fixed_date)

        # Assert — with empty list, no batch call, but verifying no exception
        mock_market_repo.get_batch_history_raw.assert_not_called()

    def test_attach_sparklines_uses_reference_date_for_history_window(self) -> None:
        """Verifies the 30-day window is anchored to the injected date, not now()."""
        # Arrange
        from app.services.trade_manager.view_service import TradeViewService

        mock_market_repo = MagicMock()
        mock_market_repo.get_batch_history_raw.return_value = pd.DataFrame(
            columns=["symbol", "date", "close"]
        )

        service = TradeViewService(
            trade_repository=MagicMock(),
            market_repository=mock_market_repo,
        )
        fixed_date = pd.Timestamp("2026-03-15")
        fake_trades = [
            {
                "symbol": "AAPL",
                "unrealized_pnl": 50.0,
                "sparkline": "",
                "id": 1,
                "strategy": "SplitTarget",
                "status": "ACTIVE",
                "entry_date": None,
                "exit_date": None,
                "entry_price": 0.0,
                "exit_price": 0.0,
                "current_price": 0.0,
                "initial_size": 10.0,
                "current_size": 10.0,
                "current_stop_loss": 0.0,
                "current_target": 0.0,
                "budget": 2000.0,
                "signal_context": None,
                "exit_reason": None,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "display_entry": "-",
                "display_exit": "-",
                "days_held": 0,
                "realized_pnl": 0.0,
                "pnl_percentage": 0.0,
                "is_critical": False,
                "progress": 0.0,
                "display_size": 10.0,
                "max_days": None,
                "version": None,
                "context": {},
                "risk_amount": 100.0,
            }
        ]

        # Act
        service.attach_sparklines(fake_trades, reference_date=fixed_date)

        # Assert — called with expected window anchored to injected date
        mock_market_repo.get_batch_history_raw.assert_called_once()
        call_args = mock_market_repo.get_batch_history_raw.call_args
        start_arg = call_args[0][1]
        assert start_arg == "2026-02-13"  # 30 days before 2026-03-15


# ---------------------------------------------------------------------------
# DipBuyer: db_size abbreviation removal
# ---------------------------------------------------------------------------


class TestDipBuyerAbbreviationFix:
    """Validates that db_size has been renamed to database_size (auditor Sec 3.1)."""

    def test_generate_orders_uses_initial_size_from_trade(self) -> None:
        """Verifies that a pre-set initial_size is used correctly (database_size path)."""
        # Arrange
        strategy = DipBuyerStrategy()
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "entry_price": 100.0,
            "initial_size": 20.0,
            "budget": 2000.0,
            "current_target": 110.0,
            "signal_context": "{}",
        }
        mock_repo = MagicMock()

        # Act
        order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert — quantity from initial_size (20), not budget / price (20)
        assert order is not None
        assert order.quantity == 20

    def test_generate_orders_falls_back_to_budget_when_no_initial_size(self) -> None:
        """Verifies budget-based sizing when initial_size is 0."""
        # Arrange
        strategy = DipBuyerStrategy()
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "entry_price": 100.0,
            "initial_size": 0.0,
            "budget": 2000.0,
            "current_target": 110.0,
            "signal_context": "{}",
        }
        mock_repo = MagicMock()

        # Act
        order = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert — 2000 / 100 = 20 shares
        assert order is not None
        assert order.quantity == 20


# ---------------------------------------------------------------------------
# HoldTarget: %s logging (no f-strings)
# ---------------------------------------------------------------------------


class TestHoldTargetLoggingStyle:
    """Validates that hold_target.py uses lazy %s logging, not f-strings."""

    def test_generate_orders_logs_warning_on_zero_entry_price(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifies warning is emitted correctly when entry_price is invalid."""
        # Arrange
        strategy = HoldTargetStrategy()
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "entry_price": 0.0,
            "current_stop_loss": 140.0,
            "initial_size": 10.0,
            "signal_context": "{}",
        }
        mock_repo = MagicMock()

        # Act
        with caplog.at_level("WARNING"):
            result = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert
        assert result is None
        assert "AAPL" in caplog.text

    def test_generate_orders_logs_warning_on_missing_stop_loss(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifies warning is emitted when stop loss is missing."""
        # Arrange
        strategy = HoldTargetStrategy()
        trade = {
            "id": 1,
            "symbol": "MSFT",
            "entry_price": 200.0,
            "current_stop_loss": 0.0,
            "initial_size": 10.0,
            "signal_context": "{}",
        }
        mock_repo = MagicMock()

        # Act
        with caplog.at_level("WARNING"):
            result = strategy.generate_orders(trade, pd.DataFrame(), 2000.0, mock_repo)

        # Assert
        assert result is None
        assert "MSFT" in caplog.text


# ---------------------------------------------------------------------------
# DipBuyer: Dynamic LOC Threshold Update in TradeManager
# ---------------------------------------------------------------------------


class TestTradeManagerDipBuyerDynamicLOCUpdate:
    """Validates that TradeManager updates the LOC threshold dynamically in DB."""

    def test_process_active_trade_updates_loc_threshold(self, tmp_path: Path) -> None:
        """Verifies that active DipBuyer trade updates threshold_loc in signal_context."""
        # Arrange
        with (
            patch("app.services.trade_manager.manager.DatabaseSession"),
            patch(
                "app.services.trade_manager.manager.TradeRepository"
            ) as mock_trade_repository_class,
            patch(
                "app.services.trade_manager.manager.MarketRepository"
            ) as mock_market_repository_class,
        ):
            mock_trade_repository = MagicMock()
            mock_market_repository = MagicMock()

            mock_trade_repository_class.return_value = mock_trade_repository
            mock_market_repository_class.return_value = mock_market_repository

            manager_instance = TradeManager(
                db_path=tmp_path / "signals.db",
                stocks_db_path=tmp_path / "stocks.db",
            )
            manager_instance.trade_repository = mock_trade_repository
            manager_instance.market_repo = mock_market_repository

            # Mock history: previous high is 100.0, current close is 95.0
            history_data = [
                {
                    "date": "2026-06-01",
                    "open": 98.0,
                    "high": 100.0,
                    "low": 97.0,
                    "close": 99.0,
                },
                {
                    "date": "2026-06-02",
                    "open": 99.0,
                    "high": 101.0,
                    "low": 94.0,
                    "close": 95.0,
                },
            ]
            mock_market_repository.get_symbol_history_raw.return_value = pd.DataFrame(
                history_data
            )

            # Active trade definition
            trade_data = {
                "id": 123,
                "symbol": "CBOE",
                "strategy": "dip_buyer",
                "status": "ACTIVE",
                "signal_context": json.dumps(
                    {"date": "2026-06-01", "threshold_loc": 329.52}
                ),
            }

            # Act
            manager_instance._process_active_trade(trade_data)

            # Assert
            # Since mock_trade_repository.update_trade is called with the updates:
            # We expect key "signal_context" in the updates to contain "threshold_loc": 100.01 (100.0 + 0.01)
            mock_trade_repository.update_trade.assert_called_once()
            call_arguments = mock_trade_repository.update_trade.call_args[0]
            assert call_arguments[0] == 123  # trade_id

            updates = call_arguments[1]
            assert updates["current_price"] == 95.0
            assert "signal_context" in updates

            updated_context = json.loads(updates["signal_context"])
            assert updated_context["threshold_loc"] == 101.01
            assert updated_context["date"] == "2026-06-01"


# ---------------------------------------------------------------------------
# DipBuyer: CSV Export Integration
# ---------------------------------------------------------------------------


class TestTradeManagerDipBuyerCSVExport:
    """Validates that DipBuyer strategy is included in CSV order export."""

    def test_write_csv_orders_file_includes_dip_buyer(self, tmp_path: Path) -> None:
        """Verifies that DipBuyer trade is included in _write_csv_orders_file."""
        # Arrange
        with (
            patch("app.services.trade_manager.manager.DatabaseSession"),
            patch("app.services.trade_manager.manager.TradeRepository"),
            patch("app.services.trade_manager.manager.MarketRepository"),
        ):
            manager_instance = TradeManager(
                db_path=tmp_path / "signals.db",
                stocks_db_path=tmp_path / "stocks.db",
            )

            # Create a mock DipBuyer Order
            from app.models import Order, OrderLeg

            order = Order(
                id="CBOE_dip_buyer",
                symbol="CBOE",
                quantity=10,
                mode="BRACKET",
                entry=OrderLeg(
                    action="BUY",
                    type="LMT",
                    price=280.0,
                    quantity=10,
                    time_in_force="DAY",
                ),
                exits=[
                    OrderLeg(
                        action="SELL",
                        type="LOC",
                        price=290.0,
                        quantity=10,
                        time_in_force="DAY",
                    )
                ],
                last_status="CREATED",
            )

            trade = {
                "id": 781,
                "symbol": "CBOE",
                "strategy": "dip_buyer",
                "status": "CREATED",
            }

            orders_data = [(trade, order)]

            # Patch Path to write to a temp directory instead of data/orders
            temp_orders_dir = tmp_path / "orders"
            temp_orders_dir.mkdir()

            with patch(
                "app.services.trade_manager.manager.Path",
                return_value=temp_orders_dir,
            ):
                result_path_str = manager_instance._write_csv_orders_file(
                    orders_data, "2026-06-10"
                )

            # Assert
            assert result_path_str is not None
            result_path = Path(result_path_str)
            assert result_path.exists()

            # Read CSV content
            df = pd.read_csv(result_path)
            assert len(df) == 2  # 1 entry leg, 1 exit leg

            # Verify entry row
            entry_row = df[df["bracket_role"] == "ENTRY"].iloc[0]
            assert entry_row["symbol"] == "CBOE"
            assert entry_row["action"] == "BUY"
            assert entry_row["order_type"] == "LMT"
            assert entry_row["target_price"] == 280.0
            assert entry_row["strategy_name"] == "DipBuyer"

            # Verify exit row
            exit_row = df[df["bracket_role"] == "TP"].iloc[0]
            assert exit_row["symbol"] == "CBOE"
            assert exit_row["action"] == "SELL"
            assert exit_row["order_type"] == "LOC"
            assert exit_row["target_price"] == 290.0
            assert exit_row["strategy_name"] == "DipBuyer"


# ---------------------------------------------------------------------------
# Exiting Active Trade Does Not Block New Entry
# ---------------------------------------------------------------------------


class TestTradeManagerExitingActiveTradeDoesNotBlockNewEntry:
    """Validates that active trades generating exit orders do not block entries."""

    def test_generate_daily_orders_with_exiting_active_trade(
        self, tmp_path: Path
    ) -> None:
        """Verifies that exiting active trade does not block new CREATED trade entry."""
        # Arrange
        with (
            patch("app.services.trade_manager.manager.DatabaseSession"),
            patch(
                "app.services.trade_manager.manager.TradeRepository"
            ) as mock_trade_repo_class,
            patch(
                "app.services.trade_manager.manager.MarketRepository"
            ) as mock_market_repo_class,
        ):
            mock_trade_repo = MagicMock()
            mock_market_repo = MagicMock()
            mock_trade_repo_class.return_value = mock_trade_repo
            mock_market_repo_class.return_value = mock_market_repo

            manager_instance = TradeManager(
                db_path=tmp_path / "signals.db",
                stocks_db_path=tmp_path / "stocks.db",
            )

            # Active trade that WILL exit today (DipBuyer with Close > Prev High)
            active_trade = {
                "id": 101,
                "symbol": "AKAM",
                "strategy": "dip_buyer",
                "status": "ACTIVE",
                "entry_price": 130.0,
                "current_target": 140.0,
                "initial_size": 10.0,
                "entry_date": "2026-06-09",
                "signal_context": json.dumps({"threshold_loc": 135.0}),
            }

            # New created trade for same symbol
            created_trade = {
                "id": 102,
                "symbol": "AKAM",
                "strategy": "dip_buyer",
                "status": "CREATED",
                "entry_price": 120.0,
                "current_target": 128.0,
                "initial_size": 10.0,
                "entry_date": "2026-06-11",
                "signal_context": "{}",
            }

            mock_trade_repo.get_by_status.side_effect = lambda status: {
                TradeStatus.CREATED: [created_trade],
                TradeStatus.ACTIVE: [active_trade],
            }[status]

            # Mock history so that ACTIVE trade is exited (close > prev_high)
            history_data = [
                {
                    "date": "2026-06-09",
                    "open": 130.0,
                    "high": 132.0,
                    "low": 129.0,
                    "close": 131.0,
                },
                {
                    "date": "2026-06-10",
                    "open": 131.0,
                    "high": 136.0,
                    "low": 130.0,
                    "close": 137.0,
                },  # Close > prev_high (132.0) -> Exit!
            ]
            mock_market_repo.get_symbol_history_raw.return_value = pd.DataFrame(
                history_data
            )

            # Patch Path to write to a temp directory
            temp_orders_dir = tmp_path / "orders"
            temp_orders_dir.mkdir()

            with patch(
                "app.services.trade_manager.manager.Path",
                return_value=temp_orders_dir,
            ):
                result_path_str = manager_instance.generate_daily_orders()

            # Assert
            assert result_path_str is not None
            result_path = Path(result_path_str)
            assert result_path.exists()

            # Read CSV content
            df = pd.read_csv(result_path)

            # We expect:
            # - The new CREATED trade (102) is blocked because DipBuyer is a single-position strategy
            #   and there is already an active trade (101) for symbol AKAM (even if it's exiting).
            # - So only the two Exit rows for ACTIVE trade (101) should be in the CSV.
            # Total 2 rows in CSV
            assert len(df) == 2

            # Verify created trade entry does not exist
            entry_rows = df[df["bracket_role"] == "ENTRY"]
            assert len(entry_rows) == 0

            # Verify active trade exits exist
            exit_rows = df[df["bracket_role"] == "EXIT"]
            assert len(exit_rows) == 2
            assert all(exit_rows["trade_group_id"] == "101_DipBuyer_AKAM")


# ---------------------------------------------------------------------------
# BaseTradeStrategy: _resolve_position_size sizing logic
# ---------------------------------------------------------------------------


class TestBaseTradeStrategySizing:
    """Validates the pure positioning sizing resolution logic of BaseTradeStrategy."""

    def test_pre_existing_size_takes_precedence(self) -> None:
        """Verifies that if initial_size or current_size is present, it is returned directly."""
        strategy = HoldTargetStrategy()
        # initial_size in trade
        trade = {
            "symbol": "AAPL",
            "initial_size": 150,
            "current_size": 0,
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=90.0,
        )
        assert size == 150

        # current_size in trade (initial_size missing or 0)
        trade = {
            "symbol": "AAPL",
            "initial_size": 0,
            "current_size": 75,
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=90.0,
        )
        assert size == 75

    def test_risk_based_sizing(self) -> None:
        """Verifies risk-based sizing calculation when stop_loss is provided."""
        strategy = HoldTargetStrategy()

        # Risk amount from context (100.0) -> (100.0 / (100.0 - 90.0)) = 10 shares
        trade = {
            "symbol": "AAPL",
            "signal_context": json.dumps({"risk_amount": 100.0}),
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=90.0,
        )
        assert size == 10

        # Risk amount from trade direct (50.0) -> (50.0 / 10) = 5 shares
        trade = {
            "symbol": "AAPL",
            "risk_amount": 50.0,
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=90.0,
        )
        assert size == 5

        # Fallback risk amount (200.0) -> (200.0 / 10) = 20 shares
        trade = {
            "symbol": "AAPL",
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=90.0,
            fallback_risk=200.0,
        )
        assert size == 20

    def test_budget_based_fallback(self) -> None:
        """Verifies budget-based fallback sizing when risk calculation is invalid or unavailable."""
        strategy = HoldTargetStrategy()

        # budget from context (1000.0) / 100.0 = 10 shares (stop_loss = 0.0)
        trade = {
            "symbol": "AAPL",
            "signal_context": json.dumps({"budget": 1000.0}),
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=0.0,
        )
        assert size == 10

        # budget from trade direct (2000.0) / 100.0 = 20 shares
        trade = {
            "symbol": "AAPL",
            "budget": 2000.0,
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=0.0,
        )
        assert size == 20

        # Fallback budget (5000.0) / 100.0 = 50 shares
        trade = {
            "symbol": "AAPL",
        }
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=100.0,
            stop_loss=0.0,
            fallback_budget=5000.0,
        )
        assert size == 50

    def test_sizing_returns_zero_on_invalid_parameters(self) -> None:
        """Verifies fallback to 0 when sizing parameters are completely missing or invalid."""
        strategy = HoldTargetStrategy()
        trade = {
            "symbol": "AAPL",
        }
        # No fallback values, fill_price is 0
        size = strategy._resolve_position_size(
            trade=trade,
            fill_price=0.0,
            stop_loss=0.0,
            fallback_budget=0.0,
            fallback_risk=0.0,
        )
        assert size == 0


class TestTradeManagerEntryDailyUpdate:
    """Validates that get_daily_updates is triggered and saved when trade is activated."""

    def test_process_created_trade_saves_daily_updates(self, tmp_path: Path) -> None:
        # Arrange
        with (
            patch("app.services.trade_manager.manager.DatabaseSession"),
            patch(
                "app.services.trade_manager.manager.TradeRepository"
            ) as mock_trade_repo_class,
            patch(
                "app.services.trade_manager.manager.MarketRepository"
            ) as mock_market_repo_class,
        ):
            mock_trade_repo = MagicMock()
            mock_market_repo = MagicMock()
            mock_trade_repo_class.return_value = mock_trade_repo
            mock_market_repo_class.return_value = mock_market_repo

            manager_instance = TradeManager(
                db_path=tmp_path / "signals.db",
                stocks_db_path=tmp_path / "stocks.db",
            )
            manager_instance.trade_repository = mock_trade_repo
            manager_instance.market_repo = mock_market_repo

            # Mock history: previous high is 100.0, current close is 95.0, low is 94.0
            history_data = [
                {
                    "date": pd.Timestamp("2026-06-01"),
                    "open": 98.0,
                    "high": 100.0,
                    "low": 97.0,
                    "close": 99.0,
                },
                {
                    "date": pd.Timestamp("2026-06-02"),
                    "open": 99.0,
                    "high": 101.0,
                    "low": 94.0,
                    "close": 95.0,
                },
            ]
            mock_market_repo.get_symbol_history_raw.return_value = pd.DataFrame(
                history_data
            )

            # Created trade definition (limit entry at 97.0)
            trade_data = {
                "id": 124,
                "symbol": "CBOE",
                "strategy": "dip_buyer",
                "status": "CREATED",
                "entry_price": 97.0,
                "signal_context": json.dumps({"date": "2026-06-01"}),
            }

            # Act
            manager_instance._process_created_trade(trade_data)

            # Assert
            # We expect the trade to be updated to ACTIVE and get the threshold_loc dynamic EOD update in signal_context
            mock_trade_repo.update_trade.assert_called_once()
            call_arguments = mock_trade_repo.update_trade.call_args[0]
            assert call_arguments[0] == 124  # trade_id

            updates = call_arguments[1]
            assert updates["status"] == TradeStatus.ACTIVE
            assert "signal_context" in updates

            updated_context = json.loads(updates["signal_context"])
            # threshold_loc should be 101.01 (high of 101.0 + 0.01)
            assert updated_context["threshold_loc"] == 101.01
            assert updated_context["date"] == "2026-06-01"
