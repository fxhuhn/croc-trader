"""Unit tests for generic backfill engine and strategy dispatcher."""

import warnings
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd  # type: ignore[import-untyped]
import pytest

from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.services.backfill_engine import (
    STRATEGY_MAP,
    run_generic_backfill,
    run_strategy_backfill,
)
from app.services.bounce_bandit_backfill import run_bounce_bandit_backfill
from app.services.bridge_scout_backfill import run_bridge_scout_backfill
from app.services.tgim_backfill import run_tgim_backfill
from app.services.trade_manager.types import TradeTransition


def test_strategy_map_contains_expected_strategies() -> None:
    """Verifies all supported strategies are registered in STRATEGY_MAP."""
    assert "tgim" in STRATEGY_MAP
    assert "bridge_scout" in STRATEGY_MAP
    assert "bounce_bandit" in STRATEGY_MAP


def test_run_strategy_backfill_invalid_strategy_raises_value_error() -> None:
    """Verifies ValueError is raised for unregistered strategy names."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()

    with pytest.raises(ValueError, match="Unknown strategy for backfill"):
        run_strategy_backfill(
            stocks_session=mock_stocks_session,
            signals_session=mock_signals_session,
            strategy_name="non_existent_strategy",
        )


@patch("app.services.backfill_engine.run_generic_backfill")
def test_run_strategy_backfill_dispatches_correctly(
    mock_run_generic: MagicMock,
) -> None:
    """Verifies run_strategy_backfill resolves strategy parameters and calls run_generic_backfill."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()
    mock_run_generic.return_value = {"status": "ok"}

    result = run_strategy_backfill(
        stocks_session=mock_stocks_session,
        signals_session=mock_signals_session,
        strategy_name="bridge_scout",
        start_date="2025-01-01",
        budget=5000.0,
    )

    assert result == {"status": "ok"}
    mock_run_generic.assert_called_once()
    kwargs = mock_run_generic.call_args.kwargs
    assert kwargs["strategy_name"] == "bridge_scout"
    assert kwargs["lookback_days"] == 60
    assert kwargs["budget"] == 5000.0


def test_run_generic_backfill_execution(tmp_path: Path) -> None:
    """Verifies run_generic_backfill executes screener, entry check and trade management loop."""
    stocks_db = tmp_path / "stocks.db"
    signals_db = tmp_path / "signals.db"
    stocks_session = DatabaseSession(str(stocks_db))
    signals_session = DatabaseSession(str(signals_db))

    trade_repo = TradeRepository(signals_session)
    trade_repo.init_schema()

    # Pre-create a candidate trade in CREATED status
    trade_repo.create_trade(
        symbol="AAPL",
        strategy="test_strat",
        size=10,
        entry=100.0,
        stop_loss=90.0,
        target=120.0,
        context={"date": "2026-08-03"},
    )

    class MockScreener:
        def __init__(self, repo: Any, provider: Any) -> None:
            pass

        def run(self, days: int = 0, analysis_date: str | None = None) -> int:
            return 1

    class MockEngine:
        def check_entry(
            self, trade: dict[str, Any], candle: pd.Series, df_sim: pd.DataFrame
        ) -> TradeTransition:
            return TradeTransition(
                reason="Entry Triggered",
                updates={
                    "status": "ACTIVE",
                    "entry_date": "2026-08-03",
                    "entry_price": 100.0,
                },
                message="Entry msg",
            )

        def manage_active_trade(
            self, trade: dict[str, Any], df_sim: pd.DataFrame
        ) -> TradeTransition:
            return TradeTransition(
                reason="Target Reached",
                updates={
                    "status": "CLOSED",
                    "exit_date": "2026-08-04",
                    "exit_price": 110.0,
                    "realized_pnl": 100.0,
                    "exit_reason": "TARGET_HIT",
                },
                message="Exit msg",
            )

    mock_history = {
        "AAPL": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-03", "2026-08-04"]),
                "open": [99.0, 100.0],
                "high": [101.0, 112.0],
                "low": [98.0, 99.0],
                "close": [100.0, 110.0],
                "volume": [1000, 1500],
            }
        )
    }

    with patch("app.services.backfill_engine.MarketDataProvider") as mock_provider_cls:
        mock_provider = cast(MagicMock, mock_provider_cls.return_value)
        mock_provider.get_batch_history.return_value = mock_history

        res = run_generic_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            strategy_name="test_strat",
            screener_class=cast(Any, MockScreener),
            strategy_engine_class=cast(Any, MockEngine),
            lookback_days=30,
            start_date="2026-08-03",
            end_date="2026-08-04",
            budget=1000.0,
            clear_existing=False,
        )

        signals_gen = cast(int, res["signals_generated"])
        assert signals_gen > 0
        assert res["trades_filled"] == 1
        assert res["trades_closed"] == 1
        assert res["total_pnl"] == 100.0
        assert res["win_rate"] == 100.0


def test_run_generic_backfill_clear_existing(tmp_path: Path) -> None:
    """Verifies clear_existing=True deletes prior trades for strategy."""
    signals_db = tmp_path / "signals.db"
    signals_session = DatabaseSession(str(signals_db))
    trade_repo = TradeRepository(signals_session)
    trade_repo.init_schema()

    trade_repo.create_trade(
        symbol="AAPL",
        strategy="test_strat",
        size=10,
        entry=100.0,
        stop_loss=90.0,
        target=110.0,
        context={},
    )

    class MockScreener:
        def __init__(self, repo: Any, provider: Any) -> None:
            pass

        def run(self, days: int = 0, analysis_date: str | None = None) -> int:
            return 0

    class MockEngine:
        pass

    with patch("app.services.backfill_engine.MarketDataProvider"):
        run_generic_backfill(
            stocks_session=MagicMock(),
            signals_session=signals_session,
            strategy_name="test_strat",
            screener_class=cast(Any, MockScreener),
            strategy_engine_class=cast(Any, MockEngine),
            lookback_days=30,
            start_date="2026-08-03",
            end_date="2026-08-03",
            clear_existing=True,
        )

    from app.types import TradeStatus

    # Pre-existing trade was deleted
    assert trade_repo.get_by_status([TradeStatus.CREATED, TradeStatus.ACTIVE]) == []


def test_deprecated_wrappers_issue_deprecation_warnings() -> None:
    """Verifies legacy backfill functions emit DeprecationWarning when called."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()

    with patch("app.services.backfill_engine.run_generic_backfill") as mock_run:
        mock_run.return_value = {}

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_bridge_scout_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_bridge_scout_backfill is deprecated" in str(record[0].message)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_bounce_bandit_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_bounce_bandit_backfill is deprecated" in str(record[0].message)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_tgim_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_tgim_backfill is deprecated" in str(record[0].message)
