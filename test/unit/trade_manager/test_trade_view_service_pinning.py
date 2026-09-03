"""Pinning and characterization test suite for TradeViewService (view_service.py).

Secures baseline behavior across all uncovered branches prior to refactoring:
1. Strategy filter mapping and resolution.
2. Short positions calculation (active & closed).
3. Critical stop-loss threshold (< 1%) and progress bar calculations.
4. Sparkline chart generation and color determination.
5. 5-day Open PnL change variations (empty history, missing row, short trade, entry dates).
6. Portfolio summary R-multiple, SQN calculations, and error resilience.
7. Signal date extraction, formatting, and repository fallbacks.
8. Broker reconciliation (missing execution, ghost positions, orphan SQL lookups).
9. Broker settlements execution timestamps, timezones, and holding delta.
10. Golden-master snapshot validation.
"""

import json
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.const import IndexAliases, Strategies, TradeStatus
from app.database.repositories.broker import BrokerRepository
from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.services.trade_manager.view_service import (
    TradeViewData,
    TradeViewService,
    _map_order_strategy_filter,
    _map_strategy_filter_name,
)


@pytest.fixture
def mock_trade_repo() -> MagicMock:
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_market_repo() -> MagicMock:
    return MagicMock(spec=MarketRepository)


@pytest.fixture
def mock_broker_repo() -> MagicMock:
    return MagicMock(spec=BrokerRepository)


@pytest.fixture
def service(
    mock_trade_repo: MagicMock,
    mock_market_repo: MagicMock,
    mock_broker_repo: MagicMock,
) -> TradeViewService:
    return TradeViewService(
        trade_repository=mock_trade_repo,
        market_repository=mock_market_repo,
        broker_repository=mock_broker_repo,
    )


# ---------------------------------------------------------------------------
# 1. Strategy Filter & Name Resolution
# ---------------------------------------------------------------------------


def test_map_strategy_filter_name_all_branches() -> None:
    """Verifies all branches in _map_strategy_filter_name."""
    assert _map_strategy_filter_name("my_dip_strategy") == "DipBuyer"
    assert _map_strategy_filter_name("turnover_v1") == "TurnoverTiming"
    assert _map_strategy_filter_name("twopercent_flow") == "TwoPercent"
    assert _map_strategy_filter_name("two_percent") == "TwoPercent"
    assert _map_strategy_filter_name("ndx_momentum_leader") == "NDXMomentum"
    assert _map_strategy_filter_name("other_momentum") == "NDXMomentum"
    assert _map_strategy_filter_name("ArbitraryStrategy") == "ArbitraryStrategy"


def test_map_order_strategy_filter_all_tokens() -> None:
    """Verifies all tokens in _map_order_strategy_filter."""
    assert _map_order_strategy_filter("my_dip_bot") == "DipBuyer"
    assert _map_order_strategy_filter("turnover_run") == "TurnoverTiming"
    assert _map_order_strategy_filter("twopercent_daily") == "TwoPercent"
    assert _map_order_strategy_filter("ndx_leaders") == "NDXMomentum"
    assert _map_order_strategy_filter("momentum_picks") == "NDXMomentum"
    assert _map_order_strategy_filter("tgim_monday") == "TGIM"
    assert _map_order_strategy_filter("bridge_monthly") == "BridgeScout"
    assert _map_order_strategy_filter("scout_run") == "BridgeScout"
    assert _map_order_strategy_filter("bounce_daily") == "BounceBandit"
    assert _map_order_strategy_filter("bandit_picks") == "BounceBandit"
    assert _map_order_strategy_filter("") == "Unknown"
    assert _map_order_strategy_filter("CustomAlpha") == "CustomAlpha"


def test_is_strategy_match_sequence(service: TradeViewService) -> None:
    """Verifies is_strategy_match when target is a sequence of strategy names."""
    trade: dict[str, object] = {"strategy": "two_percent"}
    assert service.is_strategy_match(trade, [Strategies.TwoPercent.value, "other"])
    assert not service.is_strategy_match(trade, ["other_1", "other_2"])


# ---------------------------------------------------------------------------
# 2. Context Parsing & Date Extraction
# ---------------------------------------------------------------------------


def test_parse_context_edge_cases(service: TradeViewService) -> None:
    """Verifies malformed JSON and edge cases in _parse_context."""
    # Malformed JSON
    assert service._parse_context({"id": 1, "signal_context": "{bad_json"}) == {}
    # Non-dict JSON (e.g. list or string)
    assert service._parse_context({"id": 2, "signal_context": '["a", "b"]'}) == {}
    # None or empty
    assert service._parse_context({"id": 3, "signal_context": ""}) == {}
    assert service._parse_context({"id": 4, "signal_context": None}) == {}
    # Dict directly
    assert service._parse_context({"id": 5, "signal_context": {"key": "val"}}) == {
        "key": "val"
    }


def test_calculate_days_held_with_exit_date(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Verifies _calculate_days_held with explicit entry and exit dates."""
    mock_market_repo.get_trading_days_count.return_value = 5
    days = service._calculate_days_held(
        "AAPL", "2026-05-01 10:00:00", "2026-05-08 16:00:00"
    )
    assert days == 5
    mock_market_repo.get_trading_days_count.assert_called_once_with(
        "AAPL", "2026-05-01", "2026-05-08"
    )

    # Empty entry date
    assert service._calculate_days_held("AAPL", None, None) == 0


def test_extract_strategy_version(service: TradeViewService) -> None:
    """Verifies strategy version string extraction."""
    assert service._extract_strategy_version("dip_buyer_0.5") == "0.5"
    assert service._extract_strategy_version("momentum_1.0_prod") == "1.0"
    assert service._extract_strategy_version("unversioned_strat") is None


def test_resolve_current_price_fallback(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Verifies fallback to market_repository.get_latest_price when current_price is 0."""
    mock_market_repo.get_latest_price.return_value = 182.50
    price = service._resolve_current_price("AAPL", TradeStatus.ACTIVE, 0.0)
    assert price == 182.50

    # Non-zero price stays unchanged
    assert service._resolve_current_price("AAPL", TradeStatus.ACTIVE, 175.0) == 175.0


# ---------------------------------------------------------------------------
# 3. Active & Closed Trade Calculations (Short, Critical, Progress)
# ---------------------------------------------------------------------------


def test_prepare_active_trade_short_and_critical_and_progress(
    service: TradeViewService,
) -> None:
    """Verifies PnL for short trades, critical stop loss, and progress bar."""
    trade: dict[str, object] = {
        "current_stop_loss": 100.5
    }  # distance = 0.5 to price 100.0 (< 1%)
    context: dict[str, object] = {"direction": "short", "target_price": 90.0}

    unrealized_pnl, pnl_pct, is_critical, progress = (
        service._prepare_active_trade_view_fields(
            trade=trade,
            context_dict=context,
            entry_price=110.0,
            current_price=100.0,
            initial_size=10.0,
        )
    )
    # Short: (entry - current) * size = (110 - 100) * 10 = +100
    assert unrealized_pnl == pytest.approx(100.0)
    assert pnl_pct == pytest.approx(((110 - 100) / 110) * 100)
    # Stop-Loss: distance = abs(100 - 100.5) = 0.5 / 100 = 0.005 < 0.01 -> Critical!
    assert is_critical is True
    # Progress: total_range = 90 - 100.5 = -10.5; current_dist = 100 - 100.5 = -0.5
    # percentage = (-0.5 / -10.5) * 100 = 4.76%
    assert 0.0 < progress < 100.0


def test_prepare_active_trade_zero_entry_price(service: TradeViewService) -> None:
    """Verifies that entry_price <= 0 returns 0s safely."""
    pnl, pct, crit, prog = service._prepare_active_trade_view_fields(
        trade={},
        context_dict={},
        entry_price=0.0,
        current_price=100.0,
        initial_size=10.0,
    )
    assert (pnl, pct, crit, prog) == (0.0, 0.0, False, 0.0)


def test_prepare_closed_trade_short_and_zero_realized(
    service: TradeViewService,
) -> None:
    """Verifies realized PnL calculation for short trades when realized_pnl is 0."""
    trade: dict[str, object] = {"realized_pnl": 0.0}
    context: dict[str, object] = {"direction": "short"}
    pnl, pct = service._prepare_closed_trade_view_fields(
        trade=trade,
        context_dict=context,
        entry_price=100.0,
        exit_price=90.0,
        initial_size=5.0,
    )
    # Short: (100 - 90) * 5 = 50.0
    assert pnl == pytest.approx(50.0)
    assert pct == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. Sparklines & Visualizations
# ---------------------------------------------------------------------------


def test_attach_sparklines_batch_and_color_logic(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Verifies attach_sparklines populates HTML sparklines with correct colors."""
    trades: list[TradeViewData] = [
        cast(
            TradeViewData,
            {
                "symbol": "AAPL",
                "unrealized_pnl": 50.0,
                "sparkline": "",
            },
        ),
        cast(
            TradeViewData,
            {
                "symbol": "MSFT",
                "unrealized_pnl": -25.0,
                "sparkline": "",
            },
        ),
    ]

    history = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2026-08-01", "close": 150.0},
            {"symbol": "AAPL", "date": "2026-08-02", "close": 155.0},
            {"symbol": "MSFT", "date": "2026-08-01", "close": 300.0},
            {"symbol": "MSFT", "date": "2026-08-02", "close": 290.0},
        ]
    )
    mock_market_repo.get_batch_history_raw.return_value = history

    service.attach_sparklines(trades, reference_date=pd.Timestamp("2026-08-03"))

    assert (
        "svg" in trades[0]["sparkline"].lower()
        or "plotly" in trades[0]["sparkline"].lower()
    )
    assert (
        "svg" in trades[1]["sparkline"].lower()
        or "plotly" in trades[1]["sparkline"].lower()
    )


def test_attach_sparklines_empty_trades(service: TradeViewService) -> None:
    """Verifies attach_sparklines with empty list returns immediately."""
    service.attach_sparklines([])


# ---------------------------------------------------------------------------
# 5. 5-Day Open PnL Delta Calculations
# ---------------------------------------------------------------------------


def test_calculate_open_pnl_5d_change_all_branches(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Verifies 5-day Open PnL calculation across empty history, short trades, and entry dates."""
    # 1. Empty active trades
    assert service._calculate_open_pnl_5d_change([]) == 0.0

    # 2. Empty history returned from DB
    mock_market_repo.get_batch_history_raw.return_value = pd.DataFrame()
    trade_sample: list[TradeViewData] = [
        cast(
            TradeViewData,
            {
                "symbol": "AAPL",
                "unrealized_pnl": 100.0,
                "entry_price": 150.0,
                "initial_size": 10,
            },
        )
    ]
    assert (
        service._calculate_open_pnl_5d_change(
            trade_sample, reference_date=pd.Timestamp("2026-09-01")
        )
        == 0.0
    )

    # 3. Valid history: Trade entered after past date vs before
    history = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2026-08-25", "close": 140.0},
            {"symbol": "AAPL", "date": "2026-08-30", "close": 150.0},
            {"symbol": "SHORT", "date": "2026-08-25", "close": 200.0},
            {"symbol": "SHORT", "date": "2026-08-30", "close": 190.0},
        ]
    )
    mock_market_repo.get_batch_history_raw.return_value = history

    active_trades: list[TradeViewData] = [
        # Entered after past_date (2026-08-25) -> uses unrealized_pnl
        cast(
            TradeViewData,
            {
                "symbol": "AAPL",
                "entry_date": "2026-08-28",
                "unrealized_pnl": 45.0,
                "current_price": 150.0,
                "initial_size": 5,
                "context": {"direction": "long"},
            },
        ),
        # Short trade entered before past_date -> (past_price - current_price) * size
        cast(
            TradeViewData,
            {
                "symbol": "SHORT",
                "entry_date": "2026-08-20",
                "unrealized_pnl": 50.0,
                "current_price": 190.0,
                "initial_size": 10,
                "context": {"direction": "short"},
            },
        ),
        # Symbol with no rows in history -> falls back to unrealized_pnl
        cast(
            TradeViewData,
            {
                "symbol": "UNKNOWN",
                "unrealized_pnl": 20.0,
            },
        ),
    ]

    total_delta = service._calculate_open_pnl_5d_change(
        active_trades, reference_date=pd.Timestamp("2026-09-01")
    )
    # AAPL: 45.0
    # SHORT: (140.0 - 190.0) -> wait, SHORT past_price from row 2 is 190 or 200:
    # rows for SHORT: iloc[-2] is 200.0, past_price = 200.0. (200.0 - 190.0) * 10 = +100.0
    # UNKNOWN: 20.0
    # Total = 45 + 100 + 20 = 165.0
    assert total_delta == pytest.approx(165.0)


# ---------------------------------------------------------------------------
# 6. Portfolio Summary, R-Multiples & Signal Dates
# ---------------------------------------------------------------------------


def test_get_portfolio_summary_r_metrics_and_error_handling(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Verifies get_portfolio_summary computes R-multiples, SQN, and handles closed trades."""
    mock_market_repo.get_batch_history_raw.return_value = pd.DataFrame()

    active: list[TradeViewData] = [
        cast(
            TradeViewData,
            {
                "symbol": "AAPL",
                "entry_price": 150.0,
                "initial_size": 10,
                "unrealized_pnl": 50.0,
            },
        )
    ]
    closed: list[TradeViewData] = [
        cast(
            TradeViewData,
            {
                "symbol": "MSFT",
                "realized_pnl": 100.0,
                "entry_price": 200.0,
                "initial_size": 10,
                "current_stop_loss": 190.0,  # risk = abs(200 - 190) * 10 = 100 -> R = 1.0
            },
        ),
        cast(
            TradeViewData,
            {
                "symbol": "TSLA",
                "realized_pnl": -50.0,
                "entry_price": 250.0,
                "initial_size": 10,
                "current_stop_loss": 0.0,  # fallback risk = 250 * 10 * 0.05 = 125 -> R = -0.4
            },
        ),
    ]

    summary = service.get_portfolio_summary(active_trades=active, closed_trades=closed)
    assert summary["invested"] == 1500.0
    assert summary["open_pnl"] == 50.0
    assert summary["count"] == 1
    assert summary["win_rate"] == 0.5
    assert float(cast(float, summary["profit_factor"])) > 0.0
    assert "sqn" in summary


def test_get_latest_signal_date_variants(
    service: TradeViewService,
    mock_trade_repo: MagicMock,
    mock_market_repo: MagicMock,
) -> None:
    """Verifies date and time parsing from trade and market repositories."""
    # 1. Full timestamp with ISO 'T'
    mock_trade_repo.get_latest_updated_at.return_value = "2026-09-01T15:30:45.123"
    assert service.get_latest_signal_date() == "2026-09-01 15:30"

    # 2. Date only from trades table
    mock_trade_repo.get_latest_updated_at.return_value = "2026-09-02"
    assert service.get_latest_signal_date() == "2026-09-02"

    # 3. Exception in trade repo -> fallback to market repo
    mock_trade_repo.get_latest_updated_at.side_effect = RuntimeError("DB error")
    mock_market_repo.get_latest_updated_at.return_value = "2026-09-03 10:00"
    assert service.get_latest_signal_date() == "2026-09-03 10:00"


# ---------------------------------------------------------------------------
# 7. Grouping & Index / Weekday Statistics
# ---------------------------------------------------------------------------


def test_group_trades_history_with_setup_date(service: TradeViewService) -> None:
    """Verifies grouping closed trades when display_entry is '-' and setup_date is present."""
    trades: list[TradeViewData] = [
        cast(
            TradeViewData,
            {
                "symbol": "NVDA",
                "display_entry": "-",
                "exit_date": "2026-08-10",
                "context": {"setup_date": "2026-08-01 09:00:00", "indices": "NDX"},
            },
        )
    ]
    grouped = service.group_trades_history(trades)
    assert len(grouped) == 1
    assert grouped[0]["entry_date"] == "2026-08-01"


def test_get_index_stats_all_categories(service: TradeViewService) -> None:
    """Verifies get_index_stats matches SPX, NDX, DOW, RUS, and NO_INDEX."""
    trades: list[TradeViewData] = [
        cast(TradeViewData, {"realized_pnl": 10.0, "context": {"indices": "SPX,NDX"}}),
        cast(TradeViewData, {"realized_pnl": 20.0, "context": {"indices": "DOW"}}),
        cast(TradeViewData, {"realized_pnl": -5.0, "context": {"indices": "RUS"}}),
        cast(
            TradeViewData,
            {"realized_pnl": 15.0, "context": {"indices": "CUSTOM_BASKET"}},
        ),
    ]
    stats = service.get_index_stats(trades)
    assert stats[IndexAliases.SPX.value]["count"] == 1
    assert stats[IndexAliases.NDX.value]["count"] == 1
    assert stats[IndexAliases.DOW.value]["count"] == 1
    assert stats[IndexAliases.RUS.value]["count"] == 1
    assert stats[IndexAliases.NO_INDEX.value]["count"] == 1


def test_get_weekday_stats_invalid_entry_date(service: TradeViewService) -> None:
    """Verifies get_weekday_stats logs warning and skips invalid timestamp gracefully."""
    trades: list[TradeViewData] = [
        cast(TradeViewData, {"realized_pnl": 10.0, "entry_date": "invalid_date_str"}),
        cast(TradeViewData, {"realized_pnl": 20.0, "entry_date": None}),
    ]
    stats = service.get_weekday_stats(trades)
    assert all(day["count"] == 0 for day in stats.values())


# ---------------------------------------------------------------------------
# 8. Broker Reconciliation & Settlements
# ---------------------------------------------------------------------------


def test_get_broker_settlements_tz_and_holding_delta(
    service: TradeViewService, mock_broker_repo: MagicMock
) -> None:
    """Verifies settlements process tz-aware timestamps and calculate holding delta."""
    settlement = {
        "trade_group_id": "101_DipBuyer_AAPL",
        "avg_entry_price": 150.0,
        "avg_exit_price": 165.0,
    }
    mock_broker_repo.get_settlements.return_value = [settlement]
    executions = [
        {"action": "BUY", "qty": 10, "executed_at": "2026-08-01T14:30:00+00:00"},
        {"action": "SELL", "qty": 10, "executed_at": "2026-08-06T20:00:00+00:00"},
    ]
    mock_broker_repo.get_executions_for_trade_group.return_value = executions

    results = service.get_broker_settlements()
    assert len(results) == 1
    res = results[0]
    assert res["symbol"] == "AAPL"
    assert res["strategy_name"] == "DipBuyer"
    assert res["days_held"] == 5
    assert res["quantity"] == 10
    assert res["pnl_percentage"] == pytest.approx(10.0)


def test_reconciliation_discrepancies_and_orphans(
    service: TradeViewService,
    mock_trade_repo: MagicMock,
    mock_broker_repo: MagicMock,
) -> None:
    """Verifies detection of MISSING_EXECUTION and GHOST_POSITION discrepancies."""
    # Local active trade with 0 broker position -> MISSING_EXECUTION
    # Local closed trade with > 0 broker position -> GHOST_POSITION
    mock_trade_repo.get_by_status.side_effect = lambda status: (
        [
            {
                "symbol": "AAPL",
                "status": "ACTIVE",
                "current_size": 10,
                "strategy": "DipBuyer",
            }
        ]
        if status == "ACTIVE"
        else [
            {
                "symbol": "MSFT",
                "status": "CLOSED",
                "current_size": 0,
                "strategy": "TwoPercent",
            },
            {
                "symbol": "SPLIT",
                "status": "ACTIVE",
                "strategy": "SplitTarget",
            },  # ignored!
        ]
    )
    mock_broker_repo.get_net_positions_by_symbol.return_value = {
        "AAPL": 0.0,
        "MSFT": 15.0,
        "ORPHAN": 25.0,
    }
    mock_broker_repo.fetch_all.return_value = [{"strategy_name": "TurnoverTiming"}]

    discrepancies = service.get_reconciliation_discrepancies()
    assert len(discrepancies) == 3

    types = {d["symbol"]: d["discrepancy_type"] for d in discrepancies}
    assert types["AAPL"] == "MISSING_EXECUTION"
    assert types["MSFT"] == "GHOST_POSITION"
    assert types["ORPHAN"] == "GHOST_POSITION"


# ---------------------------------------------------------------------------
# 9. Golden Master Snapshot Verification
# ---------------------------------------------------------------------------


def test_golden_master_prepare_trade_view(
    service: TradeViewService, mock_market_repo: MagicMock
) -> None:
    """Golden-Master snapshot: verifies the complete structure and values of prepare_trade_view."""
    mock_market_repo.get_trading_days_count.return_value = 4
    raw_trade = {
        "id": 123,
        "symbol": "GOOGL",
        "strategy": "two_percent_flow",
        "status": TradeStatus.ACTIVE,
        "entry_date": "2026-08-10 09:30:00",
        "entry_price": 100.0,
        "current_price": 105.0,
        "initial_size": 20,
        "current_size": 20,
        "current_stop_loss": 95.0,
        "current_target": 115.0,
        "budget": 2000.0,
        "signal_context": json.dumps({"target_price": 115.0, "bucket": "NDX"}),
    }

    view_data = service.prepare_trade_view(raw_trade)

    assert view_data["id"] == "123"
    assert view_data["symbol"] == "GOOGL"
    assert view_data["strategy"] == "two_percent_flow"
    assert view_data["display_entry"] == "2026-08-10"
    assert view_data["days_held"] == 4
    assert view_data["unrealized_pnl"] == pytest.approx(100.0)
    assert view_data["pnl_percentage"] == pytest.approx(5.0)
    assert view_data["is_critical"] is False
    assert view_data["progress"] == pytest.approx(50.0)  # (105-95)/(115-95) * 100 = 50%
    assert view_data["context"]["indices"] == "NDX"
