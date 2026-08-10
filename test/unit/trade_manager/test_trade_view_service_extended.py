"""Extended unit tests for TradeViewService in app/services/trade_manager/view_service.py."""

from unittest.mock import MagicMock

from app.services.trade_manager.view_service import (
    TradeViewData,
    TradeViewService,
)


def _make_dummy_trade_view_data(
    symbol: str = "AAPL",
    strategy: str = "DipBuyer",
    unrealized_pnl: float = 100.0,
    initial_size: int = 10,
    entry_price: float = 150.0,
    exit_date: str = "2026-01-10",
    display_entry: str = "2026-01-01",
) -> TradeViewData:
    return {
        "id": "1",
        "symbol": symbol,
        "strategy": strategy,
        "status": "ACTIVE",
        "initial_size": initial_size,
        "current_size": 10,
        "entry_price": entry_price,
        "entry_date": "2026-01-01T10:00:00",
        "exit_price": None,
        "exit_date": exit_date,
        "exit_reason": None,
        "signal_context": None,
        "display_entry": display_entry,
        "display_exit": exit_date,
        "days_held": 9,
        "unrealized_pnl": unrealized_pnl,
        "pnl_percentage": 6.67,
        "is_critical": False,
        "progress": 50.0,
        "display_size": 10.0,
        "sparkline": "",
        "max_days": 30,
        "version": "1.0",
        "context": {"setup_date": "2026-01-01 10:00:00", "indices": "SPY"},
    }


def test_sparkline_and_donut_generators() -> None:
    service = TradeViewService(
        trade_repository=MagicMock(), market_repository=MagicMock()
    )

    # Sparkline positive & negative
    spark_pos = service.generate_sparkline(
        ["2026-01-01", "2026-01-02"], [100.0, 105.0], is_positive=True
    )
    assert "<svg" in spark_pos or "plotly" in spark_pos.lower() or "div" in spark_pos

    spark_neg = service.generate_sparkline(
        ["2026-01-01", "2026-01-02"], [100.0, 95.0], is_positive=False
    )
    assert len(spark_neg) > 0

    # Donut chart
    donut = service.generate_donut_chart(["A", "B"], [60.0, 40.0], ["green", "red"])
    assert len(donut) > 0


def test_group_trades_by_symbol_and_history() -> None:
    service = TradeViewService(
        trade_repository=MagicMock(), market_repository=MagicMock()
    )

    t1 = _make_dummy_trade_view_data(symbol="AAPL", unrealized_pnl=50.0)
    t2 = _make_dummy_trade_view_data(symbol="AAPL", unrealized_pnl=30.0)
    t3 = _make_dummy_trade_view_data(symbol="MSFT", unrealized_pnl=-20.0)

    # Group by symbol
    grouped = service.group_trades_by_symbol([t1, t2, t3])
    assert len(grouped) == 2
    aapl_group = next(g for g in grouped if g["symbol"] == "AAPL")
    assert aapl_group["total_pnl"] == 80.0
    assert float(str(aapl_group["total_pnl_percentage"])) > 0

    # Group trades history
    t_hist1 = _make_dummy_trade_view_data(
        symbol="AAPL", display_entry="-", exit_date="2026-01-15"
    )
    t_hist2 = _make_dummy_trade_view_data(
        symbol="AAPL", display_entry="-", exit_date="2026-01-20"
    )
    grouped_hist = service.group_trades_history([t_hist1, t_hist2])
    assert len(grouped_hist) == 1
    assert grouped_hist[0]["max_exit"] == "2026-01-20"


def test_get_broker_settlements() -> None:
    broker_repo = MagicMock()
    broker_repo.get_settlements.return_value = [
        {
            "trade_group_id": "1_DipBuyer_AAPL",
            "avg_entry_price": 150.0,
            "avg_exit_price": 160.0,
        },
        {
            "trade_group_id": "2_TurnoverTiming_MSFT",
            "avg_entry_price": 300.0,
            "avg_exit_price": 310.0,
        },
    ]
    broker_repo.get_executions_for_trade_group.side_effect = lambda gid: (
        [
            {
                "executed_at": "2026-01-01 09:30:00",
                "qty": 10.0,
                "action": "BUY",
            },
            {
                "executed_at": "2026-01-05 16:00:00",
                "qty": 10.0,
                "action": "SELL",
            },
        ]
        if "1_" in gid
        else []
    )

    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=MagicMock(),
        broker_repository=broker_repo,
    )

    settlements = service.get_broker_settlements()
    assert len(settlements) == 2
    assert settlements[0]["strategy_filter"] == "DipBuyer"
    assert settlements[0]["days_held"] == 4
    assert settlements[0]["quantity"] == 10.0
    assert settlements[1]["strategy_filter"] == "TurnoverTiming"

    # None broker repo path
    service_no_broker = TradeViewService(
        trade_repository=MagicMock(), market_repository=MagicMock()
    )
    assert service_no_broker.get_broker_settlements() == []


def test_get_reconciliation_discrepancies_ghost_position() -> None:
    trade_repo = MagicMock()
    trade_repo.get_by_status.side_effect = lambda st: (
        []
        if st == "ACTIVE"
        else [{"symbol": "GOOG", "strategy": "DipBuyer", "status": "CLOSED"}]
    )

    broker_repo = MagicMock()
    broker_repo.get_net_positions_by_symbol.return_value = {"GOOG": 50.0}

    service = TradeViewService(
        trade_repository=trade_repo,
        market_repository=MagicMock(),
        broker_repository=broker_repo,
    )

    discrepancies = service.get_reconciliation_discrepancies()
    assert len(discrepancies) == 1
    assert discrepancies[0]["discrepancy_type"] == "GHOST_POSITION"
    assert discrepancies[0]["symbol"] == "GOOG"
    assert discrepancies[0]["strategy"] == "DipBuyer"


def test_get_broker_active_trades() -> None:
    broker_repo = MagicMock()
    broker_repo.get_active_positions.return_value = [
        {
            "symbol": "AAPL",
            "entry_price": 150.0,
            "current_price": 155.0,
            "current_size": 10.0,
            "entry_date": "2026-01-01",
            "strategy": "DipBuyer",
        }
    ]

    market_repo = MagicMock()
    market_repo.get_latest_price.return_value = 160.0

    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=market_repo,
        broker_repository=broker_repo,
    )

    active_trades = service.get_broker_active_trades()
    assert len(active_trades) == 1
    assert active_trades[0]["current_price"] == 160.0
    assert active_trades[0]["unrealized_pnl"] == (160.0 - 150.0) * 10.0
    assert active_trades[0]["strategy_filter"] == "DipBuyer"

    # None broker repo path
    service_no_broker = TradeViewService(
        trade_repository=MagicMock(), market_repository=MagicMock()
    )
    assert service_no_broker.get_broker_active_trades() == []
