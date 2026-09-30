from decimal import Decimal
from unittest.mock import MagicMock

from app.services.trade_manager.view_service import (
    TradeViewService,
    calculate_capital_allocation,
)


def test_get_broker_summary_none() -> None:
    """Verifies that view service returns fallback empty/list structures if broker_repository is None."""
    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=MagicMock(),
        broker_repository=None,
    )
    assert service.get_broker_summary() == {}
    assert service.get_broker_settlements() == []
    assert service.get_reconciliation_discrepancies() == []


def test_get_broker_summary_calculation() -> None:
    """Verifies strategy metric accumulation and formatting with simulated broker database results."""
    broker_repo_mock = MagicMock()
    broker_repo_mock.get_settlements.return_value = [
        {
            "trade_group_id": "1_DipBuyer_XYZ",
            "net_pnl": 100.0,
            "price_diff_slippage": 0.5,
            "total_commissions": 2.0,
        },
        {
            "trade_group_id": "2_TurnoverTiming_ABC",
            "net_pnl": -50.0,
            "price_diff_slippage": -0.1,
            "total_commissions": 4.0,
        },
    ]
    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=MagicMock(),
        broker_repository=broker_repo_mock,
    )

    summary = service.get_broker_summary()
    assert "all" in summary
    assert "DipBuyer" in summary
    assert "TurnoverTiming" in summary

    assert summary["all"]["pnl"] == 50.0
    assert summary["all"]["fees"] == 6.0
    assert summary["all"]["winrate"] == "50.0%"
    assert summary["all"]["pnlText"] == "+50"

    assert summary["DipBuyer"]["pnl"] == 100.0
    assert summary["DipBuyer"]["winrate"] == "100.0%"

    assert summary["TurnoverTiming"]["pnl"] == -50.0
    assert summary["TurnoverTiming"]["winrate"] == "0.0%"


def test_get_broker_active_trades_calculation() -> None:
    """Verifies retrieval, mapping, and price resolution of active positions."""
    broker_repo_mock = MagicMock()
    broker_repo_mock.get_active_positions.return_value = [
        {
            "id": 984,
            "symbol": "INTC",
            "strategy": "NDXMomentum",
            "entry_date": "2026-07-01",
            "current_size": 10.0,
            "entry_price": 100.0,
            "current_price": 100.0,
            "tws_status": "Filled",
            "tws_orders": [],
        }
    ]
    market_repo_mock = MagicMock()
    market_repo_mock.get_latest_price.return_value = 110.0

    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=market_repo_mock,
        broker_repository=broker_repo_mock,
    )

    active_trades = service.get_broker_active_trades()
    assert len(active_trades) == 1
    assert active_trades[0]["symbol"] == "INTC"
    assert active_trades[0]["strategy_filter"] == "NDXMomentum"
    assert active_trades[0]["current_price"] == 110.0
    assert active_trades[0]["unrealized_pnl"] == 100.0
    assert active_trades[0]["pnl_percentage"] == 10.0


def test_get_reconciliation_discrepancies() -> None:
    """Verifies that reconciliation discrepancy detection attaches strategy badges."""
    trade_repo_mock = MagicMock()
    trade_repo_mock.get_by_status.side_effect = [
        [
            {
                "symbol": "TSLA",
                "strategy": "DipBuyer",
                "status": "ACTIVE",
                "current_size": 10.0,
            }
        ],
        [],
    ]
    broker_repo_mock = MagicMock()
    broker_repo_mock.get_net_positions_by_symbol.return_value = {"TSLA": 0.0}

    service = TradeViewService(
        trade_repository=trade_repo_mock,
        market_repository=MagicMock(),
        broker_repository=broker_repo_mock,
    )

    discrepancies = service.get_reconciliation_discrepancies()
    assert len(discrepancies) == 1
    assert discrepancies[0]["symbol"] == "TSLA"
    assert discrepancies[0]["strategy"] == "DipBuyer"
    assert discrepancies[0]["discrepancy_type"] == "MISSING_EXECUTION"


def test_calculate_capital_allocation_empty() -> None:
    """Verifies that an empty positions list produces zero allocation summary."""
    summary = calculate_capital_allocation([])
    assert summary.total_invested == Decimal("0.00")
    assert summary.total_positions == 0
    assert summary.strategies == ()


def test_calculate_capital_allocation_multiple_strategies() -> None:
    """Verifies proportional allocation, ordering, position count, and colors."""
    positions = [
        {
            "symbol": "AAPL",
            "strategy": "DipBuyer",
            "current_size": 10.0,
            "entry_price": 150.0,
        },
        {
            "symbol": "MSFT",
            "strategy_filter": "DipBuyer",
            "current_size": 5.0,
            "entry_price": 100.0,
        },
        {
            "symbol": "QQQ",
            "strategy": "TurnoverTiming",
            "current_size": 20.0,
            "entry_price": 100.0,
        },
        {
            "symbol": "SPY",
            "strategy": "TwoPercent",
            "current_size": 10.0,
            "entry_price": 100.0,
        },
    ]

    summary = calculate_capital_allocation(positions)
    assert summary.total_positions == 4
    # AAPL: 1500 + MSFT: 500 = 2000. QQQ: 2000. SPY: 1000. Total = 5000.
    assert summary.total_invested == Decimal("5000.00")
    assert len(summary.strategies) == 3

    dip_item = next(s for s in summary.strategies if s.strategy_key == "DipBuyer")
    turnover_item = next(
        s for s in summary.strategies if s.strategy_key == "TurnoverTiming"
    )
    two_pct_item = next(s for s in summary.strategies if s.strategy_key == "TwoPercent")

    assert dip_item.invested_capital == Decimal("2000.00")
    assert dip_item.position_count == 2
    assert dip_item.allocation_percentage == 40.0
    assert dip_item.color_class == "bg-indigo-500"
    assert dip_item.strategy_label == "Dip Buyer"

    assert turnover_item.invested_capital == Decimal("2000.00")
    assert turnover_item.position_count == 1
    assert turnover_item.allocation_percentage == 40.0
    assert turnover_item.color_class == "bg-amber-500"
    assert turnover_item.strategy_label == "Turnover"

    assert two_pct_item.invested_capital == Decimal("1000.00")
    assert two_pct_item.position_count == 1
    assert two_pct_item.allocation_percentage == 20.0
    assert two_pct_item.color_class == "bg-purple-500"
    assert two_pct_item.strategy_label == "Two Percent"


def test_get_broker_capital_allocation_service_method() -> None:
    """Verifies that TradeViewService delegates correctly to calculate_capital_allocation."""
    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=MagicMock(),
        broker_repository=MagicMock(),
    )
    positions = [
        {
            "symbol": "INTC",
            "strategy": "NDXMomentum",
            "current_size": 10.0,
            "entry_price": 50.0,
        }
    ]
    summary = service.get_broker_capital_allocation(positions)
    assert summary.total_invested == Decimal("500.00")
    assert summary.total_positions == 1
    assert len(summary.strategies) == 1
    assert summary.strategies[0].strategy_key == "NDXMomentum"
    assert summary.strategies[0].strategy_label == "NDX Momentum"
    assert summary.strategies[0].color_class == "bg-rose-500"
