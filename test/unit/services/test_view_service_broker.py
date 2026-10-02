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
        },
        {
            "id": 985,
            "symbol": "MES",
            "strategy": "TGIM",
            "entry_date": "2026-07-01",
            "current_size": 1.0,
            "entry_price": 5800.0,
            "current_price": 5800.0,
            "tws_status": "Filled",
            "tws_orders": [],
        },
    ]
    market_repo_mock = MagicMock()
    market_repo_mock.get_latest_price.side_effect = lambda sym: (
        110.0 if sym == "INTC" else 5850.0
    )

    service = TradeViewService(
        trade_repository=MagicMock(),
        market_repository=market_repo_mock,
        broker_repository=broker_repo_mock,
    )

    active_trades = service.get_broker_active_trades()
    assert len(active_trades) == 2
    assert active_trades[0]["symbol"] == "INTC"
    assert active_trades[0]["unrealized_pnl"] == 100.0
    assert active_trades[0]["pnl_percentage"] == 10.0

    # MES: Multiplier 5.0 applied -> (5850 - 5800) * 1 * 5 = 250.0
    assert active_trades[1]["symbol"] == "MES"
    assert active_trades[1]["unrealized_pnl"] == 250.0


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
            "current_price": 165.0,
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
    assert dip_item.market_value == Decimal("2150.00")
    assert dip_item.unrealized_pnl == Decimal("150.00")
    assert dip_item.pnl_percentage == 7.5
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


def test_calculate_capital_allocation_futures_margin_and_multiplier() -> None:
    """Verifies that futures use initial margin for allocation weighting,

    multiplier-adjusted PnL, and compute notional exposure properly.
    """
    positions = [
        {
            "symbol": "MES",
            "strategy": "TGIM",
            "current_size": 1.0,
            "entry_price": 5800.0,
            "current_price": 5850.0,
        },
        {
            "symbol": "MNQU2026",
            "strategy": "BounceBandit",
            "current_size": 2.0,
            "entry_price": 20000.0,
            "current_price": 19950.0,
        },
        {
            "symbol": "AAPL",
            "strategy": "DipBuyer",
            "current_size": 10.0,
            "entry_price": 150.0,
            "current_price": 160.0,
        },
    ]

    summary = calculate_capital_allocation(positions)
    # MES: 1 * 1600 = 1600 margin.
    # MNQ: 2 * 2200 = 4400 margin.
    # AAPL: 10 * 150 = 1500 invested equity.
    # Total portfolio capital = 1600 + 4400 + 1500 = 7500.
    assert summary.total_invested == Decimal("7500.00")
    assert summary.total_positions == 3
    assert len(summary.strategies) == 3

    bounce_item = next(
        s for s in summary.strategies if s.strategy_key == "BounceBandit"
    )
    tgim_item = next(s for s in summary.strategies if s.strategy_key == "TGIM")
    dip_item = next(s for s in summary.strategies if s.strategy_key == "DipBuyer")

    # BounceBandit (MNQ): 2 contracts, margin = 4400, notional = 2 * 19950 * 2 = 79800
    assert bounce_item.is_derivative is True
    assert bounce_item.invested_capital == Decimal("4400.00")
    assert bounce_item.notional_exposure == Decimal("79800.00")
    # PnL: (19950 - 20000) * 2 * 2 = -200.00
    assert bounce_item.unrealized_pnl == Decimal("-200.00")
    assert round(bounce_item.pnl_percentage, 2) == -4.55
    assert bounce_item.color_class == "bg-violet-500"
    assert bounce_item.strategy_label == "Bounce Bandit"

    # TGIM (MES): 1 contract, margin = 1600, notional = 1 * 5850 * 5 = 29250
    assert tgim_item.is_derivative is True
    assert tgim_item.invested_capital == Decimal("1600.00")
    assert tgim_item.notional_exposure == Decimal("29250.00")
    # PnL: (5850 - 5800) * 1 * 5 = +250.00
    assert tgim_item.unrealized_pnl == Decimal("250.00")
    assert round(tgim_item.pnl_percentage, 2) == 15.62
    assert tgim_item.color_class == "bg-sky-500"
    assert tgim_item.strategy_label == "TGIM"

    # DipBuyer (AAPL): stock, notional == market_value = 1600, invested = 1500
    assert dip_item.is_derivative is False
    assert dip_item.invested_capital == Decimal("1500.00")
    assert dip_item.market_value == Decimal("1600.00")
    assert dip_item.unrealized_pnl == Decimal("100.00")
    assert round(dip_item.pnl_percentage, 2) == 6.67
    assert dip_item.color_class == "bg-indigo-500"


def test_calculate_capital_allocation_with_account_metrics_free_cash() -> None:
    """Verifies that free cash from account metrics is correctly integrated,

    normalizing allocation percentages against total capital (invested + free cash).
    """
    positions = [
        {
            "symbol": "AAPL",
            "strategy": "DipBuyer",
            "current_size": 10.0,
            "entry_price": 100.0,
            "current_price": 110.0,
        },
        {
            "symbol": "QQQ",
            "strategy": "TurnoverTiming",
            "current_size": 30.0,
            "entry_price": 100.0,
            "current_price": 100.0,
        },
    ]
    account_metrics = {
        "account_id": "U12345",
        "net_liquidation": 12000.0,
        "total_cash_value": 6000.0,
        "available_funds": 8000.0,
        "maint_margin_req": 2000.0,
        "cushion_pct": 83.33,
        "buying_power": 32000.0,
        "updated_at": "2026-10-01 09:00:00",
    }

    summary = calculate_capital_allocation(positions, account_metrics=account_metrics)

    # Positions: AAPL 1000 + QQQ 3000 = 4000 invested.
    assert summary.total_invested == Decimal("4000.00")
    assert summary.free_cash == Decimal("6000.00")
    # Total capital = 4000 + 6000 = 10000.
    assert summary.total_capital == Decimal("10000.00")
    assert summary.net_liquidation == Decimal("12000.00")
    assert summary.available_funds == Decimal("8000.00")
    assert summary.buying_power == Decimal("32000.00")
    assert summary.cushion_pct == 83.33
    assert summary.metrics_updated_at == "2026-10-01 09:00:00"

    # Strategies list includes TurnoverTiming, DipBuyer, and FreeCash
    assert len(summary.strategies) == 3
    turnover_item = next(
        s for s in summary.strategies if s.strategy_key == "TurnoverTiming"
    )
    dip_item = next(s for s in summary.strategies if s.strategy_key == "DipBuyer")
    cash_item = next(s for s in summary.strategies if s.strategy_key == "FreeCash")

    assert turnover_item.invested_capital == Decimal("3000.00")
    assert turnover_item.allocation_percentage == 30.0  # 3000 / 10000

    assert dip_item.invested_capital == Decimal("1000.00")
    assert dip_item.allocation_percentage == 10.0  # 1000 / 10000

    assert cash_item.invested_capital == Decimal("6000.00")
    assert cash_item.allocation_percentage == 60.0  # 6000 / 10000
    assert cash_item.strategy_label == "Free Cash"
    assert cash_item.color_class == "bg-emerald-500"
    assert cash_item.is_cash is True

    # Total allocations sum to exactly 100%
    total_pct = sum(s.allocation_percentage for s in summary.strategies)
    assert round(total_pct, 1) == 100.0

    # Test convenience properties for UX badge and tooltip
    assert summary.invested_percentage == 40.0
    assert summary.cash_percentage == 60.0
    assert summary.total_market_value == Decimal("4100.00")
    assert summary.total_unrealized_pnl == Decimal("100.00")


def test_calculate_capital_allocation_empty_positions_with_free_cash() -> None:
    """Verifies that an account with 0 positions but positive cash returns Free Cash at 100%."""
    account_metrics = {
        "account_id": "U12345",
        "net_liquidation": 5000.0,
        "total_cash_value": 5000.0,
        "available_funds": 5000.0,
    }
    summary = calculate_capital_allocation([], account_metrics=account_metrics)
    assert summary.total_invested == Decimal("0.00")
    assert summary.free_cash == Decimal("5000.00")
    assert summary.total_capital == Decimal("5000.00")
    assert len(summary.strategies) == 1
    assert summary.strategies[0].strategy_key == "FreeCash"
    assert summary.strategies[0].allocation_percentage == 100.0
    assert summary.strategies[0].is_cash is True
