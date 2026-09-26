"""Unit tests for Backtest vs. Broker Reality Check domain logic and service methods."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from app.services.trade_manager.reality_check import (
    BrokerCostBreakdown,
    HistoryRealityCheck,
    PositionRealityCheck,
    compute_broker_cost_breakdown,
    compute_dual_equity_curve,
    compute_reality_check_summary,
    is_future_strategy,
    match_active_positions,
    match_closed_history,
    parse_trade_id_from_trade_group_id,
    resolve_strategy_meta,
    to_decimal,
)
from app.services.trade_manager.view_service import TradeViewService


class TestParsingAndClassification:
    """Tests ID parsing and future strategy classification."""

    def test_parse_trade_id_valid(self) -> None:
        assert parse_trade_id_from_trade_group_id("1110_DipBuyer_SNDK") == 1110
        assert parse_trade_id_from_trade_group_id("769_TurnoverTiming_1.0_TSLA") == 769
        assert parse_trade_id_from_trade_group_id("1163_TwoPercent_SXRV.DE") == 1163
        assert parse_trade_id_from_trade_group_id("984_NDXMomentum_INTC") == 984

    def test_parse_trade_id_invalid_or_none(self) -> None:
        assert parse_trade_id_from_trade_group_id(None) is None
        assert parse_trade_id_from_trade_group_id("") is None
        assert parse_trade_id_from_trade_group_id("INVALID_PREFIX_TEST") is None

    def test_is_future_strategy(self) -> None:
        assert is_future_strategy("TGIM") is True
        assert is_future_strategy("tgim") is True
        assert is_future_strategy("BounceBandit") is True
        assert is_future_strategy("bounce_bandit") is True
        assert is_future_strategy("BridgeScout") is True
        assert is_future_strategy("bridge_scout") is True

        assert is_future_strategy("DipBuyer") is False
        assert is_future_strategy("TurnoverTiming") is False
        assert is_future_strategy("TwoPercent") is False
        assert is_future_strategy("NDXMomentum") is False

    def test_resolve_strategy_meta(self) -> None:
        name, badge = resolve_strategy_meta("DipBuyer")
        assert name == "Dip Buyer"
        assert "bg-indigo-50" in badge

        name, badge = resolve_strategy_meta("TURNOVERTIMING_0.5")
        assert name == "Turnover 0.5"
        assert "bg-amber-50" in badge

        name, badge = resolve_strategy_meta("TurnoverTiming_1.0")
        assert name == "Turnover 1.0"
        assert "bg-amber-50" in badge

        name, badge = resolve_strategy_meta("TwoPercent")
        assert name == "Two Percent"
        assert "bg-purple-50" in badge

        name, badge = resolve_strategy_meta("NDXMomentum")
        assert name == "NDX Momentum"
        assert "bg-rose-50" in badge

        name, badge = resolve_strategy_meta("TGIM")
        assert name == "TGIM"
        assert "bg-sky-50" in badge

        name, badge = resolve_strategy_meta("BridgeScout")
        assert name == "Bridge Scout"
        assert "bg-teal-50" in badge

        name, badge = resolve_strategy_meta("BounceBandit")
        assert name == "Bounce Bandit"
        assert "bg-violet-50" in badge

    def test_to_decimal_safety(self) -> None:
        assert to_decimal(12.345) == Decimal("12.35")
        assert to_decimal("100.50") == Decimal("100.50")
        assert to_decimal(None) == Decimal("0.00")
        assert to_decimal("invalid", fallback="5.00") == Decimal("5.00")


class TestMatchActivePositions:
    """Tests matching of active backtest trades with broker active positions."""

    def test_match_successful_and_unmatched_ignored(self) -> None:
        signals_active = [
            {
                "id": 100,
                "symbol": "INTC",
                "strategy": "NDXMomentum",
                "entry_price": 135.0,
                "current_price": 130.0,
                "current_size": 74.0,
                "unrealized_pnl": -370.0,
                "entry_date": "2026-07-01",
                "days_held": 10,
            },
            {
                "id": 999,  # Only in backtest, should be ignored
                "symbol": "AAPL",
                "strategy": "DipBuyer",
                "entry_price": 200.0,
                "current_size": 10.0,
            },
        ]
        broker_positions = [
            {
                "id": 100,
                "trade_group_id": "100_NDXMomentum_INTC",
                "symbol": "INTC",
                "strategy": "NDXMomentum",
                "strategy_filter": "NDXMomentum",
                "entry_price": 134.63,
                "current_price": 130.0,
                "current_size": 71.0,
                "unrealized_pnl": -328.73,
                "entry_date": "2026-07-01",
                "days_held": 10,
                "tws_status": "Filled",
            },
            {
                "id": 888,  # Only in broker, should be ignored
                "trade_group_id": "888_DipBuyer_XYZ",
                "symbol": "XYZ",
                "current_size": 5.0,
            },
        ]

        result = match_active_positions(signals_active, broker_positions)
        assert len(result) == 1
        pos = result[0]
        assert pos.trade_id == 100
        assert pos.symbol == "INTC"
        assert pos.quantity_bt == 74.0
        assert pos.quantity_broker == 71.0
        assert pos.entry_price_bt == Decimal("135.00")
        assert pos.entry_price_broker == Decimal("134.63")
        # Entry slippage = broker - bt = 134.63 - 135.00 = -0.37
        assert pos.entry_slippage == Decimal("-0.37")
        assert pos.slippage_cost == Decimal("-0.37") * Decimal("71.0")
        assert pos.has_quantity_diff is True
        assert pos.has_entry_price_diff is True
        assert pos.has_pnl_diff is True
        assert pos.strategy_display == "NDX Momentum"
        assert "bg-rose-50" in pos.strategy_badge_class

    def test_filter_by_strategy(self) -> None:
        signals_active = [
            {"id": 1, "symbol": "A", "strategy": "DipBuyer", "entry_price": 10.0},
            {"id": 2, "symbol": "B", "strategy": "TwoPercent", "entry_price": 20.0},
        ]
        broker_positions = [
            {
                "id": 1,
                "trade_group_id": "1_DipBuyer_A",
                "strategy": "DipBuyer",
                "strategy_filter": "DipBuyer",
            },
            {
                "id": 2,
                "trade_group_id": "2_TwoPercent_B",
                "strategy": "TwoPercent",
                "strategy_filter": "TwoPercent",
            },
        ]

        dip_result = match_active_positions(
            signals_active, broker_positions, strategy_filter="DipBuyer"
        )
        assert len(dip_result) == 1
        assert dip_result[0].trade_id == 1


class TestMatchClosedHistory:
    """Tests matching of closed backtest trades with broker settlements."""

    def test_match_history_equity_and_futures(self) -> None:
        signals_closed = [
            {
                "id": 1110,
                "symbol": "SNDK",
                "strategy": "DipBuyer",
                "entry_price": 1521.53,
                "exit_price": 1613.57,
                "initial_size": 1.0,
                "realized_pnl": 92.04,
                "exit_date": "2026-09-17",
            },
            {
                "id": 1142,
                "symbol": "SPY",
                "strategy": "TGIM",
                "entry_price": 742.09,
                "exit_price": 748.28,
                "initial_size": 13.0,
                "realized_pnl": 80.47,
                "exit_date": "2026-07-21",
            },
        ]
        broker_settlements = [
            {
                "trade_group_id": "1110_DipBuyer_SNDK",
                "symbol": "SNDK",
                "strategy_name": "DipBuyer",
                "strategy_filter": "DipBuyer",
                "avg_entry_price": 1522.75,
                "avg_exit_price": 1613.57,
                "quantity": 1.0,
                "net_pnl": 88.78,
                "total_commissions": 2.04,
                "settled_at": "2026-09-17 14:00:00",
                "executions": [],
            },
            {
                "trade_group_id": "1142_TGIM_MES",
                "symbol": "MES",
                "strategy_name": "TGIM",
                "strategy_filter": "TGIM",
                "avg_entry_price": 5565.0,
                "avg_exit_price": 5611.0,
                "quantity": 1.0,
                "net_pnl": 95.50,
                "total_commissions": 1.24,
                "price_diff_slippage": 0.50,
                "settled_at": "2026-07-21 15:00:00",
                "executions": [],
            },
        ]

        result = match_closed_history(signals_closed, broker_settlements)
        assert len(result) == 2

        # Sorted by date desc: SNDK (09.17) then TGIM (07.21)
        sndk_trade = result[0]
        assert sndk_trade.trade_id == 1110
        assert sndk_trade.unit_label_bt == "Stk"
        assert sndk_trade.unit_label_broker == "Stk"
        assert sndk_trade.entry_slippage == Decimal("1.22")  # 1522.75 - 1521.53
        assert sndk_trade.exit_slippage == Decimal("0.00")
        assert sndk_trade.commissions == Decimal("2.04")
        assert sndk_trade.strategy_display == "Dip Buyer"
        assert "bg-indigo-50" in sndk_trade.strategy_badge_class
        assert sndk_trade.has_quantity_diff is False
        assert sndk_trade.has_entry_price_diff is True
        assert sndk_trade.has_exit_price_diff is False
        assert sndk_trade.has_pnl_diff is True

        tgim_trade = result[1]
        assert tgim_trade.trade_id == 1142
        assert tgim_trade.unit_label_bt == "Stk"
        assert tgim_trade.unit_label_broker == "Ktr."
        assert tgim_trade.quantity_bt == 13.0
        assert tgim_trade.quantity_broker == 1.0
        assert tgim_trade.net_pnl_bt == Decimal("80.47")
        assert tgim_trade.net_pnl_broker == Decimal("95.50")
        assert tgim_trade.endpreis_delta == Decimal("15.03")  # 95.50 - 80.47
        assert tgim_trade.strategy_display == "TGIM"
        assert "bg-sky-50" in tgim_trade.strategy_badge_class
        assert tgim_trade.has_quantity_diff is False


class TestSummaryAndEquityCurve:
    """Tests KPI summary calculation and cumulative equity curves."""

    def test_compute_summary(self) -> None:
        history = [
            HistoryRealityCheck(
                trade_id=1,
                symbol="A",
                strategy="DipBuyer",
                date="2026-09-01",
                quantity_bt=10.0,
                quantity_broker=10.0,
                unit_label_bt="Stk",
                unit_label_broker="Stk",
                entry_price_bt=Decimal("100.00"),
                entry_price_broker=Decimal("100.50"),
                exit_price_bt=Decimal("110.00"),
                exit_price_broker=Decimal("109.80"),
                entry_slippage=Decimal("0.50"),
                exit_slippage=Decimal("-0.20"),
                total_slippage=Decimal("3.00"),
                net_pnl_bt=Decimal("100.00"),
                net_pnl_broker=Decimal("93.00"),
                commissions=Decimal("2.00"),
                endpreis_delta=Decimal("-7.00"),
                trade_group_id="1_DipBuyer_A",
                strategy_filter="DipBuyer",
            )
        ]
        positions: list[PositionRealityCheck] = []

        summary = compute_reality_check_summary(
            positions, history, total_secondary_pnl=Decimal("15.50")
        )
        assert summary.matched_count == 1
        assert summary.open_matched_count == 0
        assert summary.closed_matched_count == 1
        assert summary.net_pnl_bt == Decimal("100.00")
        assert summary.net_pnl_broker == Decimal("93.00")
        assert summary.total_commissions == Decimal("2.00")
        assert summary.pnl_delta == Decimal("-7.00")
        assert summary.avg_entry_slippage == Decimal("0.50")
        assert summary.avg_exit_slippage == Decimal("-0.20")
        assert summary.total_secondary_pnl == Decimal("15.50")

    def test_compute_dual_equity_curve(self) -> None:
        history = [
            HistoryRealityCheck(
                trade_id=2,
                symbol="B",
                strategy="S",
                date="2026-09-02",
                quantity_bt=1.0,
                quantity_broker=1.0,
                unit_label_bt="Stk",
                unit_label_broker="Stk",
                entry_price_bt=Decimal("10"),
                entry_price_broker=Decimal("10"),
                exit_price_bt=Decimal("15"),
                exit_price_broker=Decimal("15"),
                entry_slippage=Decimal("0"),
                exit_slippage=Decimal("0"),
                total_slippage=Decimal("0"),
                net_pnl_bt=Decimal("50.00"),
                net_pnl_broker=Decimal("60.00"),
                commissions=Decimal("1.00"),
                endpreis_delta=Decimal("10.00"),
                trade_group_id="2_S_B",
                strategy_filter="S",
            ),
            HistoryRealityCheck(
                trade_id=1,
                symbol="A",
                strategy="S",
                date="2026-09-01",
                quantity_bt=1.0,
                quantity_broker=1.0,
                unit_label_bt="Stk",
                unit_label_broker="Stk",
                entry_price_bt=Decimal("10"),
                entry_price_broker=Decimal("10"),
                exit_price_bt=Decimal("12"),
                exit_price_broker=Decimal("12"),
                entry_slippage=Decimal("0"),
                exit_slippage=Decimal("0"),
                total_slippage=Decimal("0"),
                net_pnl_bt=Decimal("20.00"),
                net_pnl_broker=Decimal("15.00"),
                commissions=Decimal("1.00"),
                endpreis_delta=Decimal("-5.00"),
                trade_group_id="1_S_A",
                strategy_filter="S",
            ),
        ]

        curve = compute_dual_equity_curve(history)
        assert len(curve) == 2
        # Chronological order: 2026-09-01 first, then 2026-09-02
        assert curve[0].date == "2026-09-01"
        assert curve[0].cumulative_pnl_bt == 20.0
        assert curve[0].cumulative_pnl_broker == 15.0
        assert curve[0].pnl_delta == -5.0

        assert curve[1].date == "2026-09-02"
        assert curve[1].cumulative_pnl_bt == 70.0  # 20 + 50
        assert curve[1].cumulative_pnl_broker == 75.0  # 15 + 60
        assert curve[1].pnl_delta == 5.0


class TestCostBreakdown:
    """Tests compute_broker_cost_breakdown for secondary trade costs and account overheads."""

    def test_empty_rows_returns_zero_breakdown(self) -> None:
        breakdown = compute_broker_cost_breakdown([])
        assert isinstance(breakdown, BrokerCostBreakdown)
        assert breakdown.total_trade_amount == Decimal("0.00")
        assert breakdown.total_trade_count == 0
        assert breakdown.total_account_amount == Decimal("0.00")
        assert breakdown.total_account_count == 0
        assert breakdown.total_secondary_pnl == Decimal("0.00")
        assert breakdown.total_all_in_costs == Decimal("0.00")
        assert len(breakdown.trade_items) == 2
        assert len(breakdown.account_items) == 3

    def test_categorization_and_double_counting_prevention(self) -> None:
        rows = [
            # Trade-level rows
            {
                "category": "COMMISSION",
                "is_trade_level": 1,
                "item_count": 10,
                "total_amount": -100.00,
            },
            {
                "category": "REGULATORY_FEE",
                "is_trade_level": 1,
                "item_count": 5,
                "total_amount": -2.50,
            },
            {
                "category": "DIVIDEND",
                "is_trade_level": 1,
                "item_count": 3,
                "total_amount": 60.00,
            },
            {
                "category": "WITHHOLDING_TAX",
                "is_trade_level": 1,
                "item_count": 3,
                "total_amount": -9.00,
            },
            {
                "category": "BORROW_FEE",
                "is_trade_level": 1,
                "item_count": 2,
                "total_amount": -0.50,
            },
            # Account-level rows
            {
                "category": "INTEREST_DEBIT",
                "is_trade_level": 0,
                "item_count": 4,
                "total_amount": -25.00,
            },
            {
                "category": "MARKET_DATA",
                "is_trade_level": 0,
                "item_count": 1,
                "total_amount": -1.33,
            },
            {
                "category": "WITHHOLDING_TAX",
                "is_trade_level": 0,
                "item_count": 2,
                "total_amount": -2.00,
            },
            {
                "category": "INTEREST_CREDIT",
                "is_trade_level": 0,
                "item_count": 6,
                "total_amount": 35.00,
            },
            {
                "category": "SYEP_INCOME",
                "is_trade_level": 0,
                "item_count": 2,
                "total_amount": 5.00,
            },
        ]
        breakdown = compute_broker_cost_breakdown(rows)

        # Trade commissions: -100.00 + -2.50 = -102.50 (15 items)
        comm_item = next(
            i for i in breakdown.trade_items if i.category == "COMMISSIONS"
        )
        assert comm_item.amount == Decimal("-102.50")
        assert comm_item.count == 15

        # Net dividends: 60.00 - 9.00 = +51.00 (6 items)
        div_item = next(i for i in breakdown.trade_items if i.category == "DIVIDENDS")
        assert div_item.amount == Decimal("51.00")
        assert div_item.count == 6

        # Borrow fees: -0.50 (2 items)
        borrow_item = next(
            i for i in breakdown.trade_items if i.category == "BORROW_FEES"
        )
        assert borrow_item.amount == Decimal("-0.50")
        assert borrow_item.count == 2

        # Trade totals: -102.50 + 51.00 - 0.50 = -52.00 (23 items)
        assert breakdown.total_trade_amount == Decimal("-52.00")
        assert breakdown.total_trade_count == 23

        # Account margin: -25.00 (4 items)
        margin_item = next(
            i for i in breakdown.account_items if i.category == "MARGIN_INTEREST"
        )
        assert margin_item.amount == Decimal("-25.00")
        assert margin_item.count == 4

        # Account market data: -1.33 (1 item)
        md_item = next(
            i for i in breakdown.account_items if i.category == "MARKET_DATA"
        )
        assert md_item.amount == Decimal("-1.33")
        assert md_item.count == 1

        # Account credit, SYEP & tax: 35.00 + 5.00 - 2.00 = +38.00 (10 items)
        credit_item = next(
            i for i in breakdown.account_items if i.category == "CREDIT_INTEREST"
        )
        assert credit_item.amount == Decimal("38.00")
        assert credit_item.count == 10

        # Account totals: -25.00 - 1.33 + 38.00 = +11.67 (15 items)
        assert breakdown.total_account_amount == Decimal("11.67")
        assert breakdown.total_account_count == 15

        # All-in total: -52.00 + 11.67 = -40.33
        assert breakdown.total_all_in_costs == Decimal("-40.33")

        # Secondary PnL (excludes order commissions to prevent double subtraction):
        # total_all_in (-40.33) - commissions (-102.50) = +62.17
        # Verify: +51.00 (net div) - 0.50 (borrow) - 25.00 (margin) - 3.33 (md) + 40.00 (credit) = +62.17
        assert breakdown.total_secondary_pnl == Decimal("62.17")

    def test_compute_broker_cost_breakdown_with_supplied_commissions(self) -> None:
        rows = [
            {
                "category": "DIVIDEND",
                "is_trade_level": 1,
                "item_count": 5,
                "total_amount": 94.87,
            },
            {
                "category": "BORROW_FEE",
                "is_trade_level": 1,
                "item_count": 2,
                "total_amount": -0.15,
            },
        ]
        breakdown = compute_broker_cost_breakdown(
            rows,
            commissions_amount=Decimal("24.00"),
            commissions_count=6,
            commissions_currency="$",
        )
        comm_item = next(
            i for i in breakdown.trade_items if i.category == "COMMISSIONS"
        )
        assert comm_item.label == "Kommissionen & Fees"
        assert comm_item.count == 6
        assert comm_item.amount == Decimal("-24.00")
        assert comm_item.currency == "$"

        # In foreign currency, total_trade_amount represents the EUR secondary carry
        assert breakdown.total_trade_amount == Decimal("94.72")
        assert breakdown.total_trade_count == 13  # 6 comms + 5 divs + 2 borrows
        assert breakdown.total_secondary_pnl == Decimal("94.72")


class TestTradeViewServiceIntegration:
    """Tests TradeViewService reality check methods with mock repositories."""

    def test_service_with_none_broker_repository(self) -> None:
        service = TradeViewService(
            trade_repository=MagicMock(),
            market_repository=MagicMock(),
            broker_repository=None,
        )
        assert service.get_reality_check_positions() == []
        assert service.get_reality_check_history() == []
        breakdown = service.get_reality_check_cost_breakdown()
        assert breakdown.total_all_in_costs == Decimal("0.00")

    def test_service_with_mock_broker_repository(self) -> None:
        mock_broker = MagicMock()
        mock_broker.get_cash_ledger_aggregates.return_value = [
            {
                "category": "DIVIDEND",
                "is_trade_level": 1,
                "item_count": 1,
                "total_amount": 47.43,
            }
        ]
        service = TradeViewService(
            trade_repository=MagicMock(),
            market_repository=MagicMock(),
            broker_repository=mock_broker,
        )
        breakdown = service.get_reality_check_cost_breakdown(
            commissions_amount=Decimal("12.00"),
            commissions_count=3,
        )
        assert breakdown.total_trade_amount == Decimal("47.43")
        assert breakdown.total_trade_count == 4
        assert breakdown.total_secondary_pnl == Decimal("47.43")
        comm_item = next(
            i for i in breakdown.trade_items if i.category == "COMMISSIONS"
        )
        assert comm_item.amount == Decimal("-12.00")
        assert comm_item.count == 3


class TestRealityCheckRoute:
    """Tests the Flask HTTP route /broker/reality-check."""

    def test_route_returns_200_and_renders_content(self) -> None:
        from app import create_app

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/broker/reality-check")
        assert response.status_code == 200
        assert b"Reality Check" in response.data
        assert b"Matched Trades" in response.data
        assert b"Positions" in response.data
        assert b"History" in response.data
        assert b"Quantity" in response.data
        assert b"Entry Price" in response.data
        assert b"Slippage" in response.data
        assert (
            "Nebenkosten &amp; Ertr\xc3\xa4ge".encode() in response.data
            or b"Nebenkosten" in response.data
        )
        assert b"Commissions" in response.data
        assert b"Reality Gap" in response.data
        assert b"Handelsbezogene Nebenkosten" in response.data

    def test_route_renders_commissions_when_present(self) -> None:
        from app import create_app
        from app.services.trade_manager.reality_check import (
            BrokerCostBreakdown,
            CashLedgerCategorySummary,
            RealityCheckSummary,
        )

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        mock_service = MagicMock()
        mock_service.get_reality_check_positions.return_value = []
        mock_service.get_reality_check_history.return_value = []
        mock_service.get_reality_check_cost_breakdown.return_value = (
            BrokerCostBreakdown(
                trade_items=[
                    CashLedgerCategorySummary(
                        category="COMMISSIONS",
                        label="Kommissionen & Fees",
                        count=5,
                        amount=Decimal("-20.00"),
                        currency="$",
                    )
                ],
                account_items=[],
                total_trade_amount=Decimal("0.00"),
                total_trade_count=5,
                total_account_amount=Decimal("0.00"),
                total_account_count=0,
                total_secondary_pnl=Decimal("0.00"),
                total_all_in_costs=Decimal("-20.00"),
            )
        )
        mock_service.get_reality_check_summary.return_value = RealityCheckSummary(
            matched_count=0,
            open_matched_count=0,
            closed_matched_count=0,
            net_pnl_bt=Decimal("0.00"),
            net_pnl_broker=Decimal("0.00"),
            total_commissions=Decimal("20.00"),
            avg_entry_slippage=Decimal("0.00"),
            avg_exit_slippage=Decimal("0.00"),
            pnl_delta=Decimal("0.00"),
            total_secondary_pnl=Decimal("0.00"),
        )
        mock_service.get_reality_check_equity_curve.return_value = []

        with patch(
            "app.routes.views.trades._get_trade_view_service",
            return_value=mock_service,
        ):
            response = client.get("/broker/reality-check")
            assert response.status_code == 200
            assert b"Kommissionen &amp; Fees" in response.data
            assert b"-20,00&nbsp;$" in response.data

    def test_kpi_cards_round_amounts_over_100_without_decimals(self) -> None:
        """Verifies that amounts >= 100 in KPI cards are rounded without decimals and < 100 with 2 decimals."""
        from app import create_app
        from app.services.trade_manager.reality_check import (
            BrokerCostBreakdown,
            RealityCheckSummary,
        )

        app = create_app()
        app.config["TESTING"] = True
        client = app.test_client()

        mock_service = MagicMock()
        mock_service.get_reality_check_positions.return_value = []
        mock_service.get_reality_check_history.return_value = []
        mock_service.get_reality_check_cost_breakdown.return_value = (
            BrokerCostBreakdown(
                trade_items=[],
                account_items=[],
                total_trade_amount=Decimal("0.00"),
                total_trade_count=0,
                total_account_amount=Decimal("0.00"),
                total_account_count=0,
                total_secondary_pnl=Decimal("150.45"),
                total_all_in_costs=Decimal("0.00"),
            )
        )
        mock_service.get_reality_check_summary.return_value = RealityCheckSummary(
            matched_count=12,
            open_matched_count=2,
            closed_matched_count=10,
            net_pnl_bt=Decimal("1234.56"),
            net_pnl_broker=Decimal("-250.80"),
            total_commissions=Decimal("18.50"),
            avg_entry_slippage=Decimal("0.25"),
            avg_exit_slippage=Decimal("-105.20"),
            pnl_delta=Decimal("45.10"),
            total_secondary_pnl=Decimal("150.45"),
        )
        mock_service.get_reality_check_equity_curve.return_value = []

        with patch(
            "app.routes.views.trades._get_trade_view_service",
            return_value=mock_service,
        ):
            response = client.get("/broker/reality-check")
            assert response.status_code == 200
            # Net PnL Backtest: 1234.56 >= 100 -> rounded without decimals (+1.235 $)
            assert b"+1.235&nbsp;$" in response.data
            # Net PnL Broker: -250.80 <= -100 -> rounded without decimals (-251 $)
            assert b"-251&nbsp;$" in response.data
            # Commissions: 18.50 < 100 -> 2 decimals (18,50 $)
            assert b"18,50&nbsp;$" in response.data
            # Entry Slippage: 0.25 < 100 -> 2 decimals (+0,25 $)
            assert b"+0,25&nbsp;$" in response.data
            # Exit Slippage: -105.20 <= -100 -> rounded without decimals (-105 $)
            assert b"-105&nbsp;$" in response.data
            # Reality Gap: 45.10 < 100 -> 2 decimals (+45,10 $)
            assert b"+45,10&nbsp;$" in response.data
            # Secondary PnL: 150.45 >= 100 -> rounded without decimals (+150 €)
            assert b"+150&nbsp;\xe2\x82\xac" in response.data
