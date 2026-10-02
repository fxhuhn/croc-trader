"""Unit tests for BrokerRepository in app/database/repositories/broker.py."""

from pathlib import Path

import pytest

from app.database.repositories.broker import BrokerRepository, ExecutionRecord
from app.database.session import DatabaseSession


@pytest.fixture
def broker_session(tmp_path: Path) -> DatabaseSession:
    db_file = tmp_path / "test_trading.db"
    session = DatabaseSession(str(db_file))
    with session.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                perm_id INTEGER,
                parent_id INTEGER,
                trade_group_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                bracket_role TEXT,
                symbol TEXT NOT NULL,
                sec_type TEXT,
                exchange TEXT,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                order_type TEXT NOT NULL,
                target_price REAL DEFAULT 0,
                tif TEXT,
                strategy_name TEXT NOT NULL,
                status TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                transmitted_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                exec_id TEXT PRIMARY KEY,
                order_id INTEGER NOT NULL,
                price REAL NOT NULL,
                qty REAL NOT NULL,
                commission REAL,
                currency TEXT,
                executed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades_settlement (
                account_id TEXT NOT NULL,
                trade_group_id TEXT PRIMARY KEY,
                avg_entry_price REAL NOT NULL,
                avg_exit_price REAL NOT NULL,
                price_diff_slippage REAL NOT NULL,
                total_commissions REAL NOT NULL,
                net_pnl REAL NOT NULL,
                settled_at TEXT NOT NULL
            )
            """
        )
    return session


def test_orders_queries(broker_session: DatabaseSession) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status, transmitted_at)
            VALUES (101, '101_TurnoverTiming_AAPL', 'U12345', 'AAPL', 'BUY', 10, 'LMT', 'TurnoverTiming', 'Submitted', '2026-08-01T10:00:00Z'),
                   (102, '102_DipBuyer_MSFT', 'U12345', 'MSFT', 'BUY', 5, 'LMT', 'DipBuyer', 'Filled', '2026-08-01T11:00:00Z'),
                   (103, '101_TurnoverTiming_AAPL', 'U12345', 'AAPL', 'SELL', 10, 'STP', 'TurnoverTiming', 'Error', '2026-08-01T10:05:00Z')
            """
        )

    all_orders = repo.get_all_orders()
    assert len(all_orders) == 3

    submitted = repo.get_orders_by_status("Submitted")
    assert len(submitted) == 1
    assert submitted[0]["order_id"] == 101

    group_orders = repo.get_orders_by_local_trade_id(101)
    assert len(group_orders) == 2


def test_executions_queries(broker_session: DatabaseSession) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (201, '201_TGIM_NVDA', 'U12345', 'NVDA', 'BUY', 20, 'LMT', 'TGIM', 'Filled')
            """
        )
        conn.execute(
            """
            INSERT INTO executions (exec_id, order_id, price, qty, commission, executed_at)
            VALUES ('exec-1', 201, 120.5, 10, 1.0, '2026-08-02T14:30:00Z'),
                   ('exec-2', 201, 121.0, 10, 1.0, '2026-08-02T14:31:00Z')
            """
        )

    execs_order = repo.get_executions_for_order(201)
    assert len(execs_order) == 2

    execs_group = repo.get_executions_for_trade_group("201_TGIM_NVDA")
    assert len(execs_group) == 2
    assert execs_group[0]["symbol"] == "NVDA"
    assert execs_group[0]["strategy_name"] == "TGIM"


def test_settlements_query(broker_session: DatabaseSession) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO trades_settlement (account_id, trade_group_id, avg_entry_price, avg_exit_price, price_diff_slippage, total_commissions, net_pnl, settled_at)
            VALUES ('U12345', '301_TwoPercent_AMZN', 180.0, 190.0, 0.1, 2.0, 98.0, '2026-08-03T16:00:00Z')
            """
        )

    settlements = repo.get_settlements()
    assert len(settlements) == 1
    assert settlements[0]["trade_group_id"] == "301_TwoPercent_AMZN"
    assert settlements[0]["net_pnl"] == 98.0


def test_net_positions_by_symbol_and_active_positions(
    broker_session: DatabaseSession,
) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (401, '401_DipBuyer_AAPL', 'U12345', 'AAPL', 'BUY', 50, 'LMT', 'DipBuyer', 'Filled'),
                   (402, '401_DipBuyer_AAPL', 'U12345', 'AAPL', 'SELL', 20, 'LMT', 'DipBuyer', 'Filled'),
                   (403, '402_HoldTarget_GOOG', 'U12345', 'GOOG', 'BUY', 100, 'LMT', 'HoldTarget', 'Filled')
            """
        )
        conn.execute(
            """
            INSERT INTO executions (exec_id, order_id, price, qty, commission, executed_at)
            VALUES ('exec-401', 401, 200.0, 50, 1.5, '2026-08-04T09:30:00Z'),
                   ('exec-402', 402, 210.0, 20, 1.5, '2026-08-04T15:30:00Z'),
                   ('exec-403', 403, 150.0, 100, 1.0, '2026-08-04T09:31:00Z')
            """
        )

    net_positions = repo.get_net_positions_by_symbol()
    assert "GOOG" not in net_positions
    assert net_positions["AAPL"] == 30.0

    active_positions = repo.get_active_positions()
    assert len(active_positions) == 1
    pos = active_positions[0]
    assert pos["symbol"] == "AAPL"
    assert pos["current_size"] == 30.0
    assert pos["entry_price"] == 200.0
    assert pos["current_price"] == 210.0
    assert pos["tws_status"] == "Filled"


def test_tws_status_and_helpers_edge_cases(broker_session: DatabaseSession) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (501, '501_TGIM_TSLA', 'U12345', 'TSLA', 'SELL', 10, 'LMT', 'TGIM', 'Error'),
                   (502, '501_TGIM_TSLA', 'U12345', 'TSLA', 'BUY', 10, 'LMT', 'TGIM', 'Submitted')
            """
        )

    tws_status, tws_orders = repo._determine_tws_status("501_TGIM_TSLA")
    assert tws_status == "Error"
    assert len(tws_orders) == 2

    # Sell-only execution to test _resolve_latest_buy_execution fallback
    sell_exec: ExecutionRecord = {
        "exec_id": "exec-501",
        "order_id": 501,
        "price": 250.0,
        "qty": 10.0,
        "commission": 0.0,
        "action": "SELL",
    }
    latest_buy = repo._resolve_latest_buy_execution([sell_exec])
    assert latest_buy == sell_exec

    # Price fallback with empty database row
    price_fallback = repo._resolve_latest_price_fallback("UNKNOWN", [sell_exec])
    assert price_fallback == 250.0

    price_fallback_empty = repo._resolve_latest_price_fallback("UNKNOWN", [])
    assert price_fallback_empty == 0.0


def test_determine_tws_status_presubmitted_and_submitted(
    broker_session: DatabaseSession,
) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (601, '601_DipBuyer_META', 'U12345', 'META', 'BUY', 10, 'LMT', 'DipBuyer', 'PreSubmitted'),
                   (602, '602_Turnover_AMD', 'U12345', 'AMD', 'BUY', 15, 'LMT', 'TurnoverTiming', 'Submitted')
            """
        )

    status_meta, orders_meta = repo._determine_tws_status("601_DipBuyer_META")
    assert status_meta == "PreSubmitted"
    assert len(orders_meta) == 1

    status_amd, orders_amd = repo._determine_tws_status("602_Turnover_AMD")
    assert status_amd == "Submitted"
    assert len(orders_amd) == 1


def test_get_active_positions_non_numeric_trade_group_id(
    broker_session: DatabaseSession,
) -> None:
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (701, 'custom_group_NFLX', 'U12345', 'NFLX', 'BUY', 5, 'LMT', 'TurnoverTiming', 'Filled')
            """
        )
        conn.execute(
            """
            INSERT INTO executions (exec_id, order_id, price, qty, commission, executed_at)
            VALUES ('exec-701', 701, 600.0, 5, 1.0, '2026-08-05T10:00:00Z')
            """
        )

    active = repo.get_active_positions()
    assert len(active) == 1
    assert active[0]["id"] == 0
    assert active[0]["symbol"] == "NFLX"
    assert active[0]["entry_price"] == 600.0


def test_trade_group_prefix_matching(
    broker_session: DatabaseSession,
) -> None:
    """Verifies that _determine_tws_status and get_executions_for_trade_group match via prefix."""
    repo = BrokerRepository(broker_session)

    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO orders (order_id, trade_group_id, account_id, symbol, action, quantity, order_type, strategy_name, status)
            VALUES (801, '1563_TurnoverTiming_0.5_MU', 'U12345', 'MU', 'BUY', 10, 'LMT', 'TurnoverTiming', 'Filled'),
                   (802, '1563_TurnoverTiming_0.5_MU_SL', 'U12345', 'MU', 'SELL', 10, 'STP', 'TurnoverTiming', 'Submitted'),
                   (803, '15630_DipBuyer_XYZ', 'U12345', 'XYZ', 'BUY', 20, 'LMT', 'DipBuyer', 'Filled')
            """
        )
        conn.execute(
            """
            INSERT INTO executions (exec_id, order_id, price, qty, commission, executed_at)
            VALUES ('exec-801', 801, 100.0, 10, 1.0, '2026-08-06T10:00:00Z'),
                   ('exec-802', 802, 95.0, 10, 1.0, '2026-08-06T11:00:00Z'),
                   ('exec-803', 803, 50.0, 20, 1.0, '2026-08-06T12:00:00Z')
            """
        )

    status, orders = repo._determine_tws_status("1563_TurnoverTiming_0.5_MU")
    assert status == "Submitted"
    assert len(orders) == 2
    order_ids = {o["order_id"] for o in orders}
    assert order_ids == {801, 802}
    assert 803 not in order_ids

    executions = repo.get_executions_for_trade_group("1563_TurnoverTiming_0.5_MU")
    assert len(executions) == 2
    exec_ids = {e["exec_id"] for e in executions}
    assert exec_ids == {"exec-801", "exec-802"}


def test_get_account_metrics(broker_session: DatabaseSession) -> None:
    """Verifies that get_account_metrics returns None if table missing,

    and returns the latest record when table exists.
    """
    repo = BrokerRepository(broker_session)

    # 1. Table does not exist in standard broker_session fixture yet
    assert repo.get_account_metrics() is None

    # 2. Create account_metrics table
    with broker_session.connect() as conn:
        conn.execute(
            """
            CREATE TABLE account_metrics (
                account_id TEXT PRIMARY KEY,
                net_liquidation REAL NOT NULL,
                total_cash_value REAL NOT NULL,
                available_funds REAL NOT NULL,
                maint_margin_req REAL NOT NULL,
                cushion_pct REAL NOT NULL,
                buying_power REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    # Table is empty
    assert repo.get_account_metrics() is None

    # 3. Insert records
    with broker_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO account_metrics (account_id, net_liquidation, total_cash_value, available_funds, maint_margin_req, cushion_pct, buying_power, updated_at)
            VALUES ('U111', 50000.0, 10000.0, 30000.0, 15000.0, 70.0, 120000.0, '2026-09-30 08:00:00'),
                   ('U222', 86923.63, 14779.08, 56974.83, 26736.74, 69.24, 379832.17, '2026-10-01 09:00:00')
            """
        )

    latest = repo.get_account_metrics()
    assert latest is not None
    assert latest["account_id"] == "U222"
    assert latest["net_liquidation"] == 86923.63
    assert latest["total_cash_value"] == 14779.08
    assert latest["available_funds"] == 56974.83
    assert latest["maint_margin_req"] == 26736.74
    assert latest["cushion_pct"] == 69.24
    assert latest["buying_power"] == 379832.17
