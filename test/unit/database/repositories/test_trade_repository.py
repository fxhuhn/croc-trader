from pathlib import Path
from typing import Any

import pytest

from app.const import ExitReason, TradeStatus
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession


@pytest.fixture
def trade_session(tmp_path: Path) -> DatabaseSession:
    db_file = tmp_path / "test_trades.db"
    session = DatabaseSession(str(db_file))
    repo = TradeRepository(session)
    repo.init_schema()
    return session


def test_init_schema_creates_tables(trade_session: DatabaseSession) -> None:
    TradeRepository(trade_session)
    with trade_session.connect() as conn:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
    assert "trades" in tables
    assert "trade_logs" in tables


def test_create_trade_and_get_trade(trade_session: DatabaseSession) -> None:
    repo = TradeRepository(trade_session)
    ctx = {"date": "2026-08-01", "setup_score": 90}
    trade_id = repo.create_trade(
        symbol="AAPL",
        strategy="dip_buyer",
        size=100.0,
        entry=150.0,
        stop_loss=145.0,
        target=160.0,
        context=ctx,
    )
    assert trade_id > 0

    trade = repo.get_trade(trade_id)
    assert trade is not None
    assert trade["symbol"] == "AAPL"
    assert trade["strategy"] == "dip_buyer"
    assert trade["status"] == "CREATED"
    assert trade["initial_size"] == 100
    assert trade["current_size"] == 100
    assert trade["entry_price"] == 150.0
    assert trade["current_stop_loss"] == 145.0
    assert trade["current_target"] == 160.0


def test_create_trade_financial_validation(
    trade_session: DatabaseSession,
) -> None:
    repo = TradeRepository(trade_session)
    with pytest.raises(
        ValueError, match="Value for entry must be a finite non-negative number"
    ):
        repo.create_trade("AAPL", "dip_buyer", 10, -10.0, 5.0, 15.0, {})

    with pytest.raises(
        ValueError, match="Value for stop_loss must be a finite non-negative number"
    ):
        repo.create_trade("AAPL", "dip_buyer", 10, 10.0, float("nan"), 15.0, {})


def test_create_trade_upsert_existing_candidate(
    trade_session: DatabaseSession,
) -> None:
    repo = TradeRepository(trade_session)
    ctx = {"date": "2026-08-01", "setup_score": 90}
    id1 = repo.create_trade("MSFT", "croc_setup", 50, 200.0, 190.0, 220.0, ctx)

    # Re-run same signal date reset existing created trade
    id2 = repo.create_trade("MSFT", "croc_setup", 60, 202.0, 192.0, 225.0, ctx)
    assert id1 == id2

    trade = repo.get_trade(id1)
    assert trade is not None
    assert trade["initial_size"] == 60
    assert trade["entry_price"] == 202.0

    # Active trade should not be reset
    repo.update_trade(id1, {"status": "ACTIVE"})
    id3 = repo.create_trade("MSFT", "croc_setup", 70, 205.0, 195.0, 230.0, ctx)
    assert id3 == id1
    trade_active = repo.get_trade(id1)
    assert trade_active is not None
    assert trade_active["status"] == "ACTIVE"
    assert trade_active["initial_size"] == 60  # unchanged


def test_update_trade_and_logs(trade_session: DatabaseSession) -> None:
    repo = TradeRepository(trade_session)
    trade_id = repo.create_trade("NVDA", "tgim", 20, 100.0, 90.0, 120.0, {})

    # Invalid column update should raise ValueError
    with pytest.raises(ValueError, match="Invalid column"):
        repo.update_trade(trade_id, {"invalid_column": 123})

    # Valid update
    repo.update_trade(
        trade_id,
        {
            "status": TradeStatus.ACTIVE,
            "entry_date": "2026-08-02 10:00:00",
            "current_size": 15,
        },
        reason="Partial Fill / Active",
    )

    updated = repo.get_trade(trade_id)
    assert updated is not None
    assert updated["status"] == "ACTIVE"
    assert updated["entry_date"] == "2026-08-02"
    assert updated["current_size"] == 15

    # Check trade_logs
    with trade_session.connect() as conn:
        logs = conn.execute(
            "SELECT * FROM trade_logs WHERE trade_id = ? ORDER BY id", (trade_id,)
        ).fetchall()
    assert len(logs) >= 2


def test_get_by_status_and_get_active_trades(
    trade_session: DatabaseSession,
) -> None:
    repo = TradeRepository(trade_session)
    repo.create_trade("TSLA", "dip_buyer", 10, 200.0, 180.0, 240.0, {})
    id2 = repo.create_trade("GOOGL", "dip_buyer", 20, 100.0, 90.0, 120.0, {})
    repo.update_trade(
        id2, {"status": TradeStatus.CLOSED, "exit_reason": ExitReason.TAKE_PROFIT}
    )

    active = repo.get_active_trades()
    assert len(active) == 1
    assert active[0]["symbol"] == "TSLA"

    created = repo.get_by_status(TradeStatus.CREATED)
    assert len(created) == 1

    closed = repo.get_by_status([TradeStatus.CLOSED, TradeStatus.INVALID])
    assert len(closed) == 1
    assert closed[0]["symbol"] == "GOOGL"


def test_get_all_traded_symbols_and_by_strategy(
    trade_session: DatabaseSession,
) -> None:
    repo = TradeRepository(trade_session)
    repo.create_trade("AMZN", "dip_buyer", 10, 100.0, 90.0, 110.0, {"key": "val"})
    repo.create_trade("AMZN", "two_percent", 5, 100.0, 95.0, 105.0, {})

    symbols = repo.get_all_traded_symbols()
    assert set(symbols) == {"AMZN"}

    dip_trades = repo.get_all_by_strategy("dip_buyer")
    assert len(dip_trades) == 1
    assert dip_trades[0]["signal_context"] == {"key": "val"}


def test_exists_and_latest_updated_at(
    trade_session: DatabaseSession,
) -> None:
    repo = TradeRepository(trade_session)
    ctx: dict[str, Any] = {"date": "2026-08-05"}
    repo.create_trade("AMD", "turnover_timing", 10, 50.0, 45.0, 60.0, ctx)

    assert repo.exists("AMD", "turnover_timing", "2026-08-05") is True
    assert repo.exists("AMD", "turnover_timing", "2026-08-06") is False

    assert repo.get_latest_updated_at() is not None


def test_clear_trades(trade_session: DatabaseSession) -> None:
    repo = TradeRepository(trade_session)
    repo.create_trade("META", "bridge_scout", 10, 300.0, 280.0, 340.0, {})
    repo.clear_trades()

    assert len(repo.get_active_trades()) == 0
    assert repo.get_latest_updated_at() is None
