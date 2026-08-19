import json
from pathlib import Path
from typing import Any, cast

import pytest

from app.const import Strategies, TradeStatus
from app.database.repositories.signal import SignalRepository
from app.database.session import DatabaseSession


@pytest.fixture
def signal_session(tmp_path: Path) -> DatabaseSession:
    db_file = tmp_path / "test_signals.db"
    session = DatabaseSession(str(db_file))
    repo = SignalRepository(session)
    repo.init_schema()

    # Also create trades table for testing get_trade_candidates
    with session.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT DEFAULT 'CREATED',
                initial_size REAL DEFAULT 0,
                current_size REAL DEFAULT 0,
                entry_price REAL,
                entry_date TIMESTAMP,
                current_price REAL,
                current_stop_loss REAL,
                current_target REAL,
                avg_exit_price REAL,
                realized_pnl REAL DEFAULT 0,
                exit_price REAL,
                exit_date TIMESTAMP,
                exit_reason TEXT,
                signal_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    return session


def test_init_schema_creates_tables_and_view(
    signal_session: DatabaseSession,
) -> None:
    SignalRepository(signal_session)
    with signal_session.connect() as conn:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        ]
    assert "croc" in tables
    assert "exchange_mappings" in tables
    assert "view_signals_enriched" in tables


def test_save_signal_and_get_by_id(signal_session: DatabaseSession) -> None:
    repo = SignalRepository(signal_session)
    payload = {
        "symbol": "AAPL",
        "timeframe": "1d",
        "signal": "DipBuyer",
        "exchange": "NASDAQ",
        "timestamp": "2026-08-01T10:00:00Z",
        "context": {"score": 85},
    }
    row_id = repo.save_signal(payload)
    assert row_id > 0

    fetched = repo.get_signal_by_id(row_id)
    assert fetched is not None
    assert fetched["symbol"] == "AAPL"
    assert fetched["timeframe"] == "1d"
    assert fetched["signal"] == "DipBuyer"
    assert fetched["exchange"] == "NASDAQ"

    # Test missing / non-existent id
    assert repo.get_signal_by_id(99999) is None


def test_save_signal_duplicate_upsert(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    payload1 = {
        "symbol": "MSFT",
        "timeframe": "1d",
        "signal": "CrocSetup",
        "exchange": "NASDAQ",
        "timestamp": "2026-08-02T10:00:00Z",
        "val": 1,
    }
    payload2 = {
        "symbol": "MSFT",
        "timeframe": "1d",
        "signal": "CrocSetup",
        "exchange": "NASDAQ",
        "timestamp": "2026-08-02T10:00:00Z",
        "val": 2,
    }
    repo.save_signal(payload1)
    repo.save_signal(payload2)

    unprocessed = repo.get_unprocessed_signals(limit=10)
    assert len(unprocessed) == 1
    assert "MSFT" in unprocessed[0]["symbol"]


def test_get_by_timestamp(signal_session: DatabaseSession) -> None:
    repo = SignalRepository(signal_session)
    timestamp = "2026-08-03T14:30:00Z"
    repo.save_signal(
        {
            "ticker": "NVDA",
            "strategy": "TGIM",
            "date": timestamp,
            "context": {"setup_score": 90},
        }
    )

    results = repo.get_by_timestamp("TGIM", timestamp)
    assert len(results) == 1
    assert results[0]["symbol"] == "NVDA"
    ctx = cast(dict[str, Any], results[0]["context"])
    assert ctx["setup_score"] == 90

    # Non-matching strategy or timestamp
    assert len(repo.get_by_timestamp("DipBuyer", timestamp)) == 0


def test_get_by_timestamp_invalid_context(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    with signal_session.connect() as conn:
        conn.execute(
            "INSERT INTO croc (symbol, timeframe, signal, timestamp, exchange, data) VALUES (?, ?, ?, ?, ?, ?)",
            ("AMZN", "1d", "TestSig", "2026-08-04", "NASDAQ", "invalid_json{"),
        )
    res = repo.get_by_timestamp("TestSig", "2026-08-04")
    assert len(res) == 1
    assert res[0]["context"] == {}


def test_get_signals_by_date_and_latest_date(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    assert repo.get_latest_signal_date() is None

    repo.save_signal(
        {
            "symbol": "GOOGL",
            "signal": "BounceBandit",
            "timestamp": "2026-08-05T09:00:00Z",
        }
    )
    repo.save_signal(
        {
            "symbol": "META",
            "signal": "BridgeScout",
            "timestamp": "2026-08-06T09:00:00Z",
        }
    )

    assert repo.get_latest_signal_date() == "2026-08-06"

    exact = repo.get_signals_by_date(analysis_date="2026-08-05")
    assert len(exact) == 1
    assert exact[0]["symbol"] == "GOOGL"

    lookback = repo.get_signals_by_date(days_lookback=30)
    assert len(lookback) >= 2


def test_get_unique_signal_attributes(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    repo.save_signal(
        {
            "symbol": "TSLA",
            "signal": "DipBuyer",
            "status": "active",
            "kerze": "hammer",
            "wolke": "grün",
            "trend": "up",
            "setter": "rsi",
            "welle": "1",
            "custom_signal_flag": "true",
        }
    )

    attrs = repo.get_unique_signal_attributes()
    assert "DipBuyer" in attrs["Signal"]
    assert "active" in attrs["Status"]
    assert "hammer" in attrs["Kerze"]
    assert "custom_signal_flag" in attrs["Signal"]


def test_get_unique_signal_attributes_malformed_json(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    with signal_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO croc (symbol, timeframe, signal, timestamp, exchange, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "1d", "TurnoverTiming", "2026-08-01", "SMART", "malformed_json{"),
        )
        conn.execute(
            """
            INSERT INTO croc (symbol, timeframe, signal, timestamp, exchange, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "NVDA",
                "1d",
                "DipBuyer",
                "2026-08-01",
                "SMART",
                json.dumps({"status": "active", "kerze": "hammer", "custom_flag": "1"}),
            ),
        )

    attrs = repo.get_unique_signal_attributes()
    assert "TurnoverTiming" in attrs["Signal"]
    assert "DipBuyer" in attrs["Signal"]
    assert "custom_flag" in attrs["Signal"]
    assert "active" in attrs["Status"]
    assert "hammer" in attrs["Kerze"]


def test_get_trade_candidates(signal_session: DatabaseSession) -> None:
    repo = SignalRepository(signal_session)
    ctx = {"setup_score": 80, "market_phase": "BULL", "date": "2026-08-05"}
    with signal_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO trades (symbol, strategy, status, signal_context)
            VALUES (?, ?, ?, ?)
            """,
            ("NFLX", "dip_buyer", "CREATED", json.dumps(ctx)),
        )
        conn.execute(
            """
            INSERT INTO trades (symbol, strategy, status, signal_context)
            VALUES (?, ?, ?, ?)
            """,
            ("AMD", "two_percent", "ACTIVE", json.dumps({"date": "2026-08-04"})),
        )

    # Search with string prefix
    cands = repo.get_trade_candidates(strategy_prefix="dip_buyer")
    assert len(cands) == 1
    assert cands[0]["symbol"] == "NFLX"
    assert cands[0]["setup_score"] == 80
    assert cands[0]["market_phase"] == "BULL"
    assert cands[0]["display_date"] == "2026-08-05"

    # Search with list of Enum / strings
    cands_list = repo.get_trade_candidates(
        strategy_prefix=[Strategies.DipBuyer, "two_percent"],
        statuses=[TradeStatus.CREATED, TradeStatus.ACTIVE],
    )
    assert len(cands_list) == 2


def test_get_trade_candidates_corrupt_json(
    signal_session: DatabaseSession,
) -> None:
    repo = SignalRepository(signal_session)
    with signal_session.connect() as conn:
        conn.execute(
            """
            INSERT INTO trades (symbol, strategy, status, signal_context)
            VALUES (?, ?, ?, ?)
            """,
            ("INTC", "dip_buyer", "CREATED", "corrupt_json{"),
        )
    cands = repo.get_trade_candidates("dip_buyer")
    assert len(cands) == 1
    assert cands[0]["context"] == {}
    assert cands[0]["display_date"] == "-"
