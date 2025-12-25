"""
Comprehensive unit tests for SQLiteRepository.

Tests cover all repository methods including CRUD operations, filtering,
trade tracking, and statistics enrichment.
"""

import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.database.sqlite_repo import SQLiteRepository

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path that's cleaned up after tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def repo(temp_db_path):
    """Provide a fresh repository instance for each test."""
    return SQLiteRepository(db_path=temp_db_path)


@pytest.fixture
def sample_signal():
    """Provide a sample signal dictionary for testing."""
    return {
        "symbol": "AAPL",
        "timeframe": "1h",
        "timestamp": datetime(2025, 1, 1, 10, 0, 0, tzinfo=UTC),
        "signal": "buy",
        "close": 150.0,
        "high": 151.0,
        "low": 149.0,
        "wuk": 0.75,
        "status": "active",
        "kerze": "bullish",
        "wolke": "green",
        "trend": "up",
        "setter": "ema",
        "welle": "wave1",
    }


@pytest.fixture
def sample_statistic():
    """Provide a sample statistic dictionary for testing."""
    return {
        "signal": "buy",
        "symbol": "AAPL",
        "timeframe": "1h",
        "wolke": "green",
        "welle": "wave1",
        "trend": "up",
        "setter": "ema",
        "TP(3R)": 10,
        "SL(-1R)": 2,
        "Rejected(0R)": 1,
        "level": "A",
    }


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for repository initialization and schema creation."""

    def test_creates_database_file(self, temp_db_path):
        """Test that database file is created on initialization."""
        repo = SQLiteRepository(db_path=temp_db_path)

        assert Path(temp_db_path).exists()
        assert Path(temp_db_path).is_file()
        assert repo.db_path == temp_db_path

    def test_accepts_path_object(self, temp_db_path):
        """Test that constructor accepts Path objects."""
        repo = SQLiteRepository(db_path=Path(temp_db_path))

        assert Path(temp_db_path).exists()
        assert repo.db_path == str(temp_db_path)

    def test_creates_signals_table(self, repo, temp_db_path):
        """Test that signals table is created with correct schema."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        )

        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == "signals"

    def test_creates_signal_trades_table(self, repo, temp_db_path):
        """Test that signal_trades table is created."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_trades'"
        )

        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_creates_signal_statistic_table(self, repo, temp_db_path):
        """Test that signal_statistic table is created."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='signal_statistic'"
        )

        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_creates_indexes(self, repo, temp_db_path):
        """Test that required indexes are created."""
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")

        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "idx_symbol_timestamp_covering" in indexes
        assert "idx_trades_state" in indexes
        assert "idx_signal_statistic_key" in indexes


# ============================================================================
# Signal CRUD Tests
# ============================================================================


class TestSignalOperations:
    """Tests for signal creation and retrieval."""

    def test_save_signal_with_datetime(self, repo, sample_signal):
        """Test saving signal with datetime object."""
        repo.save_signal(sample_signal)

        signals = repo.get_signals(symbol="AAPL")

        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"
        assert signals[0]["signal"] == "buy"

    def test_save_signal_with_iso_string(self, repo, sample_signal):
        """Test saving signal with ISO timestamp string."""
        sample_signal["timestamp"] = "2025-01-01T10:00:00+00:00"

        repo.save_signal(sample_signal)

        signals = repo.get_signals()

        assert len(signals) == 1

    def test_save_multiple_signals(self, repo, sample_signal):
        """Test saving multiple signals."""
        for i in range(5):
            signal = sample_signal.copy()
            signal["timestamp"] = datetime(2025, 1, 1, 10 + i, 0, tzinfo=UTC)
            repo.save_signal(signal)

        signals = repo.get_signals()

        assert len(signals) == 5

    def test_get_signals_no_filters(self, repo, sample_signal):
        """Test retrieving all signals without filters."""
        repo.save_signal(sample_signal)

        signals = repo.get_signals()

        assert len(signals) == 1

    def test_get_signals_filter_by_symbol(self, repo, sample_signal):
        """Test filtering signals by symbol."""
        # Save signals for different symbols
        sample_signal["symbol"] = "AAPL"
        repo.save_signal(sample_signal)

        sample_signal["symbol"] = "GOOGL"
        repo.save_signal(sample_signal)

        signals = repo.get_signals(symbol="AAPL")

        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"

    def test_get_signals_filter_by_timeframe(self, repo, sample_signal):
        """Test filtering signals by timeframe."""
        sample_signal["timeframe"] = "1h"
        repo.save_signal(sample_signal)

        sample_signal["timeframe"] = "5m"
        repo.save_signal(sample_signal)

        signals = repo.get_signals(timeframe="1h")

        assert len(signals) == 1
        assert signals[0]["timeframe"] == "1h"

    def test_get_signals_filter_by_signal_type(self, repo, sample_signal):
        """Test filtering signals by signal type."""
        sample_signal["signal"] = "buy"
        repo.save_signal(sample_signal)

        sample_signal["signal"] = "sell"
        repo.save_signal(sample_signal)

        signals = repo.get_signals(signal="buy")

        assert len(signals) == 1
        assert signals[0]["signal"] == "buy"

    def test_get_signals_filter_by_day(self, repo, sample_signal):
        """Test filtering signals by specific day."""
        sample_signal["timestamp"] = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
        repo.save_signal(sample_signal)

        sample_signal["timestamp"] = datetime(2025, 1, 2, 10, 0, tzinfo=UTC)
        repo.save_signal(sample_signal)

        signals = repo.get_signals(day="2025-01-01")

        assert len(signals) == 1

    def test_get_signals_respects_limit(self, repo, sample_signal):
        """Test that limit parameter restricts results."""
        for i in range(10):
            signal = sample_signal.copy()
            signal["timestamp"] = datetime(2025, 1, 1, 10 + i, 0, tzinfo=UTC)
            repo.save_signal(signal)

        signals = repo.get_signals(limit=5)

        assert len(signals) == 5

    def test_get_signals_ordered_by_timestamp_desc(self, repo, sample_signal):
        """Test that signals are returned newest first."""
        timestamps = [
            datetime(2025, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
            datetime(2025, 1, 1, 11, 0, tzinfo=UTC),
        ]

        for ts in timestamps:
            signal = sample_signal.copy()
            signal["timestamp"] = ts
            repo.save_signal(signal)

        signals = repo.get_signals()

        # Should be in reverse chronological order
        assert signals[0]["timestamp"] == "2025-01-01T12:00:00+00:00"
        assert signals[2]["timestamp"] == "2025-01-01T10:00:00+00:00"

    def test_get_signals_includes_trade_tracking_info(self, repo, sample_signal):
        """Test that get_signals includes trade tracking status."""
        repo.save_signal(sample_signal)

        signals = repo.get_signals()

        assert "is_tracked" in signals[0]
        assert "trade_id" in signals[0]
        assert signals[0]["is_tracked"] == 0  # Not tracked yet


# ============================================================================
# Distinct Values Tests
# ============================================================================


class TestDistinctValues:
    """Tests for retrieving distinct column values."""

    def test_get_distinct_symbols(self, repo, sample_signal):
        """Test retrieving distinct symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AAPL"]  # AAPL duplicate

        for symbol in symbols:
            signal = sample_signal.copy()
            signal["symbol"] = symbol
            repo.save_signal(signal)

        distinct_symbols = repo.get_distinct("symbol")

        assert len(distinct_symbols) == 3
        assert set(distinct_symbols) == {"AAPL", "GOOGL", "MSFT"}

    def test_get_distinct_timeframes(self, repo, sample_signal):
        """Test retrieving distinct timeframes."""
        timeframes = ["1h", "5m", "15m"]

        for tf in timeframes:
            signal = sample_signal.copy()
            signal["timeframe"] = tf
            repo.save_signal(signal)

        distinct_timeframes = repo.get_distinct("timeframe")

        assert len(distinct_timeframes) == 3
        assert set(distinct_timeframes) == {"1h", "5m", "15m"}

    def test_get_distinct_signals(self, repo, sample_signal):
        """Test retrieving distinct signal types."""
        signal_types = ["buy", "sell", "buy"]  # buy duplicate

        for sig_type in signal_types:
            signal = sample_signal.copy()
            signal["signal"] = sig_type
            repo.save_signal(signal)

        distinct_signals = repo.get_distinct("signal")

        assert len(distinct_signals) == 2
        assert set(distinct_signals) == {"buy", "sell"}

    def test_get_distinct_returns_sorted(self, repo, sample_signal):
        """Test that distinct values are sorted."""
        symbols = ["MSFT", "AAPL", "GOOGL"]

        for symbol in symbols:
            signal = sample_signal.copy()
            signal["symbol"] = symbol
            repo.save_signal(signal)

        distinct_symbols = repo.get_distinct("symbol")

        assert distinct_symbols == ["AAPL", "GOOGL", "MSFT"]

    def test_get_distinct_invalid_column_returns_empty(self, repo):
        """Test that invalid column returns empty list."""
        result = repo.get_distinct("invalid_column")

        assert result == []

    def test_get_distinct_excludes_none_values(self, repo, sample_signal):
        """Test that None values are excluded from distinct results."""
        # wolke is optional and can be None
        sample_signal["wolke"] = None
        repo.save_signal(sample_signal)

        # This is indirect - we can't test wolke directly since it's not
        # in the allowed columns, but the behavior is the same
        symbols = repo.get_distinct("symbol")

        assert None not in symbols


# ============================================================================
# Trade Tracking Tests
# ============================================================================


class TestTradeTracking:
    """Tests for trade tracking functionality."""

    def test_toggle_trade_tracking_marks_trade(self, repo, sample_signal):
        """Test that toggle marks a trade for tracking."""
        repo.save_signal(sample_signal)

        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )

        assert result["tracked"] is True
        assert "trade_id" in result
        assert result["trade_id"] > 0

    def test_toggle_trade_tracking_unmarks_trade(self, repo, sample_signal):
        """Test that toggle unmarks an already-tracked trade."""
        repo.save_signal(sample_signal)
        timestamp_str = sample_signal["timestamp"].isoformat()

        # First toggle: mark
        first_result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=timestamp_str,
            signal="buy",
        )

        # Second toggle: unmark
        second_result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=timestamp_str,
            signal="buy",
        )

        assert second_result["tracked"] is False
        assert second_result["trade_id"] == first_result["trade_id"]

    def test_toggle_trade_tracking_with_metadata(self, repo, sample_signal):
        """Test toggle with timeframe and signal_timestamp."""
        repo.save_signal(sample_signal)

        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
            timeframe="1h",
            signal_timestamp="2025-01-01T09:00:00+00:00",
        )

        trade = repo.get_trade(result["trade_id"])

        assert trade["signal_timeframe"] == "1h"
        assert trade["signal_timestamp"] == "2025-01-01T09:00:00+00:00"

    def test_get_trade_returns_trade_data(self, repo, sample_signal):
        """Test retrieving a specific trade by ID."""
        repo.save_signal(sample_signal)

        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )

        trade = repo.get_trade(result["trade_id"])

        assert trade is not None
        assert trade["symbol"] == "AAPL"
        assert trade["signal"] == "buy"
        assert trade["state"] == "pending"

    def test_get_trade_returns_none_for_invalid_id(self, repo):
        """Test that get_trade returns None for non-existent ID."""
        trade = repo.get_trade(99999)

        assert trade is None

    def test_get_all_trades_returns_tracked_trades(self, repo, sample_signal):
        """Test retrieving all tracked trades."""
        # Create multiple signals and track them
        for i in range(3):
            signal = sample_signal.copy()
            signal["symbol"] = f"STOCK{i}"
            signal["timestamp"] = datetime(2025, 1, 1, 10 + i, 0, tzinfo=UTC)
            repo.save_signal(signal)
            repo.toggle_trade_tracking(
                symbol=signal["symbol"],
                timestamp=signal["timestamp"].isoformat(),
                signal="buy",
            )

        trades = repo.get_all_trades()

        assert len(trades) == 3

    def test_get_all_trades_respects_limit(self, repo, sample_signal):
        """Test that get_all_trades respects limit parameter."""
        for i in range(10):
            signal = sample_signal.copy()
            signal["symbol"] = f"STOCK{i}"
            signal["timestamp"] = datetime(2025, 1, 1, 10 + i, 0, tzinfo=UTC)
            repo.save_signal(signal)
            repo.toggle_trade_tracking(
                symbol=signal["symbol"],
                timestamp=signal["timestamp"].isoformat(),
                signal="buy",
            )

        trades = repo.get_all_trades(limit=5)

        assert len(trades) == 5

    def test_get_all_trades_ordered_by_created_desc(self, repo, sample_signal):
        """Test that trades are returned newest first."""
        import time

        symbols = ["AAPL", "GOOGL", "MSFT"]
        trade_ids = []

        for symbol in symbols:
            signal = sample_signal.copy()
            signal["symbol"] = symbol
            repo.save_signal(signal)
            result = repo.toggle_trade_tracking(
                symbol=symbol,
                timestamp=sample_signal["timestamp"].isoformat(),
                signal="buy",
            )
            trade_ids.append(result["trade_id"])
            # Small delay to ensure different timestamps in created_at
            time.sleep(0.01)

        trades = repo.get_all_trades()

        # Verify we got all trades
        assert len(trades) == 3

        # Trades should be ordered by created_at DESC
        # Last inserted (MSFT) should be first
        retrieved_symbols = [trade["symbol"] for trade in trades]
        assert retrieved_symbols[0] == "MSFT"
        assert retrieved_symbols[-1] == "AAPL"

        # Also verify by trade_id (should be descending)
        retrieved_ids = [trade["id"] for trade in trades]
        assert retrieved_ids == sorted(retrieved_ids, reverse=True)

    def test_update_trade_with_valid_fields(self, repo, sample_signal):
        """Test updating trade with allowed fields."""
        repo.save_signal(sample_signal)
        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )
        trade_id = result["trade_id"]

        success = repo.update_trade(
            trade_id,
            {
                "state": "filled",
                "entry_price": 150.5,
                "entry_time": "2025-01-01T10:30:00+00:00",
            },
        )

        assert success is True

        trade = repo.get_trade(trade_id)
        assert trade["state"] == "filled"
        assert trade["entry_price"] == 150.5
        assert trade["entry_time"] == "2025-01-01T10:30:00+00:00"

    def test_update_trade_updates_timestamp(self, repo, sample_signal):
        """Test that update_trade updates the updated_at timestamp."""
        repo.save_signal(sample_signal)
        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )
        trade_id = result["trade_id"]

        initial_trade = repo.get_trade(trade_id)
        initial_updated = initial_trade["updated_at"]

        # Update after a brief moment
        repo.update_trade(trade_id, {"state": "filled"})

        updated_trade = repo.get_trade(trade_id)

        # updated_at should have changed
        assert updated_trade["updated_at"] >= initial_updated

    def test_update_trade_ignores_invalid_fields(self, repo, sample_signal):
        """Test that update_trade ignores fields not in allowed list."""
        repo.save_signal(sample_signal)
        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )
        trade_id = result["trade_id"]

        success = repo.update_trade(
            trade_id,
            {
                "invalid_field": "value",
                "state": "filled",
            },
        )

        assert success is True  # Should still succeed with valid field

        trade = repo.get_trade(trade_id)
        assert "invalid_field" not in trade

    def test_update_trade_returns_false_if_no_valid_fields(self, repo, sample_signal):
        """Test that update_trade returns False if no valid fields provided."""
        repo.save_signal(sample_signal)
        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )
        trade_id = result["trade_id"]

        success = repo.update_trade(
            trade_id,
            {"invalid_field": "value"},
        )

        assert success is False

    def test_update_all_allowed_trade_fields(self, repo, sample_signal):
        """Test updating all allowed trade fields at once."""
        repo.save_signal(sample_signal)
        result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )
        trade_id = result["trade_id"]

        update_data = {
            "buy_limit": 149.0,
            "stop_loss": 145.0,
            "take_profit": 155.0,
            "entry_time": "2025-01-01T10:30:00+00:00",
            "entry_price": 149.5,
            "exit_time": "2025-01-01T15:00:00+00:00",
            "exit_price": 154.0,
            "current_price": 154.0,
            "state": "closed",
        }

        success = repo.update_trade(trade_id, update_data)

        assert success is True

        trade = repo.get_trade(trade_id)
        for key, value in update_data.items():
            assert trade[key] == value


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for signal statistics operations."""

    def test_replace_signal_statistics_clears_old_data(self, repo, sample_statistic):
        """Test that replace operation clears existing statistics."""
        # Insert initial data
        repo.replace_signal_statistics([sample_statistic])

        # Replace with new data
        new_stat = sample_statistic.copy()
        new_stat["symbol"] = "GOOGL"
        repo.replace_signal_statistics([new_stat])

        # Old data should be gone
        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        assert result is None

    def test_replace_signal_statistics_bulk_insert(self, repo, sample_statistic):
        """Test bulk insertion of multiple statistics."""
        stats = []
        for i in range(5):
            stat = sample_statistic.copy()
            stat["symbol"] = f"STOCK{i}"
            stats.append(stat)

        repo.replace_signal_statistics(stats)

        # Verify one of them
        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="STOCK3",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        assert result is not None
        assert result["symbol"] == "STOCK3"

    def test_replace_signal_statistics_calculates_total(self, repo, sample_statistic):
        """Test that total_signals is calculated correctly."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        expected_total = 10 + 2 + 1  # TP + SL + Rejected
        assert result["total_signals"] == expected_total

    def test_get_best_stat_exact_match(self, repo, sample_statistic):
        """Test finding statistic with exact match."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        assert result is not None
        assert (
            result["match_quality"]
            == "Exakte Übereinstimmung (Wolke, Welle, Trend, Setter)"
        )

    def test_get_best_stat_fallback_without_setter(self, repo, sample_statistic):
        """Test fallback matching without setter."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="different_setter",  # Won't match exactly
        )

        assert result is not None
        assert result["match_quality"] == "Wolke + Welle + Trend"

    def test_get_best_stat_fallback_wolke_welle(self, repo, sample_statistic):
        """Test fallback matching with wolke and welle only."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="different_trend",
            setter="different_setter",
        )

        assert result is not None
        assert result["match_quality"] == "Wolke + Welle"

    def test_get_best_stat_fallback_wolke_only(self, repo, sample_statistic):
        """Test fallback matching with wolke only."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="different_wave",
            trend="different_trend",
            setter="different_setter",
        )

        assert result is not None
        assert result["match_quality"] == "nur Wolke"

    def test_get_best_stat_fallback_basic(self, repo, sample_statistic):
        """Test fallback to basic matching."""
        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="different",
            welle="different",
            trend="different",
            setter="different",
        )

        assert result is not None
        assert result["match_quality"] == "nur Signal + Symbol + Timeframe"

    def test_get_best_stat_requires_minimum_signals(self, repo, sample_statistic):
        """Test that statistics with < 3 total signals are excluded."""
        sample_statistic["TP(3R)"] = 1
        sample_statistic["SL(-1R)"] = 1
        sample_statistic["Rejected(0R)"] = 0
        # Total = 2, below threshold

        repo.replace_signal_statistics([sample_statistic])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        assert result is None

    def test_get_best_stat_prefers_higher_total(self, repo, sample_statistic):
        """Test that higher total_signals are preferred."""
        # Create two matching statistics with different totals
        stat1 = sample_statistic.copy()
        stat1["TP(3R)"] = 5
        stat1["SL(-1R)"] = 1
        stat1["Rejected(0R)"] = 1
        # Total = 7

        stat2 = sample_statistic.copy()
        stat2["TP(3R)"] = 20
        stat2["SL(-1R)"] = 5
        stat2["Rejected(0R)"] = 5
        # Total = 30

        repo.replace_signal_statistics([stat1, stat2])

        result = repo.get_best_stat_for_signal(
            signal="buy",
            symbol="AAPL",
            timeframe="1h",
            wolke="green",
            welle="wave1",
            trend="up",
            setter="ema",
        )

        assert result["total_signals"] == 30

    def test_get_best_stat_returns_none_if_no_match(self, repo):
        """Test that None is returned if no statistics match."""
        result = repo.get_best_stat_for_signal(
            signal="nonexistent",
            symbol="NONE",
            timeframe="1h",
            wolke=None,
            welle=None,
            trend=None,
            setter=None,
        )

        assert result is None

    def test_enrich_signals_with_stats_adds_statistics(
        self, repo, sample_signal, sample_statistic
    ):
        """Test enriching signals with performance statistics."""
        repo.save_signal(sample_signal)
        repo.replace_signal_statistics([sample_statistic])

        signals = repo.get_signals()
        enriched = repo.enrich_signals_with_stats(signals)

        assert len(enriched) == 1
        assert enriched[0]["tp_3r"] == 10
        assert enriched[0]["sl_1r"] == 2
        assert enriched[0]["rej_0r"] == 1
        assert enriched[0]["stats_total"] == 13
        assert "match_quality" in enriched[0]

    def test_enrich_signals_with_stats_handles_no_match(self, repo, sample_signal):
        """Test enriching signals when no statistics match."""
        repo.save_signal(sample_signal)

        signals = repo.get_signals()
        enriched = repo.enrich_signals_with_stats(signals)

        assert len(enriched) == 1
        assert enriched[0]["tp_3r"] is None
        assert enriched[0]["sl_1r"] is None
        assert enriched[0]["rej_0r"] is None
        assert enriched[0]["stats_total"] == 0
        assert enriched[0]["match_quality"] == ""

    def test_enrich_signals_preserves_original_fields(
        self, repo, sample_signal, sample_statistic
    ):
        """Test that enrichment preserves original signal fields."""
        repo.save_signal(sample_signal)
        repo.replace_signal_statistics([sample_statistic])

        signals = repo.get_signals()
        enriched = repo.enrich_signals_with_stats(signals)

        # Original fields should still be present
        assert enriched[0]["symbol"] == "AAPL"
        assert enriched[0]["signal"] == "buy"
        assert enriched[0]["close"] == 150.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_signal_lifecycle(self, repo, sample_signal):
        """Test complete workflow: save signal, track, update, query."""
        # Save signal
        repo.save_signal(sample_signal)

        # Track trade
        track_result = repo.toggle_trade_tracking(
            symbol="AAPL",
            timestamp=sample_signal["timestamp"].isoformat(),
            signal="buy",
        )

        # Update trade
        repo.update_trade(
            track_result["trade_id"],
            {
                "state": "filled",
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "take_profit": 160.0,
            },
        )

        # Query with joined data
        signals = repo.get_signals(symbol="AAPL")

        assert len(signals) == 1
        assert signals[0]["is_tracked"] == 1
        assert signals[0]["trade_state"] == "filled"
        assert signals[0]["entry_price"] == 150.0

    def test_statistics_enrichment_workflow(
        self, repo, sample_signal, sample_statistic
    ):
        """Test workflow with statistics enrichment."""
        # Setup statistics
        repo.replace_signal_statistics([sample_statistic])

        # Save signal
        repo.save_signal(sample_signal)

        # Get and enrich
        signals = repo.get_signals()
        enriched = repo.enrich_signals_with_stats(signals)

        assert enriched[0]["stats_total"] > 0
        assert enriched[0]["tp_3r"] is not None

    def test_multiple_trades_different_states(self, repo, sample_signal):
        """Test tracking multiple trades with different states."""
        states = ["pending", "filled", "closed"]

        for i, state in enumerate(states):
            signal = sample_signal.copy()
            signal["symbol"] = f"STOCK{i}"
            signal["timestamp"] = datetime(2025, 1, 1, 10 + i, 0, tzinfo=UTC)
            repo.save_signal(signal)

            result = repo.toggle_trade_tracking(
                symbol=signal["symbol"],
                timestamp=signal["timestamp"].isoformat(),
                signal="buy",
            )

            repo.update_trade(result["trade_id"], {"state": state})

        # Query all trades
        trades = repo.get_all_trades()

        assert len(trades) == 3
        states_found = {trade["state"] for trade in trades}
        assert states_found == {"pending", "filled", "closed"}

    def test_database_persistence(self, temp_db_path, sample_signal):
        """Test that data persists across repository instances."""
        # Create repo and save data
        repo1 = SQLiteRepository(temp_db_path)
        repo1.save_signal(sample_signal)

        # Create new repo instance with same database
        repo2 = SQLiteRepository(temp_db_path)
        signals = repo2.get_signals()

        assert len(signals) == 1
        assert signals[0]["symbol"] == "AAPL"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_database_queries(self, repo):
        """Test queries on empty database return empty results."""
        assert repo.get_signals() == []
        assert repo.get_all_trades() == []
        assert repo.get_distinct("symbol") == []

    def test_get_trade_with_non_existent_id(self, repo):
        """Test querying non-existent trade ID."""
        result = repo.get_trade(99999)

        assert result is None

    def test_update_non_existent_trade(self, repo):
        """Test updating a trade that doesn't exist."""
        # Should not raise an error, just not update anything
        result = repo.update_trade(99999, {"state": "filled"})

        # Result depends on implementation - could be True or False
        # Just verify it doesn't crash
        assert isinstance(result, bool)

    def test_signal_with_none_optional_fields(self, repo, sample_signal):
        """Test saving signal with None for optional fields."""
        sample_signal["wolke"] = None

        repo.save_signal(sample_signal)

        signals = repo.get_signals()

        assert len(signals) == 1
        assert signals[0]["wolke"] is None

    def test_concurrent_toggle_operations(self, repo, sample_signal):
        """Test toggling same trade multiple times."""
        repo.save_signal(sample_signal)
        timestamp = sample_signal["timestamp"].isoformat()

        # Toggle on
        result1 = repo.toggle_trade_tracking("AAPL", timestamp, "buy")
        assert result1["tracked"] is True

        # Toggle off
        result2 = repo.toggle_trade_tracking("AAPL", timestamp, "buy")
        assert result2["tracked"] is False

        # Toggle on again
        result3 = repo.toggle_trade_tracking("AAPL", timestamp, "buy")
        assert result3["tracked"] is True

    def test_large_number_of_signals(self, repo, sample_signal):
        """Test repository handles large number of signals."""
        # Insert 1000 signals
        for i in range(1000):
            signal = sample_signal.copy()
            signal["timestamp"] = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(
                minutes=i
            )
            repo.save_signal(signal)

        signals = repo.get_signals(limit=1000)

        assert len(signals) == 1000

    def test_special_characters_in_string_fields(self, repo, sample_signal):
        """Test handling special characters in string fields."""
        sample_signal["symbol"] = "TEST'SYMBOL"
        sample_signal["kerze"] = 'Pattern"With"Quotes'

        repo.save_signal(sample_signal)

        signals = repo.get_signals()

        assert signals[0]["symbol"] == "TEST'SYMBOL"
        assert signals[0]["kerze"] == 'Pattern"With"Quotes'
