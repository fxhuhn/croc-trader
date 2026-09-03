"""Characterization and pinning test suite for ScreenerViewService.

Pins all behavior, ranking, context parsing, date normalization,
strategy harmonization, and turnover aggregations before refactoring.
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.const import Strategies, TradeStatus
from app.services.screener.view_service import ScreenerViewService


@pytest.fixture
def mock_signal_repo() -> MagicMock:
    """Mock for SignalRepository."""
    return MagicMock()


@pytest.fixture
def service(mock_signal_repo: MagicMock) -> ScreenerViewService:
    """ScreenerViewService instance wired with mock repository."""
    return ScreenerViewService(mock_signal_repo)


# ===========================================================================
# 1. Context Parsing Tests
# ===========================================================================


def test_parse_context_variations(service: ScreenerViewService) -> None:
    """Verifies safe JSON and dict parsing in _parse_context."""
    # Dict input returns a copy
    assert service._parse_context({"score": 10}) == {"score": 10}

    # None, empty string, or falsy returns {}
    assert service._parse_context(None) == {}
    assert service._parse_context("") == {}

    # Valid JSON string returns dict
    assert service._parse_context('{"score": 10, "symbol": "AAPL"}') == {
        "score": 10,
        "symbol": "AAPL",
    }

    # JSON that parses to a non-dict returns {}
    assert service._parse_context('["item1", "item2"]') == {}
    assert service._parse_context('"just_a_string"') == {}
    assert service._parse_context("12345") == {}

    # Corrupt JSON string returns {}
    assert service._parse_context("{malformed json") == {}

    # Unexpected data type (e.g. integer or object) returns {}
    assert service._parse_context(12345) == {}
    assert service._parse_context([1, 2, 3]) == {}


# ===========================================================================
# 2. Date and Status Normalization
# ===========================================================================


def test_get_candidates_date_and_status_formatting(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies ISO timestamp splitting, setup_date fallback, and position_status mapping."""
    mock_signal_repo.get_trade_candidates.return_value = [
        {
            "id": 1,
            "symbol": "AAPL",
            "status": "ACTIVE",
            "signal_context": '{"date": "2026-05-15T14:30:00"}',
        },
        {
            "id": 2,
            "symbol": "MSFT",
            "status": "CREATED",
            "signal_context": '{"setup_date": "2026-05-16 09:30:00"}',
        },
        {
            "id": 3,
            "symbol": "GOOG",
            "status": "INVALIDATED",
            "signal_context": "{}",  # Missing date
        },
    ]

    candidates = service.get_candidates(Strategies.DipBuyer)
    assert len(candidates) == 3

    # AAPL: ACTIVE -> HOLD, ISO date split
    assert candidates[0]["symbol"] == "AAPL"
    assert candidates[0]["position_status"] == "HOLD"
    assert candidates[0]["display_date"] == "2026-05-15"

    # MSFT: CREATED -> NEW, space date split
    assert candidates[1]["symbol"] == "MSFT"
    assert candidates[1]["position_status"] == "NEW"
    assert candidates[1]["display_date"] == "2026-05-16"

    # GOOG: INVALIDATED -> NEW, missing date -> "-"
    assert candidates[2]["symbol"] == "GOOG"
    assert candidates[2]["position_status"] == "NEW"
    assert candidates[2]["display_date"] == "-"


# ===========================================================================
# 3. Strategy Resolution & Fetch Routing
# ===========================================================================


def test_get_candidates_croc_strategy_routing_and_cap(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies CrocSetup alias resolution, deduplication, and max 3 candidate limit."""
    mock_signal_repo.get_trade_candidates.side_effect = [
        # HoldTarget
        [
            {"id": 1, "symbol": "NVDA", "created_at": "2026-05-01 10:00:00"},
            {"id": 2, "symbol": "AMD", "created_at": "2026-05-01 11:00:00"},
        ],
        # SplitTarget
        [
            {
                "id": 2,
                "symbol": "AMD",
                "created_at": "2026-05-01 11:00:00",
            },  # duplicate ID
            {"id": 3, "symbol": "INTC", "created_at": "2026-05-01 12:00:00"},
        ],
        # Croc_
        [
            {"id": 4, "symbol": "QCOM", "created_at": "2026-05-01 13:00:00"},
            {"id": 5, "symbol": "AVGO", "created_at": "2026-05-01 14:00:00"},
        ],
    ]

    # Test with string alias "croc_setup"
    candidates = service.get_candidates("croc_setup", limit=10)

    # Must deduplicate id=2 and cap at strictly 3 sorted descending by created_at
    assert len(candidates) == 3
    # Top 3 created_at: id=5 (14:00), id=4 (13:00), id=3 (12:00)
    assert [c["id"] for c in candidates] == [5, 4, 3]


def test_get_candidates_list_strategy_input(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies that passing a list of strategy strings calls repository correctly."""
    mock_signal_repo.get_trade_candidates.return_value = [
        {"id": 1, "symbol": "TSLA", "status": "CREATED", "signal_context": "{}"}
    ]

    candidates = service.get_candidates(["strat_a", "strat_b"])
    assert len(candidates) == 1
    mock_signal_repo.get_trade_candidates.assert_called_with(
        ["strat_a", "strat_b"], limit=100, statuses=[TradeStatus.CREATED]
    )


# ===========================================================================
# 4. Strategy Scoring Sort Orders
# ===========================================================================


def test_get_candidates_ndx_momentum_sorting(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies NDXMomentum sorts candidates by context.momentum_score DESC."""
    mock_signal_repo.get_trade_candidates.return_value = [
        {
            "id": 1,
            "symbol": "LOW",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 10.5}',
        },
        {
            "id": 2,
            "symbol": "HIGH",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 99.9}',
        },
        {
            "id": 3,
            "symbol": "MID",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 50.0}',
        },
        {
            "id": 4,
            "symbol": "NONE",
            "status": "CREATED",
            "signal_context": "{}",  # Missing score -> 0.0
        },
    ]

    candidates = service.get_candidates(Strategies.NDXMomentum)
    symbols = [c["symbol"] for c in candidates]
    assert symbols == ["HIGH", "MID", "LOW", "NONE"]


def test_get_candidates_dip_buyer_sorting(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies DipBuyer sorts candidates by context.setup_score DESC."""
    mock_signal_repo.get_trade_candidates.return_value = [
        {
            "id": 1,
            "symbol": "B",
            "status": "CREATED",
            "signal_context": '{"setup_score": 4}',
        },
        {
            "id": 2,
            "symbol": "A",
            "status": "CREATED",
            "signal_context": '{"setup_score": 8}',
        },
        {
            "id": 3,
            "symbol": "C",
            "status": "CREATED",
            "signal_context": "{}",
        },
    ]

    candidates = service.get_candidates(Strategies.DipBuyer)
    symbols = [c["symbol"] for c in candidates]
    assert symbols == ["A", "B", "C"]


# ===========================================================================
# 5. NDX Momentum Helper Edge Cases
# ===========================================================================


def test_group_momentum_rows_skips_empty_symbols_and_duplicate_statuses(
    service: ScreenerViewService,
) -> None:
    """Verifies that empty symbols are ignored and first entry per symbol per status wins."""
    rows: list[dict[str, Any]] = [
        {"symbol": "", "status": "CREATED"},
        {"symbol": "   ", "status": "ACTIVE"},
        {"symbol": "AAPL", "status": "CREATED", "note": "first"},
        {"symbol": "AAPL", "status": "CREATED", "note": "second"},
        {"symbol": "MSFT", "status": "ACTIVE", "note": "first_active"},
        {"symbol": "MSFT", "status": "ACTIVE", "note": "second_active"},
    ]

    created_map, active_map, created_order, active_order = service._group_momentum_rows(
        rows
    )

    assert list(created_map.keys()) == ["AAPL"]
    assert created_map["AAPL"]["note"] == "first"
    assert created_order == ["AAPL"]

    assert list(active_map.keys()) == ["MSFT"]
    assert active_map["MSFT"]["note"] == "first_active"
    assert active_order == ["MSFT"]


def test_build_ndx_candidates_caps_at_max_leaders(
    service: ScreenerViewService,
) -> None:
    """Verifies that active positions fill remaining slots up to MAX_NDX_MOMENTUM_LEADERS (5)."""
    # 2 CREATED signals
    created_map = {
        "A": {"symbol": "A", "status": "CREATED"},
        "B": {"symbol": "B", "status": "CREATED"},
    }
    created_order = ["A", "B"]

    # 5 ACTIVE signals
    active_map = {
        "B": {"symbol": "B", "status": "ACTIVE"},  # Overlaps with CREATED
        "C": {"symbol": "C", "status": "ACTIVE"},
        "D": {"symbol": "D", "status": "ACTIVE"},
        "E": {"symbol": "E", "status": "ACTIVE"},
        "F": {"symbol": "F", "status": "ACTIVE"},
    }
    active_order = ["B", "C", "D", "E", "F"]

    results = service._build_ndx_candidates(
        created_map, active_map, created_order, active_order, limit=10
    )

    # Should contain: A (CREATED), B (ACTIVE because in active_map), and C, D, E from active.
    # Total count = 5. F is truncated!
    assert len(results) == 5
    assert results[0]["symbol"] == "A"
    assert results[0]["status"] == str(TradeStatus.CREATED)
    assert results[1]["symbol"] == "B"
    assert results[1]["status"] == str(TradeStatus.ACTIVE)
    assert [r["symbol"] for r in results] == ["A", "B", "C", "D", "E"]


# ===========================================================================
# 6. Index Harmonization Tests
# ===========================================================================


def test_harmonize_indices_cases(service: ScreenerViewService) -> None:
    """Verifies mapping of raw index strings to short codes."""
    assert service.harmonize_indices("") == "-"
    assert service.harmonize_indices(None) == "-"  # type: ignore[arg-type]

    # Explicit mappings
    assert service.harmonize_indices("NASDAQ_100") == "NDX"
    assert service.harmonize_indices("SP_500") == "SPX"
    assert service.harmonize_indices("RUSSELL_1000") == "RUS"
    assert service.harmonize_indices("RUSSELL_2000") == "RUT"
    assert service.harmonize_indices("DOW_JONES") == "DOW"

    # Multi-value comma separated
    assert (
        service.harmonize_indices("NASDAQ_100, SP_500, RUSSELL_1000") == "NDX, SPX, RUS"
    )

    # Unmapped index fallback replaces underscores with spaces
    assert service.harmonize_indices("CUSTOM_TECH_INDEX") == "CUSTOM TECH INDEX"
    assert (
        service.harmonize_indices("NASDAQ_100, CUSTOM_BASKET") == "NDX, CUSTOM BASKET"
    )


# ===========================================================================
# 7. Turnover Candidates Aggregation Tests
# ===========================================================================


def test_get_turnover_candidates_aggregation_and_error_handling(
    service: ScreenerViewService, mock_signal_repo: MagicMock
) -> None:
    """Verifies multi-variant aggregation, metric extraction, and error resilience."""
    mock_signal_repo.get_trade_candidates.return_value = [
        # AAPL 0.5 variant
        {
            "symbol": "AAPL",
            "strategy": Strategies.TurnOverTiming_05,
            "entry_price": 150.0,
            "signal_context": json.dumps(
                {
                    "setup_close": 155.0,
                    "setup_atr": 3.5,
                    "setup_turnover_sma": 2_000_000_000.0,
                    "date": "2026-05-10T00:00:00",
                    "indices": "NASDAQ_100, SP_500",
                }
            ),
        },
        # AAPL 1.0 variant (merges into existing AAPL bucket)
        {
            "symbol": "AAPL",
            "strategy": Strategies.TurnOverTiming_10,
            "entry_price": 152.5,
            "signal_context": json.dumps(
                {
                    "setup_close": 155.0,
                    "setup_atr": 3.5,
                    "setup_turnover_sma": 2_000_000_000.0,
                    "date": "2026-05-10T00:00:00",
                    "bucket": "NASDAQ_100",  # Tests bucket fallback
                }
            ),
        },
        # MSFT with setup_date fallback
        {
            "symbol": "MSFT",
            "strategy": Strategies.TurnOverTiming_05,
            "entry_price": 400.0,
            "signal_context": json.dumps(
                {
                    "setup_close": 410.0,
                    "setup_atr": 6.0,
                    "setup_turnover_sma": 3_000_000_000.0,
                    "setup_date": "2026-05-09",
                }
            ),
        },
        # Malformed row with invalid numbers (should be caught by exception handler)
        {
            "symbol": "ERR",
            "strategy": Strategies.TurnOverTiming_05,
            "entry_price": "not-a-number",
            "signal_context": '{"setup_close": "invalid_float"}',
        },
    ]

    results = service.get_turnover_candidates()

    # Sorted by dollar_volume descending: MSFT (3B), AAPL (2B), ERR (0.0)
    assert len(results) == 3
    assert results[0]["symbol"] == "MSFT"
    assert results[0]["dollar_volume"] == 3_000_000_000.0
    assert results[0]["display_date"] == "2026-05-09"
    assert results[0]["entry_0_5"] == 400.0
    assert results[0]["entry_1_0"] is None

    assert results[1]["symbol"] == "AAPL"
    assert results[1]["dollar_volume"] == 2_000_000_000.0
    assert results[1]["entry_0_5"] == 150.0
    assert results[1]["entry_1_0"] == 152.5
    assert results[1]["close"] == 155.0
    assert results[1]["atr"] == 3.5
    assert results[1]["index"] == "NDX"  # Overwritten by second row's bucket fallback
    assert results[1]["display_date"] == "2026-05-10"

    assert results[2]["symbol"] == "ERR"
    assert results[2]["dollar_volume"] == 0.0
