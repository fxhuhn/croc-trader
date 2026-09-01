"""Tests for ScreenerViewService NDX Momentum candidate deduplication and status mapping.

Verifies that NDX Momentum candidates are consolidated to the Top 5 leaders:
- Symbols already held in the portfolio (ACTIVE) are labeled as HOLD.
- Newly entering symbols are labeled as NEW.
- Total displayed count is strictly capped at the top leaders (5).
"""

from unittest.mock import MagicMock

from app.const import Strategies, TradeStatus
from app.services.screener.view_service import ScreenerViewService


def test_ndx_momentum_candidate_deduplication_all_held() -> None:
    """When all 5 new rebalance candidates are already active positions, all 5 show as HOLD."""
    mock_repository = MagicMock()

    # 5 new rebalance CREATED trades from month-end
    created_trades = [
        {
            "id": 101,
            "symbol": "SNDK",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 3000, "date": "2026-08-31"}',
        },
        {
            "id": 102,
            "symbol": "MU",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 800, "date": "2026-08-31"}',
        },
        {
            "id": 103,
            "symbol": "LITE",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 600, "date": "2026-08-31"}',
        },
        {
            "id": 104,
            "symbol": "STX",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 500, "date": "2026-08-31"}',
        },
        {
            "id": 105,
            "symbol": "WDC",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 450, "date": "2026-08-31"}',
        },
    ]

    # 5 active portfolio positions
    active_trades = [
        {
            "id": 1,
            "symbol": "SNDK",
            "status": "ACTIVE",
            "signal_context": '{"momentum_score": 2500, "date": "2026-04-30"}',
        },
        {
            "id": 2,
            "symbol": "MU",
            "status": "ACTIVE",
            "signal_context": '{"momentum_score": 700, "date": "2025-12-31"}',
        },
        {
            "id": 3,
            "symbol": "LITE",
            "status": "ACTIVE",
            "signal_context": '{"momentum_score": 550, "date": "2026-07-31"}',
        },
        {
            "id": 4,
            "symbol": "STX",
            "status": "ACTIVE",
            "signal_context": '{"momentum_score": 480, "date": "2026-07-31"}',
        },
        {
            "id": 5,
            "symbol": "WDC",
            "status": "ACTIVE",
            "signal_context": '{"momentum_score": 400, "date": "2025-12-31"}',
        },
    ]

    def mock_get_candidates(
        strategy: str, limit: int = 100, statuses: list[TradeStatus] | None = None
    ) -> list[dict]:
        return created_trades + active_trades

    mock_repository.get_trade_candidates.side_effect = mock_get_candidates

    service = ScreenerViewService(mock_repository)
    results = service.get_candidates(Strategies.NDXMomentum)

    assert len(results) == 5
    symbols = [r["symbol"] for r in results]
    assert symbols == ["SNDK", "MU", "LITE", "STX", "WDC"]

    # All 5 should be marked as HOLD
    for r in results:
        assert r["position_status"] == "HOLD"
        assert r["display_date"] == "2026-08-31"


def test_ndx_momentum_candidate_with_new_entrant() -> None:
    """When a new stock enters Top 5, it is labeled as NEW while existing stocks are HOLD."""
    mock_repository = MagicMock()

    # NVDA is new, replacing WDC in Top 5
    created_trades = [
        {
            "id": 201,
            "symbol": "NVDA",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 3500, "date": "2026-08-31"}',
        },
        {
            "id": 202,
            "symbol": "SNDK",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 3000, "date": "2026-08-31"}',
        },
        {
            "id": 203,
            "symbol": "MU",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 800, "date": "2026-08-31"}',
        },
        {
            "id": 204,
            "symbol": "LITE",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 600, "date": "2026-08-31"}',
        },
        {
            "id": 205,
            "symbol": "STX",
            "status": "CREATED",
            "signal_context": '{"momentum_score": 500, "date": "2026-08-31"}',
        },
    ]

    # Existing active positions in portfolio
    active_trades = [
        {
            "id": 1,
            "symbol": "SNDK",
            "status": "ACTIVE",
            "signal_context": '{"date": "2026-04-30"}',
        },
        {
            "id": 2,
            "symbol": "MU",
            "status": "ACTIVE",
            "signal_context": '{"date": "2025-12-31"}',
        },
        {
            "id": 3,
            "symbol": "LITE",
            "status": "ACTIVE",
            "signal_context": '{"date": "2026-07-31"}',
        },
        {
            "id": 4,
            "symbol": "STX",
            "status": "ACTIVE",
            "signal_context": '{"date": "2026-07-31"}',
        },
        {
            "id": 5,
            "symbol": "WDC",
            "status": "ACTIVE",
            "signal_context": '{"date": "2025-12-31"}',
        },
    ]

    def mock_get_candidates(
        strategy: str, limit: int = 100, statuses: list[TradeStatus] | None = None
    ) -> list[dict]:
        return created_trades + active_trades

    mock_repository.get_trade_candidates.side_effect = mock_get_candidates

    service = ScreenerViewService(mock_repository)
    results = service.get_candidates("ndx-momentum")

    assert len(results) == 5

    # NVDA is NEW
    nvda_result = next(r for r in results if r["symbol"] == "NVDA")
    assert nvda_result["position_status"] == "NEW"

    # Other 4 are HOLD
    for sym in ["SNDK", "MU", "LITE", "STX"]:
        held_result = next(r for r in results if r["symbol"] == sym)
        assert held_result["position_status"] == "HOLD"
