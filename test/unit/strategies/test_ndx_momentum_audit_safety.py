# filename: test_ndx_momentum_audit_safety.py
"""
Safety Net Unit Test Suite for NDXMomentumTradeStrategy audit remediation.
Locks in month-switch detection, exception logging, and leader extraction behavior.
"""

import json
import logging

import pandas as pd
import pytest

from app.services.trade_manager.strategies.ndx_momentum import (
    NDXMomentumTradeStrategy,
    _RebalanceCache,
)


def test_rebalance_cache_immutability():
    """Ensures _RebalanceCache is frozen and cannot be mutated."""
    cache = _RebalanceCache(cache_key="2026-02", latest_signal_date="2026-02-01")
    with pytest.raises(AttributeError):
        cache.cache_key = "2026-03"  # type: ignore[misc]


def test_is_month_switch_order_logs_warning_on_invalid_reference_date(caplog):
    """Ensures invalid reference_date logs a warning instead of swallowing errors silently."""
    strategy = NDXMomentumTradeStrategy()
    dataframe_history = pd.DataFrame([{"date": "2026-01-30"}])

    with caplog.at_level(logging.WARNING):
        is_switch = strategy._is_month_switch_order(
            dataframe_history, reference_date="invalid-date"
        )

    assert is_switch is False
    assert "Failed to parse reference_date" in caplog.text or len(caplog.records) > 0


def test_extract_latest_leaders_parses_json_and_dict_contexts():
    """Validates leader extraction with both pre-parsed dicts and JSON strings."""
    trades = [
        {
            "id": 1,
            "symbol": "AAPL",
            "signal_context": json.dumps({"date": "2026-02-01"}),
        },
        {
            "id": 2,
            "symbol": "MSFT",
            "signal_context": {"date": "2026-02-01"},
        },
        {
            "id": 3,
            "symbol": "NVDA",
            "signal_context": {"date": "2026-01-01"},
        },
    ]

    leaders = NDXMomentumTradeStrategy.extract_latest_leaders(trades)
    assert leaders == {"AAPL", "MSFT"}
