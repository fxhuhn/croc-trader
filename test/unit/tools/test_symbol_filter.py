"""Unit tests for SymbolFilter tool in app/tools/symbol_filter.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.tools.symbol_filter import SymbolFilter


@pytest.fixture
def mock_cache_file(tmp_path: Path) -> Path:
    cache_file = tmp_path / "preferred_symbols.json"
    mapping = {"GOOG": ["GOOGL"], "FOXA": ["FOX"]}
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(mapping, f)
    return cache_file


def test_symbol_filter_singleton() -> None:
    filter1 = SymbolFilter()
    filter2 = SymbolFilter()
    assert filter1 is filter2


def test_symbol_filter_filter_symbols() -> None:
    sf = SymbolFilter()
    sf._mapping = {"GOOG": ["GOOGL"], "FOXA": ["FOX"]}

    # Case 1: Both winner and loser present -> loser removed
    filtered = sf.filter_symbols(["AAPL", "GOOG", "GOOGL", "MSFT"])
    assert "GOOGL" not in filtered
    assert "GOOG" in filtered
    assert "AAPL" in filtered
    assert "MSFT" in filtered

    # Case 2: Only loser present -> loser kept!
    filtered_loser_only = sf.filter_symbols(["AAPL", "GOOGL"])
    assert "GOOGL" in filtered_loser_only


def test_symbol_filter_load_and_save_cache(tmp_path: Path) -> None:
    sf = SymbolFilter()
    cache_file = tmp_path / "test_cache.json"

    with (
        patch("app.tools.symbol_filter.CACHE_FILE", cache_file),
        patch("app.tools.symbol_filter.CACHE_DIR", tmp_path),
    ):
        sf._mapping = {"AAPL": ["AAPL.OLD"]}
        sf._save_to_cache()
        assert cache_file.exists()

        sf._mapping = {}
        sf._load_from_cache()
        assert sf._mapping == {"AAPL": ["AAPL.OLD"]}


def test_build_mapping_handling(tmp_path: Path) -> None:
    sf = SymbolFilter()

    # Simulate unreachable Yahoo Finance metadata
    with patch("app.tools.symbol_filter.yf.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("Service unavailable")
        res = sf._build_mapping(["SPY", "AAPL"])
        assert res == {}
