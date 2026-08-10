"""Unit tests for ExchangeMapper (app/mapping.py)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.mapping import ExchangeMapper


@pytest.fixture
def mapper_instance() -> ExchangeMapper:
    m = ExchangeMapper()
    # Reset internal mapping state between tests
    m._mapping = {}
    return m


def test_singleton_instance(mapper_instance: ExchangeMapper) -> None:
    m2 = ExchangeMapper()
    assert mapper_instance is m2


def test_load_mapping_file_not_found(
    mapper_instance: ExchangeMapper, tmp_path: Path
) -> None:
    non_existent_file = tmp_path / "non_existent.json"
    with patch("app.mapping.settings.get_path", return_value=non_existent_file):
        mapper_instance._load_mapping()
        assert mapper_instance._mapping == {}


def test_load_mapping_file_invalid_json(
    mapper_instance: ExchangeMapper, tmp_path: Path
) -> None:
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("NOT_VALID_JSON{", encoding="utf-8")
    with patch("app.mapping.settings.get_path", return_value=invalid_file):
        mapper_instance._load_mapping()
        assert mapper_instance._mapping == {}


def test_load_mapping_success(mapper_instance: ExchangeMapper, tmp_path: Path) -> None:
    valid_file = tmp_path / "mapping.json"
    valid_file.write_text(
        json.dumps({"AAPL": "NASDAQ", "MSFT": "NASDAQ"}), encoding="utf-8"
    )
    with patch("app.mapping.settings.get_path", return_value=valid_file):
        mapper_instance.load()
        assert mapper_instance.get_exchange("AAPL") == "NASDAQ"
        assert mapper_instance.get_exchange("MSFT") == "NASDAQ"


def test_get_exchange_fallback_etf_and_default(
    mapper_instance: ExchangeMapper, tmp_path: Path
) -> None:
    non_existent_file = tmp_path / "empty.json"
    with patch("app.mapping.settings.get_path", return_value=non_existent_file):
        # Default ETF fallback
        assert mapper_instance.get_exchange("QQQ") == "NASDAQ"
        assert mapper_instance.get_exchange("SPY") == "AMEX"
        assert mapper_instance.get_exchange("SXRV.DE") == "XETR"
        # Unknown symbol fallback
        assert mapper_instance.get_exchange("UNKNOWN_TICKER", default="NYSE") == "NYSE"
        assert mapper_instance.get_exchange("UNKNOWN_TICKER") is None
