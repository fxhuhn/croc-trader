"""Unit tests for ExchangeMapper canonical exchange resolution, heuristics, and discovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.mapping import ExchangeMapper


def test_exchange_mapper_heuristics_german_symbols() -> None:
    """Verifies that German symbols (.DE suffix) deterministically map to XETR."""
    mapper = ExchangeMapper()
    assert mapper.get_exchange("SXRV.DE") == "XETR"
    assert mapper.get_exchange("BMW.DE") == "XETR"
    assert mapper.get_exchange("SAP.de") == "XETR"


def test_exchange_mapper_heuristics_indices_and_etfs() -> None:
    """Verifies that canonical indices and core ETFs resolve via heuristics."""
    mapper = ExchangeMapper()
    assert mapper.get_exchange("^VIX") == "CBOE"
    assert mapper.get_exchange("VIX") == "CBOE"
    assert mapper.get_exchange("^NDX") == "NASDAQ"
    assert mapper.get_exchange("NDX") == "NASDAQ"
    assert mapper.get_exchange("^GSPC") == "INDEX"
    assert mapper.get_exchange("^DJI") == "DJ"
    assert mapper.get_exchange("^RUT") == "RUSSELL"
    assert mapper.get_exchange("SPY") == "AMEX"
    assert mapper.get_exchange("QQQ") == "NASDAQ"
    assert mapper.get_exchange("TLT") == "NASDAQ"


def test_exchange_mapper_normalize_exchange_code() -> None:
    """Tests standardizing provider-specific exchange codes."""
    assert ExchangeMapper.normalize_exchange_code("NYQ") == "NYSE"
    assert ExchangeMapper.normalize_exchange_code("NYSE") == "NYSE"
    assert ExchangeMapper.normalize_exchange_code("NMS") == "NASDAQ"
    assert ExchangeMapper.normalize_exchange_code("NGS") == "NASDAQ"
    assert ExchangeMapper.normalize_exchange_code("ASE") == "AMEX"
    assert ExchangeMapper.normalize_exchange_code("GER") == "XETR"
    assert ExchangeMapper.normalize_exchange_code("XETRA") == "XETR"
    assert ExchangeMapper.normalize_exchange_code("CBOE") == "CBOE"
    assert ExchangeMapper.normalize_exchange_code("XYZ_UNKNOWN") is None


def test_exchange_mapper_register_exchange_merges_without_overwrite(
    tmp_path: Path,
) -> None:
    """Tests registering a new exchange merges non-destructively into persistent store."""
    mapping_file = tmp_path / "symbol_exchange.json"
    initial_data = {"AAPL": "NASDAQ", "P": "NYSE", "BNY": "NYSE"}
    mapping_file.write_text(json.dumps(initial_data), encoding="utf-8")

    mapper = ExchangeMapper()
    with patch("app.mapping.settings.get_path", return_value=mapping_file):
        mapper._mapping = initial_data.copy()
        mapper.register_exchange("CUSTOM_TICKER", "NYSE", persist=True)

        assert mapper.get_exchange("CUSTOM_TICKER") == "NYSE"
        assert mapper.get_exchange("AAPL") == "NASDAQ"

        # Check persisted file on disk
        persisted = json.loads(mapping_file.read_text(encoding="utf-8"))
        assert persisted["CUSTOM_TICKER"] == "NYSE"
        assert persisted["AAPL"] == "NASDAQ"
        assert persisted["P"] == "NYSE"
        assert persisted["BNY"] == "NYSE"


@patch("yfinance.Ticker")
def test_exchange_mapper_auto_discover_exchange(mock_ticker_cls: MagicMock) -> None:
    """Tests discovering exchange via yfinance metadata."""
    mock_ticker = MagicMock()
    mock_ticker.fast_info.exchange = "NYQ"
    mock_ticker_cls.return_value = mock_ticker

    mapper = ExchangeMapper()
    with patch.object(mapper, "register_exchange") as mock_register:
        discovered = mapper.auto_discover_exchange("DISCOVER_ME")

        assert discovered == "NYSE"
        mock_register.assert_called_once_with("DISCOVER_ME", "NYSE", persist=True)
