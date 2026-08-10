"""Unit tests for app/services/market/types.py."""

from app.services.market.types import YahooRow


def test_yahoo_row_typed_dict() -> None:
    row: YahooRow = {
        "date": "2026-01-01",
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 154.5,
        "volume": 1000000,
    }
    assert row["date"] == "2026-01-01"
    assert row["open"] == 150.0
    assert row["volume"] == 1000000
