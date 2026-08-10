"""Unit tests for MarketHolidayChecker in app/tools/market_holidays.py."""

import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import yaml

from app.tools.market_holidays import MarketHolidayChecker


@pytest.fixture(autouse=True)
def reset_singleton() -> None:
    """Reset singleton instance before each test."""
    MarketHolidayChecker._instance = None
    MarketHolidayChecker._initialized = False


def test_market_holiday_checker_singleton(tmp_path: Path) -> None:
    yaml_file = tmp_path / "holidays.yaml"
    data = {"holidays": {"2026-01-01": "New Year's Day"}}
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")

    checker1 = MarketHolidayChecker(yaml_path=yaml_file)
    checker2 = MarketHolidayChecker(yaml_path=yaml_file)
    assert checker1 is checker2


def test_market_holiday_checker_valid(tmp_path: Path) -> None:
    yaml_file = tmp_path / "holidays.yaml"
    data = {"holidays": {"2026-07-04": "Independence Day", "2026-12-25": "Christmas"}}
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")

    checker = MarketHolidayChecker(yaml_path=yaml_file)

    # Date string
    assert checker.is_holiday("2026-07-04") is True
    assert checker.get_holiday_name("2026-07-04") == "Independence Day"

    # datetime.date
    d = datetime.date(2026, 12, 25)
    assert checker.is_holiday(d) is True
    assert checker.get_holiday_name(d) == "Christmas"

    # Non holiday
    assert checker.is_holiday("2026-07-05") is False
    assert checker.get_holiday_name("2026-07-05") is None


def test_market_holiday_checker_parse_date_types(tmp_path: Path) -> None:
    yaml_file = tmp_path / "holidays.yaml"
    data = {"holidays": {"2026-01-01": "New Year"}}
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")

    checker = MarketHolidayChecker(yaml_path=yaml_file)

    # datetime.datetime
    dt = datetime.datetime(2026, 1, 1, 10, 0, 0)
    assert checker.is_holiday(dt) is True

    # pd.Timestamp
    ts = pd.Timestamp("2026-01-01")
    assert checker.is_holiday(ts) is True

    # Invalid date string format
    with pytest.raises(ValueError, match="Invalid date string provided"):
        checker.is_holiday("01-01-2026")

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported date type"):
        checker.is_holiday(12345)  # type: ignore[arg-type]


def test_market_holiday_checker_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "non_existent.yaml"
    with pytest.raises(
        FileNotFoundError, match="Critical: Holidays configuration file missing"
    ):
        MarketHolidayChecker(yaml_path=missing_file)


def test_market_holiday_checker_empty_yaml(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")

    checker = MarketHolidayChecker(yaml_path=empty_file)
    assert checker.is_holiday("2026-01-01") is False


def test_market_holiday_checker_invalid_date_in_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "bad_dates.yaml"
    data = {"holidays": {"BAD-DATE": "Invalid", "2026-01-01": "Valid"}}
    yaml_file.write_text(yaml.dump(data), encoding="utf-8")

    checker = MarketHolidayChecker(yaml_path=yaml_file)
    # BAD-DATE is skipped during load
    assert checker.is_holiday("2026-01-01") is True
    assert checker.is_holiday("2026-01-02") is False


def test_market_holiday_checker_default_path() -> None:
    yaml_content = yaml.dump({"holidays": {"2026-01-01": "New Year"}})
    m = mock_open(read_data=yaml_content)

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", m),
    ):
        checker = MarketHolidayChecker()
        assert checker.is_holiday("2026-01-01") is True
