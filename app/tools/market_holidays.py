import datetime
import logging
import threading
from pathlib import Path
from typing import Optional, TypedDict

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HolidayConfig(TypedDict):
    """Type definition for the holidays YAML structure."""

    holidays: dict[str, str]


class MarketHolidayChecker:
    """
    Singleton class to load public holidays from a YAML file and provide check methods.
    Thread-safe implementation.
    """

    _instance: Optional["MarketHolidayChecker"] = None
    _initialized: bool = False
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "MarketHolidayChecker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, yaml_path: str | Path | None = None) -> None:
        """
        Initialize the MarketHolidayChecker.

        Args:
            yaml_path: Optional path to the holidays.yaml file.
                       If None, attempts to locate it relative to this file.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.debug("Initializing MarketHolidayChecker...")

            self._holidays: dict[datetime.date, str] = {}

            # Determine path if not provided
            if yaml_path is None:
                # Assuming app/tools/market_holidays.py -> ../../data/holidays.yaml
                current_dir = Path(__file__).resolve().parent
                project_root = current_dir.parent.parent
                self.yaml_path = project_root / "data" / "holidays.yaml"
            else:
                self.yaml_path = Path(yaml_path)

            self._load_holidays()
            self._initialized = True

            logger.debug(
                f"✓ MarketHolidayChecker initialized with {len(self._holidays)} holidays"
            )

    def _load_holidays(self) -> None:
        """Loads holidays from the YAML file into memory."""
        try:
            if not self.yaml_path.exists():
                logger.error(f"Holidays file not found at: {self.yaml_path}")
                # We do not raise here to allow the app to run (maybe without holiday checks),
                # but in strict mode we might want to raise. For now, log error.
                # Per user rules: Critical errors -> Raise. Missing config file IS critical if we rely on it.
                raise FileNotFoundError(
                    f"Critical: Holidays configuration file missing at {self.yaml_path}"
                )

            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data: HolidayConfig = yaml.safe_load(f)

            if not data or "holidays" not in data:
                logger.warning("Holidays file is empty or missing 'holidays' key")
                return

            for date_string, holiday_name in data["holidays"].items():
                try:
                    date_object = datetime.datetime.strptime(
                        date_string, "%Y-%m-%d"
                    ).date()
                    self._holidays[date_object] = holiday_name
                except ValueError as error:
                    logger.warning(
                        "Invalid date format in holidays file for '%s': %s",
                        date_string,
                        error,
                    )

        except Exception as e:
            logger.critical(f"Failed to load market holidays: {e}")
            raise

    def is_holiday(self, date_check: datetime.date | str) -> bool:
        """
        Check if the given date is a public holiday.

        Args:
            date_check: datetime.date object or string in 'YYYY-MM-DD' format.

        Returns:
            bool: True if it is a holiday, False otherwise.
        """
        date_obj = self._parse_date(date_check)
        return date_obj in self._holidays

    def get_holiday_name(self, date_check: datetime.date | str) -> str | None:
        """
        Get the name of the holiday if the date is a holiday.

        Args:
            date_check: datetime.date object or string in 'YYYY-MM-DD' format.

        Returns:
            str | None: The name of the holiday, or None if not a holiday.
        """
        date_obj = self._parse_date(date_check)
        return self._holidays.get(date_obj)

    def _parse_date(self, date_check: datetime.date | str) -> datetime.date:
        """Helper to ensure we have a datetime.date object."""
        if isinstance(date_check, datetime.datetime):
            return date_check.date()
        if isinstance(date_check, datetime.date):
            return date_check
        if hasattr(date_check, "date") and callable(date_check.date):
            # Handles pandas Timestamp and similar objects
            return date_check.date()
        if isinstance(date_check, str):
            try:
                return datetime.datetime.strptime(date_check, "%Y-%m-%d").date()
            except ValueError:
                logger.error(f"Invalid date string format: {date_check}")
                raise ValueError(
                    f"Invalid date string provided: {date_check}. Expected YYYY-MM-DD."
                )
        raise TypeError(f"Unsupported date type: {type(date_check)}")


if __name__ == "__main__":
    # Simple test
    checker = MarketHolidayChecker()
    test_date = "2025-12-25"
    if checker.is_holiday(test_date):
        print(f"{test_date} is a holiday: {checker.get_holiday_name(test_date)}")
    else:
        print(f"{test_date} is NOT a holiday")
