"""US Stock Market Holidays Updater.

Queries NYSE/NASDAQ calendar definitions from the exchange_calendars package,
maps both regular and ad-hoc holidays, sorts them chronologically, and writes
the updated dictionary directly to the project's holidays definition file.

Usage:
    python script/update_holidays.py [--start YEAR] [--end YEAR]

Side Effects:
    Overwrites the 'data/holidays.yaml' file.
"""

import argparse
import datetime
import logging
from pathlib import Path
from typing import TypedDict

import exchange_calendars
import yaml

# Set up logging format and levels
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class HolidaysYamlStructure(TypedDict):
    """Structure for holidays.yaml file contents."""

    holidays: dict[str, str]


def build_holidays_mapping(
    start_year: int,
    end_year: int,
    regular_holidays_mapping: dict[datetime.date, str],
    adhoc_holidays_list: list[datetime.date],
) -> dict[str, str]:
    """Functional Core: Combines regular and ad-hoc holidays into a sorted dictionary.

    Args:
        start_year: The first year to include.
        end_year: The last year to include.
        regular_holidays_mapping: Dict mapping dates to regular holiday names.
        adhoc_holidays_list: List of ad-hoc holiday dates.

    Returns:
        dict[str, str]: Sorted dictionary mapping date strings to holiday names.
    """
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    combined_holidays: dict[str, str] = {}

    # Add regular holidays
    for holiday_date, holiday_name in regular_holidays_mapping.items():
        if start_date <= holiday_date <= end_date:
            combined_holidays[str(holiday_date)] = holiday_name

    # Add ad-hoc holidays
    for unprocessed_holiday_date in adhoc_holidays_list:
        holiday_date = (
            unprocessed_holiday_date.date()
            if isinstance(unprocessed_holiday_date, datetime.datetime)
            else unprocessed_holiday_date
        )
        if start_date <= holiday_date <= end_date:
            date_string = str(holiday_date)
            if date_string not in combined_holidays:
                combined_holidays[date_string] = "Ad-hoc Market Holiday"

    # Sort keys chronologically
    sorted_date_strings = sorted(combined_holidays.keys())
    return {date_str: combined_holidays[date_str] for date_str in sorted_date_strings}


def update_market_holidays_file(
    start_year: int,
    end_year: int,
    destination_file_path: Path,
) -> None:
    """Imperative Shell: Loads data from exchange_calendars and updates YAML file.

    Args:
        start_year: The first year to include.
        end_year: The last year to include.
        destination_file_path: Path to the holidays.yaml output file.
    """
    logger.info("Loading NYSE/NASDAQ trading calendar (XNYS) to update holidays...")

    # Load NYSE/NASDAQ trading calendar
    trading_calendar = exchange_calendars.get_calendar("XNYS")

    start_date_string = f"{start_year}-01-01"
    end_date_string = f"{end_year}-12-31"

    # Fetch regular holidays
    regular_holidays_data = trading_calendar.regular_holidays.holidays(
        start=start_date_string, end=end_date_string, return_name=True
    )

    # Map regular holidays to python datetime.date -> str mapping
    regular_holidays_mapping: dict[datetime.date, str] = {
        pandas_timestamp.date(): str(holiday_name)
        for pandas_timestamp, holiday_name in regular_holidays_data.items()
    }

    # Extract raw ad-hoc holidays list
    adhoc_holidays_list: list[datetime.date] = [
        pandas_timestamp.date() for pandas_timestamp in trading_calendar.adhoc_holidays
    ]

    # Combine using functional core
    sorted_holidays_mapping = build_holidays_mapping(
        start_year,
        end_year,
        regular_holidays_mapping,
        adhoc_holidays_list,
    )

    # Write the YAML structure
    yaml_output_data: HolidaysYamlStructure = {"holidays": sorted_holidays_mapping}

    logger.info(
        f"Writing {len(sorted_holidays_mapping)} market holidays for "
        f"{start_year} to {end_year} into {destination_file_path}..."
    )

    # Save to file
    with open(destination_file_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            f"# US Stock Market Holidays {start_year}-{end_year} "
            "(Generated via script/update_holidays.py)\n"
        )
        yaml.safe_dump(
            yaml_output_data,
            file_handle,
            default_flow_style=False,
            allow_unicode=True,
        )

    logger.info("✓ Market holidays YAML file successfully updated.")


def main() -> None:
    """Orchestrator for the CLI arguments and file updating process."""
    command_line_argument_parser = argparse.ArgumentParser(
        description="Update US Stock Market holidays in data/holidays.yaml."
    )
    command_line_argument_parser.add_argument(
        "--start",
        type=int,
        default=2025,
        help="First year to include in the holidays database (default: 2025)",
    )
    command_line_argument_parser.add_argument(
        "--end",
        type=int,
        default=2028,
        help="Last year to include in the holidays database (default: 2028)",
    )

    parsed_arguments = command_line_argument_parser.parse_args()

    project_root_directory = Path(__file__).resolve().parent.parent
    holidays_file_path = project_root_directory / "data" / "holidays.yaml"

    update_market_holidays_file(
        parsed_arguments.start,
        parsed_arguments.end,
        holidays_file_path,
    )


if __name__ == "__main__":
    main()
