import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..database.repositories.signal import SignalRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankingVerificationResult:
    """Contains verification results for a specific attribute key."""

    attribute_key: str
    missing_values: list[str]
    available_values: list[str]


def verify_ranking_system(
    ranking_yaml_path: Path, signal_repository: SignalRepository
) -> None:
    """Loads ranking configuration and verifies attributes against the database."""
    if not ranking_yaml_path.exists():
        logger.warning(
            "Ranking check: Ranking file %s not found.",
            ranking_yaml_path,
        )
        return

    try:
        with open(ranking_yaml_path, encoding="utf-8") as f:
            ranking_data = yaml.safe_load(f) or {}

        database_attributes = signal_repository.get_unique_signal_attributes()
        results = check_ranking_attributes(ranking_data, database_attributes)

        for result in results:
            if result.missing_values:
                logger.warning(
                    "Ranking check WARNING: Values for '%s' from %s missing in DB: %s",
                    result.attribute_key,
                    ranking_yaml_path.name,
                    ", ".join(result.missing_values),
                )
            if result.available_values:
                logger.info(
                    "Ranking check OK: %d '%s' values from %s found in DB.",
                    len(result.available_values),
                    result.attribute_key,
                    ranking_yaml_path.name,
                )
    except Exception as e:
        logger.error("Ranking check error for %s: %s", ranking_yaml_path.name, e)


def check_ranking_attributes(
    ranking_data: object,
    database_attributes: dict[str, set[str]],
) -> list[RankingVerificationResult]:
    """Checks if values defined in the ranking configuration exist in the database.

    This is a pure logic function (Functional Core). It normalizes keys
    case-insensitively and performs case-insensitive value matching to be
    robust.
    """
    check_keys = [
        "Signal",
        "Status",
        "Kerze",
        "Wolke",
        "Trend",
        "Setter",
        "Welle",
    ]

    results = []

    for key in check_keys:
        if isinstance(ranking_data, list):
            required_values = _collect_values_from_list(ranking_data, key)
        elif isinstance(ranking_data, dict):
            required_values = _collect_values_from_dict(ranking_data, key)
        else:
            required_values = set()

        if not required_values:
            continue

        db_values = database_attributes.get(key, set())
        db_values_lower = {v.lower() for v in db_values}

        available = []
        missing = []

        for req_val in required_values:
            if key == "Signal" and "+" in req_val:
                parts = [p.strip() for p in req_val.split("+") if p.strip()]
                all_parts_present = bool(parts) and all(
                    p.lower() in db_values_lower for p in parts
                )
                if all_parts_present or req_val.lower() in db_values_lower:
                    available.append(req_val)
                else:
                    missing.append(req_val)
            elif req_val.lower() in db_values_lower:
                # Find the original cased value from the database if possible
                original_cased = next(
                    (v for v in db_values if v.lower() == req_val.lower()),
                    req_val,
                )
                available.append(original_cased)
            else:
                missing.append(req_val)

        results.append(
            RankingVerificationResult(
                attribute_key=key,
                missing_values=sorted(missing),
                available_values=sorted(available),
            )
        )

    return results


def _collect_values_from_list(ranking_list: list[Any], key: str) -> set[str]:
    """Extracts values for a key from a list of configuration items."""
    values = set()
    for item in ranking_list:
        if not isinstance(item, dict):
            continue
        val = _get_case_insensitive_value(item, key)
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str:
            continue
        if key == "Signal" and " (" in val_str:
            val_str = val_str.split(" (")[0].strip()
        values.add(val_str)
    return values


def _collect_values_from_dict(ranking_dict: dict[str, Any], key: str) -> set[str]:
    """Extracts values for a key from a dictionary of configuration items."""
    values = set()
    if key == "Signal":
        for signal_name, rules in ranking_dict.items():
            if not isinstance(rules, dict):
                continue
            if "Score" in rules or "SQN" in rules:
                val_str = str(signal_name).strip()
                if " (" in val_str:
                    val_str = val_str.split(" (")[0].strip()
                values.add(val_str)
    else:
        for rules in ranking_dict.values():
            if not isinstance(rules, dict):
                continue
            val = _get_case_insensitive_value(rules, key)
            if val is not None:
                val_str = str(val).strip()
                if val_str:
                    values.add(val_str)
    return values


def _get_case_insensitive_value(item: dict[str, Any], key: str) -> Any | None:
    """Gets a value from a dictionary using case-insensitive key lookup."""
    target_key = key.lower()
    for item_key, item_value in item.items():
        if item_key.lower() == target_key:
            return item_value
    return None
