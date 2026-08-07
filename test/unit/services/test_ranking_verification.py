"""Unit tests for RankingVerificationService in app/services/ranking_verification.py."""

import logging
from pathlib import Path

import pytest

from app.database.repositories.signal import SignalRepository
from app.database.session import DatabaseSession
from app.services.ranking_verification import (
    check_ranking_attributes,
    verify_ranking_system,
)


def test_check_ranking_attributes_list_structure() -> None:
    ranking_data = [
        {"Signal": "DipBuyer (Setup Score > 80)", "Kerze": "Hammer", "Trend": "Up"},
        {"Signal": "TwoPercent", "Kerze": "Doji", "Trend": "Down"},
    ]
    database_attributes = {
        "Signal": {"DipBuyer", "TwoPercent", "TGIM"},
        "Kerze": {"Hammer", "Doji", "ShootingStar"},
        "Trend": {"Up", "Down"},
    }

    results = check_ranking_attributes(ranking_data, database_attributes)
    result_map = {res.attribute_key: res for res in results}

    assert "Signal" in result_map
    assert "DipBuyer" in result_map["Signal"].available_values
    assert "TwoPercent" in result_map["Signal"].available_values
    assert not result_map["Signal"].missing_values


def test_check_ranking_attributes_dict_structure_and_missing_values() -> None:
    ranking_data = {
        "CrocSetup (RSI < 30)": {"Score": 95, "Kerze": "Engulfing"},
        "TGIM": {"SQN": 2.5, "Kerze": "UnknownCandle"},
    }
    database_attributes = {
        "Signal": {"CrocSetup", "TGIM"},
        "Kerze": {"Engulfing", "Hammer"},
    }

    results = check_ranking_attributes(ranking_data, database_attributes)
    result_map = {res.attribute_key: res for res in results}

    assert "CrocSetup" in result_map["Signal"].available_values
    assert "TGIM" in result_map["Signal"].available_values
    assert "Engulfing" in result_map["Kerze"].available_values
    assert "UnknownCandle" in result_map["Kerze"].missing_values


def test_verify_ranking_system_workflow(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    yaml_file = tmp_path / "ranking_2026.yaml"
    yaml_file.write_text(
        """
- Signal: BounceBandit
  Kerze: Marubozu
        """,
        encoding="utf-8",
    )

    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))
    repo = SignalRepository(session)
    repo.init_schema()

    with caplog.at_level(logging.INFO):
        verify_ranking_system(yaml_file, repo)

    assert "Ranking check WARNING" in caplog.text or "Ranking check OK" in caplog.text


def test_verify_ranking_system_file_not_found(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing_file = tmp_path / "non_existent.yaml"
    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))
    repo = SignalRepository(session)

    with caplog.at_level(logging.WARNING):
        verify_ranking_system(missing_file, repo)

    assert "not found" in caplog.text


def test_verify_ranking_system_exception_handling(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corrupt_file = tmp_path / "corrupt.yaml"
    corrupt_file.write_text("invalid: yaml: [", encoding="utf-8")
    db_file = tmp_path / "test.db"
    session = DatabaseSession(str(db_file))
    repo = SignalRepository(session)

    with caplog.at_level(logging.ERROR):
        verify_ranking_system(corrupt_file, repo)

    assert "Ranking check error" in caplog.text


def test_check_ranking_attributes_edge_cases() -> None:
    # Non-dict and invalid structures
    results_invalid_type = check_ranking_attributes(12345, {})  # type: ignore[arg-type]
    assert results_invalid_type == []

    list_with_invalid_items = [
        "not_a_dict",
        {"Signal": ""},
        {"Signal": "  "},
        {"Status": None},
    ]
    results_list = check_ranking_attributes(list_with_invalid_items, {})
    assert results_list == []

    dict_with_invalid_rules = {
        "ValidSignal": "not_a_dict_rules",
        "SignalWithScore": {"Score": 100},
        "OtherSignal": {"Status": "Active"},
    }
    results_dict = check_ranking_attributes(dict_with_invalid_rules, {})
    result_map = {res.attribute_key: res for res in results_dict}
    assert "SignalWithScore" in result_map["Signal"].missing_values
