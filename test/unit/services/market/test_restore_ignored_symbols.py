"""Unit tests for automated weekly check and restoration of ignored symbols."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from app.services.market.quality import MarketQualityService
from app.tasks import run_ignored_symbols_check


def test_restore_ignored_symbols_empty_list() -> None:
    """Verifies that restore_ignored_symbols returns empty list when no symbols are ignored."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_updater.repo = mock_repo
    mock_repo.get_ignored_symbols.return_value = set()

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    restored = service.restore_ignored_symbols()

    assert restored == []
    mock_updater.provider.fetch_batch_raw.assert_not_called()
    mock_telegram.send_message.assert_not_called()


def test_restore_ignored_symbols_success() -> None:
    """Verifies that symbols with valid closing prices are removed from blacklist and notified."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_updater.repo = mock_repo
    mock_repo.get_ignored_symbols.return_value = {"CROX", "VIX"}

    dummy_df = pd.DataFrame({"close": [100.0, 102.0]})
    mock_updater.provider.fetch_batch_raw.return_value = (MagicMock(), [])
    mock_updater.provider.extract_symbol_data.return_value = dummy_df

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    restored = service.restore_ignored_symbols()

    assert sorted(restored) == ["CROX", "VIX"]
    assert mock_repo.remove_ignored_symbol.call_count == 2
    mock_repo.remove_ignored_symbol.assert_any_call("CROX")
    mock_repo.remove_ignored_symbol.assert_any_call("VIX")

    mock_telegram.send_message.assert_called_once()
    message = mock_telegram.send_message.call_args[0][0]
    assert "Ignored Symbols Check" in message
    assert "2 Symbole wieder aktiv" in message
    assert "CROX" in message
    assert "VIX" in message


def test_restore_ignored_symbols_still_failing() -> None:
    """Verifies that symbols that still fail to download or have empty data remain blacklisted."""
    mock_updater = MagicMock()
    mock_repo = MagicMock()
    mock_updater.repo = mock_repo
    mock_repo.get_ignored_symbols.return_value = {"DELISTED1", "EMPTY2"}

    # DELISTED1 fails in batch, EMPTY2 returns empty dataframe
    mock_updater.provider.fetch_batch_raw.return_value = (MagicMock(), ["DELISTED1"])
    mock_updater.provider.extract_symbol_data.return_value = pd.DataFrame()

    mock_telegram = MagicMock()
    service = MarketQualityService(updater=mock_updater, telegram_bot=mock_telegram)

    restored = service.restore_ignored_symbols()

    assert restored == []
    mock_repo.remove_ignored_symbol.assert_not_called()
    mock_telegram.send_message.assert_not_called()


@patch("app.tasks.MarketQualityService")
@patch("app.tasks.MarketDataUpdater")
@patch("app.tasks.DatabaseSession")
def test_run_ignored_symbols_check_task(
    mock_session: MagicMock,
    mock_updater_class: MagicMock,
    mock_quality_class: MagicMock,
    tmp_path: Path,
) -> None:
    """Verifies that run_ignored_symbols_check orchestrates service and returns restored list."""
    mock_quality_instance = MagicMock()
    mock_quality_instance.restore_ignored_symbols.return_value = ["CROX"]
    mock_quality_class.return_value = mock_quality_instance

    mock_telegram = MagicMock()
    db_path = tmp_path / "stocks.db"

    result = run_ignored_symbols_check(db_path=db_path, telegram_bot=mock_telegram)

    assert result == ["CROX"]
    mock_quality_instance.restore_ignored_symbols.assert_called_once()


@patch("app.tasks.DatabaseSession", side_effect=RuntimeError("Database lock"))
def test_run_ignored_symbols_check_task_handles_exception(
    mock_session: MagicMock,
    tmp_path: Path,
) -> None:
    """Verifies that run_ignored_symbols_check catches exceptions and returns empty list safely."""
    db_path = tmp_path / "stocks.db"
    result = run_ignored_symbols_check(db_path=db_path)
    assert result == []
