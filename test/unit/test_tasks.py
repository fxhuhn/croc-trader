# filename: test_tasks.py
from pathlib import Path
from unittest.mock import patch

from app.database.session import DatabaseSession
from app.tasks import run_market_data_update


def test_run_market_data_update_outside_app_context(tmp_path: Path) -> None:
    """Verifies run_market_data_update executes safely outside of a Flask app context."""
    dummy_db = tmp_path / "stocks.db"

    with (
        patch("app.tasks.DatabaseSession", spec=DatabaseSession),
        patch("app.tasks.MarketDataUpdater") as mock_updater_class,
        patch("app.tasks.MarketQualityService") as mock_quality_class,
        patch("app.tasks.ConfigManager") as mock_config_class,
    ):
        mock_config_instance = mock_config_class.return_value
        mock_config_instance.app.telegram.token = "dummy_token"
        mock_config_instance.app.telegram.chat_id = "12345"
        mock_config_instance.app.telegram.enabled = False

        # Execute function without any active Flask request or application context
        run_market_data_update(db_path=dummy_db)

        mock_updater_class.return_value.run_update.assert_called_once_with(
            full_reload=False
        )
        mock_quality_class.return_value.perform_gap_check.assert_called_once()
        mock_quality_class.return_value.check_last_trading_day_completeness.assert_called_once()
