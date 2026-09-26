"""Unit tests for scheduler configuration in app/services/setup.py."""

from unittest.mock import MagicMock, patch

from app.services.setup import configure_scheduler


def test_configure_scheduler_order_generation_schedule() -> None:
    """Verifies that order_generation is scheduled for mon-fri."""
    mock_app = MagicMock()
    mock_tm = MagicMock()
    mock_app.extensions = {"trade_manager": mock_tm}
    mock_config = MagicMock()
    mock_config.get_db_path.return_value = "data/stocks.db"

    with patch("app.services.setup.BackgroundScheduler") as mock_scheduler_class:
        mock_scheduler = MagicMock()
        mock_scheduler_class.return_value = mock_scheduler

        configure_scheduler(mock_app, mock_config)

        order_gen_calls = [
            call
            for call in mock_scheduler.add_job.call_args_list
            if call.kwargs.get("id") == "order_generation"
        ]
        assert len(order_gen_calls) == 1
        trigger = order_gen_calls[0].kwargs.get("trigger")
        assert trigger is not None
        # CronTrigger fields: day_of_week should cover Monday to Friday (0-4 or mon-fri)
        day_of_week_field = str(trigger.fields[4])
        assert "sat" not in day_of_week_field
        assert "sun" not in day_of_week_field
        # Verify str(trigger) contains mon-fri
        assert "mon-fri" in str(trigger)
