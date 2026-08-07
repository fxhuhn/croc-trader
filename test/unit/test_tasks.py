from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

from app.database.session import DatabaseSession
from app.tasks import (
    _enforce_backup_retention,
    _warm_single_route,
    run_daily_strategy_check,
    run_db_backup,
    run_db_maintenance,
    run_market_data_update,
    run_order_generation,
)


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

        run_market_data_update(db_path=dummy_db)

        mock_updater_class.return_value.run_update.assert_called_once_with(
            full_reload=False
        )
        mock_quality_class.return_value.perform_gap_check.assert_called_once()
        mock_quality_class.return_value.check_last_trading_day_completeness.assert_called_once()


def test_run_daily_strategy_check() -> None:
    app = Flask(__name__)
    mock_engine = MagicMock()
    app.extensions["screener_engine"] = mock_engine

    run_daily_strategy_check(app)
    mock_engine.run_all.assert_called_once_with(days=0)


def test_run_daily_strategy_check_missing_engine() -> None:
    app = Flask(__name__)
    # screener_engine missing
    run_daily_strategy_check(app)


def test_run_order_generation() -> None:
    app = Flask(__name__)
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.return_value = "/path/to/orders.csv"
    app.extensions["trade_manager"] = mock_tm

    run_order_generation(app)
    mock_tm.generate_daily_orders.assert_called_once()


def test_run_order_generation_missing_tm() -> None:
    app = Flask(__name__)
    run_order_generation(app)


def test_run_db_maintenance(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    session = DatabaseSession(str(db_path))
    with session.connect() as conn:
        conn.execute("CREATE TABLE t (id INT)")

    run_db_maintenance(db_path)


def test_run_db_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    session = DatabaseSession(str(db_path))
    with session.connect() as conn:
        conn.execute("CREATE TABLE t (id INT)")

    run_db_backup(db_path)
    backup_dir = tmp_path / "backup"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("test.db.*"))
    assert len(backups) == 1


def test_enforce_backup_retention(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    # Create 7 backup files
    for i in range(1, 8):
        (backup_dir / f"stocks.db.2026-08-0{i}").touch()

    _enforce_backup_retention(backup_dir, "stocks.db", max_backups=5)

    remaining = list(backup_dir.glob("stocks.db.*"))
    assert len(remaining) == 5


def test_warm_single_route() -> None:
    mock_client = MagicMock()
    mock_client.get.return_value.status_code = 200
    _warm_single_route(mock_client, "/analytics")
    mock_client.get.assert_called_once_with("/analytics")


def test_prewarm_target_routes_includes_monthly_matrix() -> None:
    app = Flask(__name__)
    with patch("app.tasks._warm_single_route") as mock_warm:
        from app.tasks import _prewarm_target_routes

        _prewarm_target_routes(app)
        mock_warm.assert_any_call(
            mock_warm.call_args_list[0][0][0], "/analytics/monthly-matrix"
        )
