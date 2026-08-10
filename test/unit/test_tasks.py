"""Unit tests for app/tasks.py achieving 100% test coverage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask

from app.database.session import DatabaseSession
from app.tasks import (
    _clear_and_prewarm_cache,
    _enforce_backup_retention,
    _warm_single_route,
    run_cache_prewarm,
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


def test_run_market_data_update_in_app_context(tmp_path: Path) -> None:
    app = Flask(__name__)
    mock_bot = MagicMock()
    app.extensions["telegram"] = mock_bot
    dummy_db = tmp_path / "stocks.db"

    with (
        app.app_context(),
        patch("app.tasks.DatabaseSession", spec=DatabaseSession),
        patch("app.tasks.MarketDataUpdater"),
        patch("app.tasks.MarketQualityService"),
    ):
        run_market_data_update(db_path=dummy_db)


def test_run_market_data_update_config_error_fallback(tmp_path: Path) -> None:
    dummy_db = tmp_path / "stocks.db"
    with (
        patch("app.tasks.ConfigManager", side_effect=ValueError("Config fail")),
        patch("app.tasks.DatabaseSession", spec=DatabaseSession),
        patch("app.tasks.MarketDataUpdater"),
        patch("app.tasks.MarketQualityService"),
    ):
        run_market_data_update(db_path=dummy_db)


def test_run_market_data_update_exception(tmp_path: Path) -> None:
    dummy_db = tmp_path / "stocks.db"
    with patch("app.tasks.DatabaseSession", side_effect=RuntimeError("DB Error")):
        # Exception should be caught and logged
        run_market_data_update(db_path=dummy_db)


def test_run_daily_strategy_check() -> None:
    app = Flask(__name__)
    mock_engine = MagicMock()
    app.extensions["screener_engine"] = mock_engine

    run_daily_strategy_check(app)
    mock_engine.run_all.assert_called_once_with(days=0)


def test_run_daily_strategy_check_missing_engine() -> None:
    app = Flask(__name__)
    run_daily_strategy_check(app)


def test_run_daily_strategy_check_exception() -> None:
    app = Flask(__name__)
    mock_engine = MagicMock()
    mock_engine.run_all.side_effect = RuntimeError("Screener Error")
    app.extensions["screener_engine"] = mock_engine

    run_daily_strategy_check(app)


def test_run_cache_prewarm_success() -> None:
    app = Flask(__name__)
    with patch("app.tasks._clear_and_prewarm_cache") as mock_prewarm:
        run_cache_prewarm(app)
        mock_prewarm.assert_called_once_with(app)


def test_run_cache_prewarm_exception() -> None:
    app = Flask(__name__)
    with patch(
        "app.tasks._clear_and_prewarm_cache", side_effect=RuntimeError("Cache Error")
    ):
        run_cache_prewarm(app)


def test_run_order_generation_success() -> None:
    app = Flask(__name__)
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.return_value = "/path/to/orders.csv"
    app.extensions["trade_manager"] = mock_tm

    run_order_generation(app)
    mock_tm.generate_daily_orders.assert_called_once()


def test_run_order_generation_no_orders() -> None:
    app = Flask(__name__)
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.return_value = None
    app.extensions["trade_manager"] = mock_tm

    run_order_generation(app)


def test_run_order_generation_missing_tm() -> None:
    app = Flask(__name__)
    run_order_generation(app)


def test_run_order_generation_exception() -> None:
    app = Flask(__name__)
    mock_tm = MagicMock()
    mock_tm.generate_daily_orders.side_effect = RuntimeError("Order Error")
    app.extensions["trade_manager"] = mock_tm

    run_order_generation(app)


def test_run_db_maintenance_success(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    session = DatabaseSession(str(db_path))
    with session.connect() as conn:
        conn.execute("CREATE TABLE t (id INT)")

    run_db_maintenance(db_path)


def test_run_db_maintenance_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "non_existent.db"
    with patch("app.tasks.DatabaseSession", side_effect=RuntimeError("Connect fail")):
        run_db_maintenance(db_path)


def test_run_db_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    session = DatabaseSession(str(db_path))
    with session.connect() as conn:
        conn.execute("CREATE TABLE t (id INT)")

    # Run backup twice to test overwrite path (backup_file_path.exists())
    run_db_backup(db_path)
    run_db_backup(db_path)

    backup_dir = tmp_path / "backup"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("test.db.*"))
    assert len(backups) == 1


def test_run_db_backup_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with patch("app.tasks.DatabaseSession", side_effect=RuntimeError("Backup error")):
        run_db_backup(db_path)


def test_clear_and_prewarm_cache_prod_mode() -> None:
    app = Flask(__name__)
    app.debug = False
    app.testing = False

    with (
        patch.dict("sys.modules", {"pytest": None}),
        patch.dict("os.environ", {}, clear=True),
        patch("app.tasks.cache.clear") as mock_clear,
        patch("app.tasks._prewarm_target_routes") as mock_prewarm,
    ):
        _clear_and_prewarm_cache(app)
        mock_clear.assert_called_once()
        mock_prewarm.assert_called_once_with(app)


def test_warm_single_route_branches() -> None:
    mock_client = MagicMock()

    # Success 200
    mock_client.get.return_value.status_code = 200
    _warm_single_route(mock_client, "/analytics")

    # Non-200 status code
    mock_client.get.return_value.status_code = 500
    _warm_single_route(mock_client, "/analytics")

    # Exception during request
    mock_client.get.side_effect = RuntimeError("Network error")
    _warm_single_route(mock_client, "/analytics")


def test_enforce_backup_retention(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    # Create 7 backup files
    for i in range(1, 8):
        (backup_dir / f"stocks.db.2026-08-0{i}").touch()

    _enforce_backup_retention(backup_dir, "stocks.db", max_backups=5)

    remaining = list(backup_dir.glob("stocks.db.*"))
    assert len(remaining) == 5


def test_enforce_backup_retention_unlink_error(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    for i in range(1, 7):
        (backup_dir / f"stocks.db.2026-08-0{i}").touch()

    with patch.object(Path, "unlink", side_effect=PermissionError("Permission denied")):
        _enforce_backup_retention(backup_dir, "stocks.db", max_backups=5)
