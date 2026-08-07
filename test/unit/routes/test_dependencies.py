"""Unit tests for app/routes/views/dependencies.py covering database path resolution, view services, and chart generation."""

from pathlib import Path
from unittest.mock import MagicMock

from flask import Flask

from app.routes.views.dependencies import (
    _get_database_path,
    _get_screener_view_service,
    _get_signal_repository,
    _get_trade_view_service,
    generate_donut_chart,
    generate_sparkline,
)


def test_get_database_path_memory_and_real_path() -> None:
    app = Flask(__name__)
    mock_config = MagicMock()
    mock_config.get_db_path.side_effect = lambda name: (
        ":memory:" if name == "memory_key" else "/tmp/test.db"
    )
    app.config["APP_CONFIG"] = mock_config

    with app.app_context():
        mem_path = _get_database_path("memory_key")
        assert mem_path == Path(":memory:")

        real_path = _get_database_path("real_key")
        assert real_path == Path("/tmp/test.db").resolve()


def test_get_repositories_and_view_services() -> None:
    app = Flask(__name__)
    mock_config = MagicMock()
    mock_config.get_db_path.return_value = ":memory:"
    app.config["APP_CONFIG"] = mock_config

    with app.app_context():
        signal_repo = _get_signal_repository()
        assert signal_repo is not None

        screener_service = _get_screener_view_service()
        assert screener_service is not None

        trade_service = _get_trade_view_service()
        assert trade_service is not None


def test_generate_sparkline_positive_and_negative() -> None:
    dates = ["2026-08-01", "2026-08-02"]
    prices = [100.0, 105.0]

    pos_html = generate_sparkline(dates, prices, is_positive=True)
    assert "plotly" in pos_html.lower()

    neg_html = generate_sparkline(dates, prices, is_positive=False)
    assert "plotly" in neg_html.lower()


def test_generate_donut_chart() -> None:
    labels = ["DipBuyer", "TurnoverTiming"]
    values = [60.0, 40.0]
    colors = ["#10b981", "#ef4444"]

    donut_html = generate_donut_chart(labels, values, colors)
    assert "plotly" in donut_html.lower()
