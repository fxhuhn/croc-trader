"""Tests for deprecation warnings on Croc Setup components and TradingView webhook."""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.database.repositories.signal import SignalRepository
from app.routes.views.trades import _aggregate_croc_signals
from app.services.screener.strategies.croc_setup import CrocSetupStrategy
from app.services.screener.view_service import ScreenerViewService, _is_croc_strategy
from app.services.trade_manager.strategies.hold_target import HoldTargetStrategy


@pytest.fixture
def mock_repo_class() -> Generator[MagicMock, None, None]:
    with patch("app.routes.api.SignalRepository") as mock:
        yield mock


@pytest.fixture
def mock_db_session_class() -> Generator[MagicMock, None, None]:
    with patch("app.routes.api.DatabaseSession") as mock:
        yield mock


@pytest.fixture
def app_instance(
    mock_repo_class: MagicMock, mock_db_session_class: MagicMock
) -> Generator[Flask, None, None]:
    with patch("app.tools.symbol_lists.ExchangeSymbol._refresh_data"):
        app = create_app()
        app.config.update({"TESTING": True})
        app.config["APP_CONFIG"] = MagicMock()
        app.config["APP_CONFIG"].app.security.whitelist = ["127.0.0.1"]
        app.config["APP_CONFIG"].app.security.mode = "block"
        yield app


@pytest.fixture
def client(app_instance: Flask) -> FlaskClient:
    return app_instance.test_client()


def test_webhook_emits_deprecation_warning(
    client: FlaskClient,
    mock_repo_class: MagicMock,
    mock_db_session_class: MagicMock,
) -> None:
    """Verifies that calling /webhook triggers a DeprecationWarning."""
    mock_instance = mock_repo_class.return_value
    mock_instance.save_signal.return_value = 1

    payload = {"symbol": "AAPL", "signal": "BUY", "price": 150.0}

    with pytest.warns(
        DeprecationWarning, match="TradingView webhook endpoint /webhook is deprecated"
    ):
        response = client.post(
            "/webhook", json=payload, environ_base={"REMOTE_ADDR": "127.0.0.1"}
        )

    assert response.status_code == 201


def test_hold_target_strategy_init_emits_deprecation_warning() -> None:
    """Verifies that HoldTargetStrategy.__init__ triggers a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="HoldTargetStrategy is deprecated"):
        strategy = HoldTargetStrategy()
    assert strategy.name == "hold_target"


def test_croc_setup_strategy_init_emits_deprecation_warning() -> None:
    """Verifies that CrocSetupStrategy.__init__ triggers a DeprecationWarning."""
    mock_trade_repo = MagicMock()
    mock_data_provider = MagicMock()
    mock_signal_repo = MagicMock()
    mock_telegram_bot = MagicMock()

    mock_stat = MagicMock()
    mock_stat.st_size = 100
    with (
        patch(
            "app.services.screener.strategies.croc_setup.settings.get_path",
            return_value=Path("mock.yaml"),
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.stat", return_value=mock_stat),
        patch("builtins.open", mock_open(read_data="ranking_2026: []")),
        patch("app.services.screener.strategies.croc_setup.ExchangeSymbol"),
    ):
        with pytest.warns(DeprecationWarning, match="CrocSetupStrategy is deprecated"):
            strategy = CrocSetupStrategy(
                trade_repository=mock_trade_repo,
                data_provider=mock_data_provider,
                signal_repository=mock_signal_repo,
                telegram_bot=mock_telegram_bot,
            )
        assert strategy.name == "croc_setup"


def test_signal_repository_save_signal_emits_deprecation_warning() -> None:
    """Verifies that SignalRepository.save_signal triggers a DeprecationWarning."""
    mock_session = MagicMock()
    repo = SignalRepository(mock_session)

    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 42
    repo.execute = MagicMock(return_value=mock_cursor)  # type: ignore[method-assign]

    with pytest.warns(
        DeprecationWarning,
        match="Persisting signals to 'croc' table via SignalRepository is deprecated",
    ):
        signal_id = repo.save_signal({"symbol": "AAPL", "signal": "BUY"})

    assert signal_id == 42


def test_view_service_croc_functions_emit_deprecation_warning() -> None:
    """Verifies that Croc-specific candidate fetch methods in view_service trigger DeprecationWarnings."""
    assert _is_croc_strategy("croc_setup", "croc_setup", "croc_setup") is True
    assert _is_croc_strategy("dip_buyer", "dip_buyer", "dip_buyer") is False

    mock_repo = MagicMock()
    mock_repo.get_trade_candidates.return_value = []
    service = ScreenerViewService(mock_repo)

    with pytest.warns(
        DeprecationWarning,
        match="'_fetch_croc_candidates' is deprecated",
    ):
        candidates = service._fetch_croc_candidates(10)
        assert candidates == []

    with pytest.warns(
        DeprecationWarning,
        match="'_filter_new_candidates' is deprecated",
    ):
        dest: list[dict[str, Any]] = []
        service._filter_new_candidates([], set(), dest)
        assert dest == []


def test_view_trades_croc_emits_deprecation_warning(
    client: FlaskClient,
) -> None:
    """Verifies that accessing /trades/croc triggers a DeprecationWarning."""
    with patch("app.routes.views.trades._get_trade_view_service") as mock_trade_service:
        mock_service_instance = mock_trade_service.return_value
        mock_service_instance.get_trades.return_value = []
        mock_service_instance.get_portfolio_summary.return_value = {
            "invested": 0.0,
            "open_pnl": 0.0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
        }
        mock_service_instance.get_closed_summary.return_value = {
            "count": 0,
            "average_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
        }
        mock_service_instance.get_index_stats.return_value = {}
        mock_service_instance.group_trades_by_symbol.return_value = []
        mock_service_instance.group_trades_history.return_value = []

        with pytest.warns(
            DeprecationWarning,
            match="'view_trades_croc' is deprecated",
        ):
            response = client.get("/trades/croc")

        assert response.status_code == 200


def test_aggregate_croc_signals_emits_deprecation_warning() -> None:
    """Verifies that calling _aggregate_croc_signals triggers a DeprecationWarning."""
    mock_service = MagicMock()
    with pytest.warns(
        DeprecationWarning,
        match="'_aggregate_croc_signals' is deprecated",
    ):
        result = _aggregate_croc_signals([], mock_service)

    assert "Breakout (L20)" in result
