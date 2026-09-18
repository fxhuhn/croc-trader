"""Integration smoke test suite for application routes and cache pre-warming.

Verifies that all registered EOD UI dashboard and screener routes render
without 500 Internal Server Errors, template syntax anomalies, or unhandled
exceptions under realistic runtime conditions.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.tasks import _prewarm_target_routes

TARGET_SMOKE_ROUTES: tuple[str, ...] = (
    "/analytics",
    "/analytics/monthly-matrix",
    "/trades",
    "/trades/dip-buyer",
    "/trades/turnover",
    "/trades/ndx-momentum",
    "/trades/twopercent",
    "/trades/tgim",
    "/trades/bridge-scout",
    "/trades/bounce-bandit",
    "/screener",
    "/screener/dip-buyer",
    "/screener/turnover",
    "/screener/twopercent",
    "/screener/ndx-momentum",
    "/screener/tgim",
    "/screener/bridge-scout",
    "/screener/bounce-bandit",
)


@pytest.fixture(name="smoke_app")
def fixture_smoke_app() -> Generator[Flask, None, None]:
    """Provides an isolated Flask application instance configured for smoke testing."""
    application_instance = create_app()
    application_instance.config["TESTING"] = True
    application_instance.config["APP_CONFIG"] = MagicMock()
    application_instance.config["APP_CONFIG"].app.security.whitelist = ["127.0.0.1"]
    application_instance.config["APP_CONFIG"].app.security.mode = "block"
    application_instance.config["APP_CONFIG"].get_db_path.return_value = ":memory:"

    yield application_instance


@pytest.fixture(name="smoke_client")
def fixture_smoke_client(smoke_app: Flask) -> FlaskClient:
    """Provides a test client bound to the smoke test application."""
    return smoke_app.test_client()


@pytest.mark.parametrize("route_path", TARGET_SMOKE_ROUTES)
def test_all_prewarm_target_routes_render_successfully(
    smoke_client: FlaskClient, route_path: str
) -> None:
    """Verifies that each registered pre-warm target route returns HTTP 200 without crashes."""
    response = smoke_client.get(route_path)

    assert response.status_code == 200, (
        f"Smoke test failed for route '{route_path}' with HTTP {response.status_code}. "
        f"Response body snippet: {response.data[:300]!r}"
    )
    assert b"500 Internal Server Error" not in response.data
    assert b"Interner Serverfehler" not in response.data


def test_prewarm_cache_task_executes_without_errors(smoke_app: Flask) -> None:
    """Verifies that the EOD cache pre-warming routine runs completely without unhandled errors."""
    with smoke_app.app_context():
        _prewarm_target_routes(smoke_app)
