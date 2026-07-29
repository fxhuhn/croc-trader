"""Unit tests for generic backfill engine and strategy dispatcher."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from app.services.backfill_engine import (
    STRATEGY_MAP,
    run_strategy_backfill,
)
from app.services.bounce_bandit_backfill import run_bounce_bandit_backfill
from app.services.bridge_scout_backfill import run_bridge_scout_backfill
from app.services.tgim_backfill import run_tgim_backfill


def test_strategy_map_contains_expected_strategies() -> None:
    """Verifies all supported strategies are registered in STRATEGY_MAP."""
    assert "tgim" in STRATEGY_MAP
    assert "bridge_scout" in STRATEGY_MAP
    assert "bounce_bandit" in STRATEGY_MAP


def test_run_strategy_backfill_invalid_strategy_raises_value_error() -> None:
    """Verifies ValueError is raised for unregistered strategy names."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()

    with pytest.raises(ValueError, match="Unknown strategy for backfill"):
        run_strategy_backfill(
            stocks_session=mock_stocks_session,
            signals_session=mock_signals_session,
            strategy_name="non_existent_strategy",
        )


@patch("app.services.backfill_engine.run_generic_backfill")
def test_run_strategy_backfill_dispatches_correctly(
    mock_run_generic: MagicMock,
) -> None:
    """Verifies run_strategy_backfill resolves strategy parameters and calls run_generic_backfill."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()
    mock_run_generic.return_value = {"status": "ok"}

    result = run_strategy_backfill(
        stocks_session=mock_stocks_session,
        signals_session=mock_signals_session,
        strategy_name="bridge_scout",
        start_date="2025-01-01",
        budget=5000.0,
    )

    assert result == {"status": "ok"}
    mock_run_generic.assert_called_once()
    kwargs = mock_run_generic.call_args.kwargs
    assert kwargs["strategy_name"] == "bridge_scout"
    assert kwargs["lookback_days"] == 60
    assert kwargs["budget"] == 5000.0


def test_deprecated_wrappers_issue_deprecation_warnings() -> None:
    """Verifies legacy backfill functions emit DeprecationWarning when called."""
    mock_stocks_session = MagicMock()
    mock_signals_session = MagicMock()

    with patch("app.services.backfill_engine.run_generic_backfill") as mock_run:
        mock_run.return_value = {}

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_bridge_scout_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_bridge_scout_backfill is deprecated" in str(record[0].message)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_bounce_bandit_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_bounce_bandit_backfill is deprecated" in str(record[0].message)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run_tgim_backfill(mock_stocks_session, mock_signals_session)
            assert len(record) == 1
            assert issubclass(record[0].category, DeprecationWarning)
            assert "run_tgim_backfill is deprecated" in str(record[0].message)
