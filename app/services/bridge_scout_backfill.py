"""Service for executing historical trade backfills for the Bridge Scout strategy."""

import warnings

from ..database.session import DatabaseSession
from .backfill_engine import run_strategy_backfill


def run_bridge_scout_backfill(
    stocks_session: DatabaseSession,
    signals_session: DatabaseSession,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    budget: float = 10000.0,
    clear_existing: bool = True,
) -> dict[str, object]:
    """Executes a backfill simulation for Bridge Scout strategy from start_date to end_date.

    .. deprecated::
        Use :func:`app.services.backfill_engine.run_strategy_backfill` instead.

    Args:
        stocks_session: DatabaseSession for market historical prices.
        signals_session: DatabaseSession for trade persistence.
        start_date: Start date string (YYYY-MM-DD), defaults to "2025-01-01".
        end_date: End date string (YYYY-MM-DD), defaults to current date.
        budget: Capital allocation budget per trade.
        clear_existing: If True, clears existing bridge_scout trades before running.

    Returns:
        dict[str, object]: Backfill summary metrics and list of closed trades.
    """
    warnings.warn(
        "run_bridge_scout_backfill is deprecated, use app.services.backfill_engine instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return run_strategy_backfill(
        stocks_session=stocks_session,
        signals_session=signals_session,
        strategy_name="bridge_scout",
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        clear_existing=clear_existing,
    )
