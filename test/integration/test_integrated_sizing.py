"""Stub for removed legacy backtester test_integrated_sizing.py."""

import pytest


@pytest.mark.skip(
    reason="Legacy backtester removed; sizing is tested via TradeManager & PortfolioManager."
)
def test_integrated_sizing_flow() -> None:
    """Stub test function to prevent IDE test-runner errors for stale test targets."""
    pass
