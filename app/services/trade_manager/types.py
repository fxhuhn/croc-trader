"""Types for the Trade Manager component."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeTransition:
    """Immutable record of database updates and logging metadata from strategy logic."""

    updates: dict[str, object]
    reason: str
    message: str
