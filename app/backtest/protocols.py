"""Protocols for backtesting strategies."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyProtocol(Protocol):
    """Pluggable strategy interface used by BacktestEngine."""

    @property
    def name(self) -> str: ...

    @property
    def lookback_days(self) -> int: ...

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame: ...

    def generate_signals(
        self,
        features: pd.DataFrame,
        signal_date: datetime | pd.Timestamp,
    ) -> list[str]: ...
