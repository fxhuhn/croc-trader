"""Base strategy helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def lookback_days(self) -> int:
        return 400

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def get_rebalance_schedule(self, data: pd.DataFrame) -> pd.DataFrame: ...

    @abstractmethod
    def generate_signals(
        self, features: pd.DataFrame, signal_date: pd.Timestamp
    ) -> list[str]: ...

    def _date_index(self, df: pd.DataFrame, start_date: str) -> pd.DatetimeIndex:
        try:
            date_level = df.index.get_level_values("date").unique()
        except (KeyError, AttributeError):
            if isinstance(df.index, pd.MultiIndex):
                date_level = df.index.get_level_values(0).unique()
            else:
                date_level = df.index.unique()

        dates = pd.to_datetime(date_level).sort_values()
        return dates[dates >= pd.to_datetime(start_date)]
