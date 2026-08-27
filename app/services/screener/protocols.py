from typing import Protocol, runtime_checkable

from ...const import Strategies


@runtime_checkable
class StrategyProtocol(Protocol):
    @property
    def name(self) -> str | Strategies: ...

    def run(self, days: int = 0, analysis_date: str | None = None) -> int: ...
