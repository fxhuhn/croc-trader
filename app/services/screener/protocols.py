from typing import Protocol, runtime_checkable


@runtime_checkable
class StrategyProtocol(Protocol):
    name: str

    def run(self, days: int = 0, analysis_date: str | None = None) -> int: ...
