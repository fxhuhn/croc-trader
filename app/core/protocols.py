from typing import Any, Optional, Protocol


class QueueProtocol(Protocol):
    def put(self, item: dict[str, Any]) -> None: ...
    def get(self, timeout: float | None = None) -> dict[str, Any]: ...
    def empty(self) -> bool: ...


class SignalRepository(Protocol):
    def save_signal(self, data: dict[str, Any]) -> None: ...

    def get_signals(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        signal: Optional[str] = None,
        day: Optional[str] = None,
        limit: int = 500,
    ) -> list[dict]: ...

    def get_distinct(self, column: str) -> list[str]: ...

    # trade tracking
    def toggle_trade_tracking(
        self,
        symbol: str,
        timestamp: str,
        signal: str,
        timeframe: str | None = None,
        signal_timestamp: str | None = None,
    ) -> dict: ...

    def get_trade(self, trade_id: int) -> Optional[dict]: ...
    def update_trade(self, trade_id: int, data: dict) -> bool: ...
    def get_all_trades(self, limit: int = 500) -> list[dict]: ...
