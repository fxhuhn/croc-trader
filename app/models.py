from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import OrderAction, OrderType, TimeInForce

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeParams:
    """Immutable container for strategy-specific state parameters."""

    stop_loss: float
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    take_profit_3: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderLeg:
    """Immutable representation of a single order leg (entry or exit)."""

    action: OrderAction
    type: OrderType
    price: Decimal
    quantity: int | None = None
    time_in_force: TimeInForce = "DAY"


@dataclass
class Order:
    id: str
    symbol: str
    quantity: int
    mode: str
    entry: OrderLeg | None = None
    exits: list[OrderLeg] = field(default_factory=list)
    last_status: str = "PendingSubmit"
    last_update: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class MarketPrice:
    """
    Immutable representation of a daily price bar.
    Strictly typed and validated upon creation via factory.
    """

    symbol: str
    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int
    provider: str = "yahoo"
    timeframe: str = "1D"

    @classmethod
    def from_yahoo(cls, symbol: str, row: dict[str, Any]) -> MarketPrice:
        """
        Factory method to create a MarketPrice from a Yahoo row dictionary.
        Validation logic (e.g., non-negative prices) implies here.
        """
        # Close must be valid, others can be 0 if missing.
        close_price = float(row.get("close", 0.0))
        if close_price < 0:
            raise ValueError(f"Negative close price for {symbol}")

        # Ensure date format is correct (Yahoo often gives Timestamp)
        date_value = row.get("date")
        if hasattr(date_value, "strftime"):
            date_string = str(date_value.strftime("%Y-%m-%d"))  # type: ignore[union-attr]
        else:
            # Fallback for string or index-based date passed as column
            date_string = (
                str(date_value)
                if date_value
                else datetime.now(UTC).strftime("%Y-%m-%d")
            )

        return cls(
            symbol=symbol,
            date=date_string,
            open=float(row.get("open", 0.0)),
            high=float(row.get("high", 0.0)),
            low=float(row.get("low", 0.0)),
            close=close_price,
            volume=int(row.get("volume", 0)),
        )

    @classmethod
    def from_tradingview(cls, symbol: str, row: Mapping[str, object]) -> MarketPrice:
        """Factory method to create a MarketPrice from a TradingView row dictionary."""
        close_price = float(str(row.get("close") or 0.0))
        if close_price < 0:
            raise ValueError(f"Negative close price for {symbol}")

        date_value = row.get("date") or row.get("datetime")
        if hasattr(date_value, "strftime"):
            date_string = str(date_value.strftime("%Y-%m-%d"))
        else:
            date_string = (
                str(date_value)[:10]
                if date_value
                else datetime.now(UTC).strftime("%Y-%m-%d")
            )

        return cls(
            symbol=symbol,
            date=date_string,
            open=float(str(row.get("open") or 0.0)),
            high=float(str(row.get("high") or 0.0)),
            low=float(str(row.get("low") or 0.0)),
            close=close_price,
            volume=int(float(str(row.get("volume") or 0))),
            provider="tradingview",
            timeframe="1D",
        )

    def to_db_row(self) -> tuple[object, ...]:
        """Optimized for executemany (tuple based)."""
        return (
            self.symbol,
            self.date,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.provider,
            self.timeframe,
        )


@dataclass(frozen=True)
class FuturesPrice:
    """Immutable representation of a 30-minute futures price bar.

    Timestamps are in exchange time (US Eastern) as returned by TradingView.
    """

    symbol: str  # Internal base symbol, e.g. "MNQ"
    contract: str  # Full TradingView contract, e.g. "MNQU2026"
    bar_time: str  # ISO 8601 exchange time, e.g. "2026-08-29T14:00:00"
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "30min"
    provider: str = "tradingview"

    @classmethod
    def from_tradingview(
        cls,
        symbol: str,
        contract: str,
        row: Mapping[str, object],
    ) -> FuturesPrice:
        """Factory method to create a FuturesPrice from a TradingView row dictionary.

        Args:
            symbol: Internal base symbol (e.g. "MNQ").
            contract: Full contract identifier (e.g. "MNQU2026").
            row: Dictionary record from TvDatafeed containing OHLCV and datetime.

        Raises:
            ValueError: If the close price is negative.
        """
        close_price = float(str(row.get("close") or 0.0))
        if close_price < 0:
            raise ValueError(f"Negative close price for {contract}")

        datetime_value = row.get("datetime") or row.get("date")
        if hasattr(datetime_value, "strftime"):
            bar_time_string = str(datetime_value.strftime("%Y-%m-%dT%H:%M:%S"))
        else:
            bar_time_string = str(datetime_value) if datetime_value else ""

        return cls(
            symbol=symbol,
            contract=contract,
            bar_time=bar_time_string,
            open=float(str(row.get("open") or 0.0)),
            high=float(str(row.get("high") or 0.0)),
            low=float(str(row.get("low") or 0.0)),
            close=close_price,
            volume=int(float(str(row.get("volume") or 0))),
        )

    def to_db_row(self) -> tuple[object, ...]:
        """Serializes to a tuple for executemany insertion."""
        return (
            self.symbol,
            self.contract,
            self.bar_time,
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            self.timeframe,
            self.provider,
        )
