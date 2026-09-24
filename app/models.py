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


class CandleIntegrityError(ValueError):
    """Raised when a price candle violates geometric invariants or contains anomalous values."""


@dataclass(frozen=True)
class CandleBar:
    """Immutable parameter container representing OHLCV values for candle validation."""

    symbol: str
    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int


@dataclass(frozen=True)
class MarketPrice:
    """Immutable representation of a daily price bar.

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

    ANOMALOUS_BODY_RATIO: float = 0.25

    @staticmethod
    def validate_candle(
        bar: CandleBar,
        epsilon: float = 1e-4,
    ) -> None:
        """Validates fundamental geometric and boundary invariants for a price candle."""
        if bar.close_price < 0:
            raise CandleIntegrityError(
                f"Negative close price for {bar.symbol}: {bar.close_price}"
            )
        if (
            bar.open_price <= 0
            or bar.high_price <= 0
            or bar.low_price <= 0
            or bar.close_price <= 0
        ):
            raise CandleIntegrityError(
                f"Non-positive price for {bar.symbol} on {bar.date}: "
                f"O={bar.open_price}, H={bar.high_price}, L={bar.low_price}, C={bar.close_price}"
            )
        if bar.volume < 0:
            raise CandleIntegrityError(
                f"Negative volume ({bar.volume}) for {bar.symbol} on {bar.date}"
            )
        if bar.high_price < bar.low_price - epsilon:
            raise CandleIntegrityError(
                f"High ({bar.high_price}) < Low ({bar.low_price}) for {bar.symbol} on {bar.date}"
            )
        max_open_close = max(bar.open_price, bar.close_price)
        if bar.high_price < max_open_close - epsilon:
            raise CandleIntegrityError(
                f"High ({bar.high_price}) < max(Open, Close) ({max_open_close}) for {bar.symbol} on {bar.date}"
            )
        min_open_close = min(bar.open_price, bar.close_price)
        if bar.low_price > min_open_close + epsilon:
            raise CandleIntegrityError(
                f"Low ({bar.low_price}) > min(Open, Close) ({min_open_close}) for {bar.symbol} on {bar.date}"
            )

    @classmethod
    def is_anomalous_wick(
        cls,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        spike_threshold: float = 0.12,
    ) -> bool:
        """Detects flash-crash or spike bad-print anomalies on lower or upper wicks.

        Identifies candles where an extreme wick exceeds spike_threshold (e.g. 12%)
        of the closing price, yet the real body is <= 25% of the wick (indicating
        an abnormal momentary needle and rapid rebound typical of bad prints).
        """
        if close_price <= 0:
            return False

        # Lower wick anomaly (flash crash / bad tick downwards)
        min_open_close = min(open_price, close_price)
        lower_wick = min_open_close - low_price
        if lower_wick > 0 and (lower_wick / close_price) >= spike_threshold:
            body = abs(close_price - open_price)
            if body <= cls.ANOMALOUS_BODY_RATIO * lower_wick:
                return True

        # Upper wick anomaly (bad print spike upwards)
        max_open_close = max(open_price, close_price)
        upper_wick = high_price - max_open_close
        if upper_wick > 0 and (upper_wick / close_price) >= spike_threshold:
            body = abs(close_price - open_price)
            if body <= cls.ANOMALOUS_BODY_RATIO * upper_wick:
                return True

        return False

    @classmethod
    def from_yahoo(cls, symbol: str, row: dict[str, Any]) -> MarketPrice:
        """Factory method to create a MarketPrice from a Yahoo row dictionary.

        Enforces candle integrity invariants (positive prices, High >= max(Open, Close),
        Low <= min(Open, Close), High >= Low, non-negative volume).
        """
        raw_close = row.get("close", 0.0)
        close_price = float(raw_close) if raw_close is not None else 0.0

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

        raw_open = row.get("open")
        raw_high = row.get("high")
        raw_low = row.get("low")

        open_price = float(raw_open) if raw_open is not None else close_price
        high_price = (
            float(raw_high) if raw_high is not None else max(open_price, close_price)
        )
        low_price = (
            float(raw_low) if raw_low is not None else min(open_price, close_price)
        )
        volume = int(float(row.get("volume", 0) or 0))

        bar = CandleBar(
            symbol=symbol,
            date=date_string,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
        )
        cls.validate_candle(bar)

        return cls(
            symbol=symbol,
            date=date_string,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )

    @classmethod
    def from_tradingview(cls, symbol: str, row: Mapping[str, object]) -> MarketPrice:
        """Factory method to create a MarketPrice from a TradingView row dictionary."""
        raw_close = row.get("close") or 0.0
        close_price = float(str(raw_close))

        date_value = row.get("date") or row.get("datetime")
        if hasattr(date_value, "strftime"):
            date_string = str(date_value.strftime("%Y-%m-%d"))
        else:
            date_string = (
                str(date_value)[:10]
                if date_value
                else datetime.now(UTC).strftime("%Y-%m-%d")
            )

        raw_open = row.get("open")
        raw_high = row.get("high")
        raw_low = row.get("low")

        open_price = float(str(raw_open)) if raw_open is not None else close_price
        high_price = (
            float(str(raw_high))
            if raw_high is not None
            else max(open_price, close_price)
        )
        low_price = (
            float(str(raw_low)) if raw_low is not None else min(open_price, close_price)
        )
        volume = int(float(str(row.get("volume") or 0)))

        bar = CandleBar(
            symbol=symbol,
            date=date_string,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
        )
        cls.validate_candle(bar)

        return cls(
            symbol=symbol,
            date=date_string,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
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
