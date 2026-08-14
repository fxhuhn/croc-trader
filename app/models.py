from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .mapping import mapper

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
    extras: dict = field(default_factory=dict)


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
class CrocContext:
    """Immutable high/low price context for Croc signals."""

    high: float
    low: float


@dataclass(frozen=True)
class SQNClassification:
    label: str
    color: str


@dataclass(frozen=True)
class BacktestMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    maximum_drawdown: float
    sharpe_ratio: float
    kelly_criterion: float
    expectancy: float
    system_quality_number: float
    average_win: float
    average_loss: float

    # Efficiency
    average_maximum_adverse_excursion: float
    average_maximum_favorable_excursion: float

    # Robustness
    risk_of_ruin: float

    # Comparison
    benchmark_return: float
    strategy_return: float

    # Kelly
    kelly_mean: float
    kelly_std: float
    kelly_safe: float

    # Advanced
    market_exposure_pct: float
    risk_adjusted_benchmark: float
    exposure_efficiency: float
    return_over_maximum_drawdown: float
    diversification_score: float


@dataclass(frozen=True)
class PortfolioMetrics:
    combined_mean_kelly: float
    safe_kelly_25: float
    correlation_fail_rate: float
    suggested_multiplier: float
    leveraged_max_drawdown: float
    max_concurrent_trades: int
    max_total_exposure: float
    # Unconstrained Simulation
    uncapped_multiplier: float
    uncapped_max_total_exposure: float
    uncapped_leveraged_max_drawdown: float
    max_trades_per_strategy: dict[str, int] = field(default_factory=dict)

    # Days at Max Concurrency
    max_concurrent_trades_days: int = 0
    max_trades_per_strategy_days: dict[str, int] = field(default_factory=dict)

    # Percentile-Based Sizing (Phase 1)
    percentile_95_concurrent_trades: float = 0.0
    percentile_95_trades_per_strategy: dict[str, float] = field(default_factory=dict)

    # Capacity Ratios (Phase 7)
    global_capacity_ratio: float = 1.0
    strategy_capacity_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class CrocSignal:
    """Represents an incoming trading signal from the Croc system."""

    symbol: str
    signal: str
    timeframe: str
    close: float
    high: float
    low: float
    wuk: float
    status: str
    kerze: str
    trend: str
    setter: str
    welle: str

    exchange: str | None = None
    full_symbol: str | None = None
    rsi: float | None = None
    sma_200: float | None = None
    sma_20: float | None = None
    wolke: str | None = None
    deluxe: str | None = None
    strategy_id: str | None = None
    reference: str | None = None
    # Factory default ensures correct timestamp at instantiation time
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Sanitizes data and sets defaults after initialization."""
        # String sanitization
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                setattr(self, key, value.strip())

        # Fallback logic
        if not self.exchange and self.full_symbol:
            self.exchange = self.full_symbol

        if self.exchange is None:
            self.exchange = self.full_symbol or self.symbol  # Last resort fallback

        mapped_exchange = mapper.get_exchange(self.symbol)

        if mapped_exchange:
            # Match found in JSON -> override
            self.exchange = mapped_exchange
        elif self.exchange == "BATS" or self.exchange is None:
            # No match in JSON, and BATS or empty -> set to unknown
            self.exchange = "UNKNOWN"

        # Timestamp parsing (handles string input)
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except (ValueError, TypeError):
                logger.warning(
                    "Could not parse timestamp '%s', using 'now'.",
                    self.timestamp,
                )
                self.timestamp = datetime.now(UTC)

        # Unique ID generation
        if self.reference is None:
            ts_str = self.timestamp.strftime("%Y%m%d%H%M%S")
            self.reference = f"{self.symbol}_{ts_str}"

    def to_db_row(self) -> dict[str, Any]:
        """Converts the object for SQLite storage."""
        d = asdict(self)
        # SQLite requires ISO strings for datetime
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class SignalStat:
    signal: str
    symbol: str
    timeframe: str
    level: str
    total: float
    win: float
    loss: float
    rejected: float
    win_rate: float
    loss_rate: float

    # Optional fields (may be empty according to CSV examples)
    wolke: str | None = None
    welle: str | None = None
    trend: str | None = None
    setter: str | None = None
    exchange: str | None = None

    # Metadata
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        try:
            self.total = float(self.total)
            self.win = float(self.win)
            self.loss = float(self.loss)
            self.rejected = float(self.rejected)
            self.win_rate = float(self.win_rate)
            self.loss_rate = float(self.loss_rate)

            self.wolke = self.wolke if self.wolke else None
            self.welle = self.welle if self.welle else None
            self.trend = self.trend if self.trend else None
            self.setter = self.setter if self.setter else None
            self.exchange = self.exchange if self.exchange else None

        except (ValueError, TypeError) as e:
            logger.warning("Import error for field: %s", e)

    def to_db_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["updated_at"] = self.updated_at.isoformat()
        d.pop("win_rate", None)
        d.pop("loss_rate", None)
        return d


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
            date_string = date_value.strftime("%Y-%m-%d")
        else:
            # Fallback for string or index-based date passed as column
            date_string = (
                str(date_value) if date_value else datetime.now().strftime("%Y-%m-%d")
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
    def from_tradingview(cls, symbol: str, row: dict[str, Any]) -> MarketPrice:
        """Factory method to create a MarketPrice from a TradingView row dictionary."""
        close_price = float(row.get("close", 0.0))
        if close_price < 0:
            raise ValueError(f"Negative close price for {symbol}")

        date_value = row.get("date") or row.get("datetime")
        if hasattr(date_value, "strftime"):
            date_string = date_value.strftime("%Y-%m-%d")
        else:
            date_string = (
                str(date_value)[:10]
                if date_value
                else datetime.now().strftime("%Y-%m-%d")
            )

        return cls(
            symbol=symbol,
            date=date_string,
            open=float(row.get("open", 0.0)),
            high=float(row.get("high", 0.0)),
            low=float(row.get("low", 0.0)),
            close=close_price,
            volume=int(row.get("volume", 0)),
            provider="tradingview",
            timeframe="1D",
        )

    def to_db_row(self) -> tuple:
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
