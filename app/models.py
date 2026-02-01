import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .mapping import mapper

logger = logging.getLogger(__name__)


@dataclass
class CrocSignal:
    """
    Repräsentiert ein Handelssignal.
    """

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
    # Factory default sorgt für korrekten Zeitpunkt bei Instanziierung
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Bereinigt Daten und setzt Defaults nach der Initialisierung."""
        # String Bereinigung
        for key, value in self.__dict__.items():
            if isinstance(value, str):
                setattr(self, key, value.strip())

        # Fallback Logic
        if not self.exchange and self.full_symbol:
            self.exchange = self.full_symbol

        if self.exchange is None:
            self.exchange = self.full_symbol or self.symbol  # Letzter Fallback

        mapped_exchange = mapper.get_exchange(self.symbol)

        if mapped_exchange:
            # Wir haben einen Treffer im JSON -> Überschreiben
            self.exchange = mapped_exchange
        elif self.exchange == "BATS" or self.exchange is None:
            # Kein Treffer im JSON, aber BATS oder Leer -> Fallback Versuche
            self.exchange = "UNKNOWN"

        # Timestamp Parsing (falls String übergeben wurde)
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except (ValueError, TypeError):
                logger.warning(
                    f"Konnte Timestamp '{self.timestamp}' nicht parsen, nutze 'now'."
                )
                self.timestamp = datetime.now(timezone.utc)

        # Unique ID Generierung
        if self.reference is None:
            ts_str = self.timestamp.strftime("%Y%m%d%H%M%S")
            self.reference = f"{self.symbol}_{ts_str}"

    def to_db_row(self) -> dict[str, Any]:
        """Konvertiert das Objekt für die SQLite Speicherung."""
        d = asdict(self)
        # SQLite braucht ISO Strings für Datetime
        d["timestamp"] = self.timestamp.isoformat()
        return d


# app/models.py
# ... (deine existierenden Imports und CrocSignal) ...


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

    # Optionale Felder (können leer sein laut CSV Beispiel)
    wolke: str | None = None
    welle: str | None = None
    trend: str | None = None
    setter: str | None = None
    exchange: str | None = None

    # Metadaten
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

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
            logger.warning(f"import fehler bei {e}")

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
    def from_yahoo(cls, symbol: str, row: dict[str, Any]) -> "MarketPrice":
        """
        Factory method to create a MarketPrice from a Yahoo row dictionary.
        Validation logic (e.g., non-negative prices) implies here.
        """
        # Close must be valid, others can be 0 if missing.
        c = float(row.get("close", 0.0))
        if c < 0:
            raise ValueError(f"Negative close price for {symbol}")

        # Ensure date format is correct (Yahoo often gives Timestamp)
        d_val = row.get("date")
        if hasattr(d_val, "strftime"):
             d_str = d_val.strftime("%Y-%m-%d")
        else:
            # Fallback for string or index-based date passed as column
            d_str = str(d_val) if d_val else datetime.now().strftime("%Y-%m-%d")

        return cls(
            symbol=symbol,
            date=d_str,
            open=float(row.get("open", 0.0)),
            high=float(row.get("high", 0.0)),
            low=float(row.get("low", 0.0)),
            close=c,
            volume=int(row.get("volume", 0)),
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
