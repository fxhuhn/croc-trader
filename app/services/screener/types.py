from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class BaseScreenerResult:
    """
    Basisdaten, die jeder Screener liefern MUSS.
    """

    date: str
    symbol: str
    exchange: str
    timeframe: str
    close: float

    def to_dict(self) -> dict[str, Any]:
        """Hilfsmethode für Datenbank-Kompatibilität."""
        return asdict(self)


@dataclass(slots=True)
class DipBuyerResult(BaseScreenerResult):
    """
    Spezifische Daten für die DipBuyer Strategie.
    """

    high: float
    atr_r3: float
    setup_score: float
    entry_limit: float
    atr5: float


# Beispiel für eine andere Strategie (nur zur Veranschaulichung der Flexibilität)
@dataclass(slots=True)
class TurnoverResult(BaseScreenerResult):
    volume: float
    turnover_usd: float
    relative_volume: float
