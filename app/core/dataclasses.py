from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class CrocSignal:
    symbol: str
    signal: str
    timeframe: str
    timestamp: datetime

    close: float
    high: float
    low: float
    wuk: float

    status: str
    kerze: str
    trend: str
    setter: str
    welle: str
    wolke: Optional[str] = None

    strategy_id: Optional[str] = None
    reference: Optional[str] = None

    def __post_init__(self):
        if self.reference is None:
            self.reference = f"{self.symbol}_{self.timestamp.isoformat()}"

    @classmethod
    def from_dict(cls, d: dict) -> "CrocSignal":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
