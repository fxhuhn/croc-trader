from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional

OrderAction = Literal["BUY", "SELL"]
OrderType = Literal["LMT", "MKT", "LOC", "STP", "MOC"]
TimeInForce = Literal["DAY", "GTC"]
TradeStatus = Literal["CREATED", "ACTIVE", "CLOSED", "MISSED"]


@dataclass
class TradeParams:
    """
    General container for strategy-specific state parameters.
    For DipBuyer:
    - stop_loss: 0.0 (No hard stop)
    - tp_1: The Fixed ATR Target
    - extras['threshold_loc']: The Previous Day High (Dynamic limit)
    """

    stop_loss: float
    tp_1: Optional[float] = None
    tp_2: Optional[float] = None
    tp_3: Optional[float] = None
    extras: dict = field(default_factory=dict)


@dataclass
class OrderLeg:
    action: OrderAction
    type: OrderType
    price: float
    qty: Optional[int] = None
    tif: TimeInForce = "DAY"


@dataclass
class Order:
    id: str
    symbol: str
    qty: int
    mode: str
    entry: Optional[OrderLeg] = None
    exits: List[OrderLeg] = field(default_factory=list)
    last_status: str = "PendingSubmit"
    last_update: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CrocContext:
    high: float
    low: float
