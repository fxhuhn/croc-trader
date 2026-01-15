from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional

OrderAction = Literal["BUY", "SELL"]
OrderType = Literal["LMT", "MKT", "LOC", "STP", "MOC"]  # MOC hinzugefügt
TimeInForce = Literal["DAY", "GTC"]
TradeStatus = Literal["CREATED", "ACTIVE", "CLOSED", "MISSED"]


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
    # Entry ist optional für reine Management-Orders (Active Trades)
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
