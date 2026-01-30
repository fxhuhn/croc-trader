from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal, Dict, Any, Optional, List, TypedDict

# --- Enums (originally from database/enums.py) ---

class TradeStatus(StrEnum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    MISSED = "MISSED"
    INVALID = "INVALID"

class ExitReason(StrEnum):
    # Profit Exits
    TAKE_PROFIT = "TAKE_PROFIT"
    LOC_PROFIT = "LOC_PROFIT"
    TARGET_HIT = "TARGET_HIT"

    # Stop / Time Exits
    STOP_LOSS = "STOP_LOSS"
    TIME_STOP = "TIME_STOP"

    # Validity Exits
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    
    # Other
    MANUAL = "MANUAL"

class TradeEventType(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    
    UPDATE = "UPDATE"
    SL_UPDATE = "SL_UPDATE"
    TP_UPDATE = "TP_UPDATE"
    STATUS_UPDATE = "STATUS_UPDATE"
    
    PARTIAL_EXIT = "PARTIAL_EXIT"
    
    INFO = "INFO"

# --- New Enums (consolidated from strategy usage) ---

class EntryReason(StrEnum):
    GAP_UP = "GAP UP (Stop)"
    BREAKOUT = "BREAKOUT (Stop)"

# --- Types (originally from trade_manager/types.py) ---

OrderAction = Literal["BUY", "SELL"]
OrderType = Literal["LMT", "MKT", "LOC", "STP", "MOC"]
TimeInForce = Literal["DAY", "GTC"]

class TradeData(TypedDict, total=False):
    """
    Strikt typisierte Struktur für das Trade-Objekt aus der DB.
    'total=False' erlaubt partielle Updates, aber wir sollten für Reads vorsichtig sein.
    Wichtige Felder für Strategien:
    """
    id: str
    symbol: str
    strategy: str
    status: str # TradeStatus
    
    # Entry
    entry_price: float | None
    entry_date: str | None
    initial_size: int | None
    current_size: int | None
    budget: float | None
    
    # Management
    current_stop_loss: float | None
    current_target: float | None
    
    # Context
    signal_context: str | None # JSON String
    
    # Result
    exit_price: float | None
    exit_date: str | None
    exit_reason: str | None
    realized_pnl: float | None


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
    tp_1: float | None = None
    tp_2: float | None = None
    tp_3: float | None = None
    extras: dict = field(default_factory=dict)


@dataclass
class OrderLeg:
    action: OrderAction
    type: OrderType
    price: float
    qty: int | None = None
    tif: TimeInForce = "DAY"


@dataclass
class Order:
    id: str
    symbol: str
    qty: int
    mode: str
    entry: OrderLeg | None = None
    exits: list[OrderLeg] = field(default_factory=list)
    last_status: str = "PendingSubmit"
    last_update: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CrocContext:
    high: float
    low: float
