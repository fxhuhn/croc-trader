
from typing import Literal, TypedDict


# --- Enums (Moved to app/const.py) ---
from .const import (
    TradeStatus as TradeStatus,
    ExitReason as ExitReason,
    TradeEventType as TradeEventType,
    EntryReason as EntryReason,
    Strategies as Strategies,
    TargetColumn as TargetColumn,
)

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

class MetricsOverview(TypedDict):
    """Mapping of metrics to their source (Database vs Runtime Calculation)."""
    metric_name: str
    source: Literal["Database", "Simulation"]
    description: str
