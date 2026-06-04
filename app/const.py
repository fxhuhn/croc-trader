"""Domain constants and enumerations for trade management."""

from enum import StrEnum


class Strategies(StrEnum):
    """Canonical identifiers for all trading strategies."""
    # Core Strategies
    DipBuyer = "dip_buyer"
    TwoPercent = "two_percent"

    # Croc Variants
    CrocSetup = "croc_setup"
    SplitTarget = "split_target"
    HoldTarget = "hold_target"

    # Turnover Variants
    TurnOverTiming = "turnover_timing"
    TurnOverTiming_10 = "turnover_timing_1.0"
    TurnOverTiming_05 = "turnover_timing_0.5"

    # NDX Momentum
    NDXMomentum = "ndx_momentum"


STRATEGY_ALIASES = {
    # DipBuyer
    "dipbuyer": Strategies.DipBuyer,
    "dip_buyer": Strategies.DipBuyer,
    "dip buyer": Strategies.DipBuyer,
    # Turnover
    "turnovertiming": Strategies.TurnOverTiming,
    "turnover_timing": Strategies.TurnOverTiming,
    "turnover timing": Strategies.TurnOverTiming,
    "turnovertiming_1.0": Strategies.TurnOverTiming_10,
    "turnover_timing_1.0": Strategies.TurnOverTiming_10,
    "turnovertiming_0.5": Strategies.TurnOverTiming_05,
    "turnover_timing_0.5": Strategies.TurnOverTiming_05,
    # Hold Target / Croc
    "holdtarget": Strategies.HoldTarget,
    "hold_target": Strategies.HoldTarget,
    "hold target": Strategies.HoldTarget,
    "croc_holdtp3": Strategies.HoldTarget,
    "crocholdtp3": Strategies.HoldTarget,
    "croc_tp3": Strategies.HoldTarget,
    "croc_hold": Strategies.HoldTarget,
    # Split Target
    "splittarget": Strategies.SplitTarget,
    "split_target": Strategies.SplitTarget,
    "split target": Strategies.SplitTarget,
    "croc split": Strategies.SplitTarget,
    "croc_split": Strategies.SplitTarget,
    "croc_split (tp1/3)": Strategies.SplitTarget,
    "splits": Strategies.SplitTarget,
    # Two Percent
    "twopercent": Strategies.TwoPercent,
    "two_percent": Strategies.TwoPercent,
    "two percent": Strategies.TwoPercent,
    "twopercentstrategy": Strategies.TwoPercent,
    "two_percent_strategy": Strategies.TwoPercent,
    # NDX Momentum
    "ndx_momentum": Strategies.NDXMomentum,
    "ndxmomatum": Strategies.NDXMomentum,
    "ndx momentum": Strategies.NDXMomentum,
    # Turnover Variants (Explicit user inputs)
    "turnover 1.0": Strategies.TurnOverTiming_10,
    "turnover 0.5": Strategies.TurnOverTiming_05,
    "crocsetup": Strategies.CrocSetup,
    "croc_setup": Strategies.CrocSetup,
    "croc setup": Strategies.CrocSetup,
}

# --- Trade Management Enums ---


class TradeStatus(StrEnum):
    """Lifecycle states of a trade record."""
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    MISSED = "MISSED"  # Entry not filled (Limit not reached)
    INVALID = "INVALID"  # Setup invalidated before entry (e.g. Stop Validation)
    SKIPPED = "SKIPPED"  # Rejected by Portfolio Manager


class ExitReason(StrEnum):
    """Reasons why a trade was closed."""
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
    GREEN_SEQUENCE = "GREEN_SEQUENCE"


class TradeEventType(StrEnum):
    """Types of events logged in the trade lifecycle."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"

    UPDATE = "UPDATE"
    SL_UPDATE = "SL_UPDATE"
    TP_UPDATE = "TP_UPDATE"
    STATUS_UPDATE = "STATUS_UPDATE"

    PARTIAL_EXIT = "PARTIAL_EXIT"

    INFO = "INFO"


class EntryReason(StrEnum):
    """Specific conditions that triggered a trade entry."""
    GAP_UP = "GAP UP (Stop)"
    BREAKOUT = "BREAKOUT (Stop)"
    GAP_DOWN = "GAP DOWN (Stop)"
    BREAKDOWN = "BREAKDOWN (Stop)"


class TargetColumn(StrEnum):
    """Database column names for various take-profit targets."""
    TARGET_PRICE = "target_price"
    TP3 = "tp3"
    TAKE_PROFIT_3 = "take_profit_3"
    TP1 = "tp1"
    TAKE_PROFIT_1 = "take_profit_1"


class IndexAliases(StrEnum):
    """Short-name aliases for major market indices."""
    SPX = "SPX"
    NDX = "NDX"
    DOW = "DOW"
    RUS = "RUS"
    NO_INDEX = "No Index"
