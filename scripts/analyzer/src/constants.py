"""
Constants for Trading Analysis System

Defines all column names used across the trading system.
"""

from enum import StrEnum


class ColumnNames(StrEnum):
    """Column name constants for DataFrames."""

    # ==================== PRICE DATA ====================
    HIGH = "high"
    LOW = "low"
    OPEN = "open"
    CLOSE = "close"
    VOLUME = "volume"
    TIME = "time"

    # ==================== METADATA ====================
    FILE_NAME = "file_name"
    SYMBOL = "symbol"

    # --- NEW: TIMEFRAME & EXCHANGE support ---
    TIMEFRAME = "timeframe"
    EXCHANGE = "exchange"

    # ==================== SIGNAL INDICATORS (from mappings) ====================
    DELUXE = "deluxe"
    WELLE = "welle"
    TREND = "trend"
    SETTER = "setter"
    WOLKE = "wolke"
    KERZE = "kerze"
    STATUS = "status"
    MSI_DAY = "msi_day"
    MSI_WEEK = "msi_week"
    MSI_MONTH = "msi_month"

    # ==================== CLOUD LINES (hardcoded) ====================
    WOLKE_LINIE_GRUEN = "Wolke Linie Grün"
    WOLKE_LINIE_PINK = "Wolke Linie Pink"

    # ==================== RANGE ANALYSIS ====================
    LONG_RANGE = "Long_Range"
    LONG_AKTIV = "Long_aktiv"
    LONG_TP1 = "Long_TP1"
    LONG_TP2 = "Long_TP2"
    LONG_TP3 = "Long_TP3"
    LONG_TP = "Long_TP"

    SHORT_RANGE = "Short_Range"
    SHORT_AKTIV = "Short_aktiv"
    SHORT_TP1 = "Short_TP1"
    SHORT_TP2 = "Short_TP2"
    SHORT_TP3 = "Short_TP3"
    SHORT_TP = "Short_TP"

    # ==================== INDIACATORS ====================
    RSI = "RSI"


class BullSignals(StrEnum):
    """Bull (Long) signal column names from renaming."""

    BULL_LINE = "bull_line"
    BULL_WOLKE = "bull_wolke"
    BULL_ROT = "bull_rot"
    BULL_HELL_GRUEN = "bull_hell_gruen"
    BULL_ORANGE = "bull_orange"
    BULL_BLAU = "bull_blau"
    BULL_GRAU = "bull_grau"
    BULL_GRAU_KLEIN = "bull_grau_klein"
    BULL_PLUS = "bull_plus"
    BULL_GRUEN = "bull_gruen"
    BULL_BB = "bull_bb"
    BULL_SCHWARZ = "bull_schwarz"
    BULL_GRABBER = "bull_grabber"
    BULL_PINK = "bull_pink"
    BULL_1 = "bull_1"
    BULL_2 = "bull_2"
    BULL_3 = "bull_3"
    BULL_4 = "bull_4"
    BULL_5 = "bull_5"
    BULL_6 = "bull_6"
    BULL_7 = "bull_7"
    BULL_8 = "bull_8"
    BULL_9 = "bull_9"
    BULL_10 = "bull_10"
    BULL_11 = "bull_11"
    BULL_12 = "bull_12"
    BULL_13 = "bull_13"
    BULL_14 = "bull_14"
    BULL_15 = "bull_15"


class BearSignals(StrEnum):
    """Bear (Short) signal column names from renaming."""

    BEAR_LINE = "bear_line"
    BEAR_ROT = "bear_rot"
    BEAR_HELL_GRUEN = "bear_hell_gruen"
    BEAR_GRAU = "bear_grau"
    BEAR_GRAU_KLEIN = "bear_grau_klein"
    BEAR_BLAU = "bear_blau"
    BEAR_PLUS = "bear_plus"
    BEAR_ORANGE = "bear_orange"
    BEAR_WOLKE = "bear_wolke"
    BEAR_GRUEN = "bear_gruen"
    BEAR_BB = "bear_bb"
    BEAR_GRABBER = "bear_grabber"
    BEAR_PINK = "bear_pink"
    BEAR_1 = "bear_1"
    BEAR_2 = "bear_2"
    BEAR_3 = "bear_3"
    BEAR_4 = "bear_4"
    BEAR_5 = "bear_5"
    BEAR_6 = "bear_6"
    BEAR_7 = "bear_7"
    BEAR_8 = "bear_8"
    BEAR_9 = "bear_9"
    BEAR_10 = "bear_10"
    BEAR_11 = "bear_11"
    BEAR_12 = "bear_12"
    BEAR_13 = "bear_13"
    BEAR_14 = "bear_14"
    BEAR_15 = "bear_15"


class SignalColors(StrEnum):
    """Color values used in mappings."""

    BLACK = "black"
    RED = "red"
    DARKRED = "darkred"
    GREEN = "green"
    DARKGREEN = "darkgreen"
    YELLOW = "yellow"
    SCHWARZ = "schwarz"
    ROT = "rot"
    GRUEN = "gruen"


class ConfigKeys(StrEnum):
    """Configuration key constants."""

    SIGNALS = "signals"
    MAPPINGS = "mappings"
    RENAMING = "renaming"
    FILE_PREFIXES = "file_prefixes"
    INPUT_PATTERN = "input_pattern"
    OUTPUT_DIR = "output_dir"
    MIN_TRADES = "min_trades"

    # NEW: Shift Configuration
    SHIFT_COLUMNS = "shift_columns"
    SHIFT_DEPTH = "shift_depth"


# Default configuration values
DEFAULT_MIN_TRADES = 3
DEFAULT_INPUT_PATTERN = "*d*.csv"
DEFAULT_OUTPUT_DIR = "output"
