from typing import TypedDict


class YahooRow(TypedDict):
    """Type definition for a raw row from Yahoo Finance DataFrame (Deprecated).

    .. deprecated:: 1.0.0
       Unused TypedDict; market updater processes DataFrames directly.
    """

    date: str  # Usually string index or column
    open: float
    high: float
    low: float
    close: float
    volume: int
