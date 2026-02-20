from typing import TypedDict


class YahooRow(TypedDict):
    """
    Type definition for a raw row from Yahoo Finance DataFrame.
    Used for type hints in basic ingestion logic.
    """

    date: str  # Usually string index or column
    open: float
    high: float
    low: float
    close: float
    volume: int
