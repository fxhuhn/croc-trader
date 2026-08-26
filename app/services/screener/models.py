"""Typed data models for screener signal reporting and notification dispatch."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalReportItem:
    """Immutable representation of an individual strategy signal for reporting.

    Attributes:
        symbol: Uppercase ticker symbol (e.g. 'AAPL', 'SXRV.DE').
        action: Execution instruction (e.g. 'BUY LMT', 'BUY MOC', 'BUY MKT').
        entry_price: Target limit or entry calculation price.
        stop_loss: Optional stop loss price level (defaults to 0.0).
        target_profit: Optional target profit price level (defaults to 0.0).
        details: Additional strategy-specific metrics (e.g. 'ATR', 'Score', 'LOC').
    """

    symbol: str
    action: str
    entry_price: float
    stop_loss: float = 0.0
    target_profit: float = 0.0
    details: dict[str, str | float | int] = field(default_factory=dict)

    def to_row_dict(self) -> dict[str, str]:
        """Converts the signal report item into a clean tabular dictionary for display."""
        row: dict[str, str] = {
            "Symbol": self.symbol.upper(),
            "Action": self.action,
            "Entry": f"{self.entry_price:.2f}",
        }

        if self.stop_loss > 0.0:
            row["Stop"] = f"{self.stop_loss:.2f}"
        if self.target_profit > 0.0:
            row["TP"] = f"{self.target_profit:.2f}"

        for detail_key, detail_value in self.details.items():
            if isinstance(detail_value, float):
                row[detail_key] = f"{detail_value:.2f}"
            else:
                row[detail_key] = str(detail_value)

        return row


@dataclass(frozen=True)
class StrategyScreeningResult:
    """Immutable outcome of a single strategy screening execution.

    Attributes:
        strategy_name: Canonical identifier of the strategy.
        trading_date: Trading date string (YYYY-MM-DD).
        signals_count: Total number of signals identified and stored.
        report_items: List of formatted report items for notifications.
    """

    strategy_name: str
    trading_date: str
    signals_count: int
    report_items: list[SignalReportItem] = field(default_factory=list)
