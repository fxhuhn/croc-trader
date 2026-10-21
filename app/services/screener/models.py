from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


class SingleSymbolStatus(StrEnum):
    """Execution status for single symbol technical analysis."""

    QUALIFIED = "QUALIFIED"
    INTRINSIC_PASSED_RELATIVE_UNVERIFIED = "INTRINSIC_PASSED_RELATIVE_UNVERIFIED"
    REJECTED = "REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class SingleSymbolAnalysisResult(TypedDict, total=False):
    """Standardized result structure for single symbol diagnosis and prechecks."""

    symbol: str
    strategy: str
    as_of_market_date: str
    upcoming_eod_date: str
    status: str
    overall_passed: bool
    requires_universe_ranking: bool
    calendar_active_today: bool | None
    macro_regime_passed: bool | None
    intrinsic_filters: dict[str, bool]
    metric_values: dict[str, float | int | str | None]
    relative_ranking: dict[str, Any] | None
    rejection_reasons: list[str]
    # Backwards-compatibility fields for legacy callers
    checks: dict[str, bool]
    values: dict[str, Any]
    technical_filters: dict[str, Any]
    all_technical_filters_pass: bool
    calendar: dict[str, Any]
    macro_regime: dict[str, Any]
    scoring: dict[str, Any]
    data_valid: bool
    error: str | None


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
