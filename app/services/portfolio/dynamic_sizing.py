from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ...const import Strategies

# Capacity utilization thresholds for position sizing aggression zones
UTILIZATION_LOW_THRESHOLD: float = 0.5
UTILIZATION_MEDIUM_THRESHOLD: float = 0.75


@dataclass
class CapacityMonitor:
    """Tracks active trades to calculate utilization percentiles."""

    _concurrent_trades: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {
            Strategies.DipBuyer.value: [],
            Strategies.TurnOverTiming_05.value: [],
            Strategies.TurnOverTiming_10.value: [],
            Strategies.TwoPercent.value: [],
            "global": [],
        }
    )

    def update(
        self,
        date: datetime,
        active_trades: dict[str, list[dict[str, object]]],
    ) -> None:
        """
        Track daily concurrent trades.

        Args:
            date: The current simulation date.
            active_trades: Dictionary mapping strategy names to lists of active trade objects.
        """
        total_count = 0
        for strategy, trades in active_trades.items():
            count = len(trades)
            total_count += count

            # Initialize list if strategy seen for first time
            if strategy not in self._concurrent_trades:
                self._concurrent_trades[strategy] = []

            self._concurrent_trades[strategy].append({"date": date, "count": count})

        # Track global concurrency
        self._concurrent_trades["global"].append({"date": date, "count": total_count})

    def get_percentile(self, strategy: str, percentile: int) -> float:
        """Get Nth percentile of concurrent trades for a strategy or global."""
        if strategy not in self._concurrent_trades:
            return 0.0

        counts = [
            int(str(entry["count"]))
            for entry in self._concurrent_trades[strategy]
            if "count" in entry
        ]
        if not counts:
            return 0.0

        return float(np.percentile(counts, percentile))

    def get_current_utilization(self, strategy: str, current_count: int) -> float:
        """
        Calculate current utilization vs 95th percentile.

        Args:
            strategy: Strategy name (or 'global').
            current_count: Number of currently active trades.
        """
        percentile_95 = self.get_percentile(strategy, 95)

        # Safety: If P95 is 0 (no history), avoid division by zero
        # Return 0.0 utilization (safe default)
        return float(current_count) / percentile_95 if percentile_95 > 0 else 0.0


class DynamicPositionSizer:
    """Calculates position multipliers based on capacity utilization."""

    def __init__(
        self,
        base_kelly: float = 0.39,
        target_percentile: int = 95,
        max_multiplier: float = 2.0,
    ) -> None:
        self.base_kelly = base_kelly
        self.target_percentile = target_percentile
        self.max_multiplier = max_multiplier

    def calculate_multiplier(
        self,
        strategy: str,
        current_concurrent_trades: int,
        capacity_monitor: CapacityMonitor,
    ) -> float:
        """
        Calculate position size multiplier based on current capacity utilization.

        Logic:
        - < 50% capacity: Max Aggression
        - 50-75% capacity: Normal Aggression
        - 75-100% capacity: Conservative
        - > 100% capacity: De-leverage (Overflow Protection)
        """
        utilization = capacity_monitor.get_current_utilization(
            strategy, current_concurrent_trades
        )

        if utilization < UTILIZATION_LOW_THRESHOLD:
            return self.max_multiplier
        elif utilization < UTILIZATION_MEDIUM_THRESHOLD:
            return 1.5
        elif utilization < 1.0:
            return 1.0
        else:
            # Overflow: Reduce sizes inversely proportional to excess
            # e.g. at 125% capacity (1.25), mult = 0.8 * (1/1.25) = 0.64
            safe_utilization = max(utilization, 0.01)  # Avoid div/0
            return 0.8 * (1.0 / safe_utilization)

    def calculate_position_size(
        self,
        capital: float,
        strategy: str,
        current_concurrent_trades: int,
        capacity_monitor: CapacityMonitor,
    ) -> float:
        """Calculate the exact dollar amount for a position."""
        multiplier = self.calculate_multiplier(
            strategy, current_concurrent_trades, capacity_monitor
        )
        effective_kelly = self.base_kelly * multiplier
        return capital * effective_kelly
