from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class CapacityMonitor:
    """Tracks active trades to calculate utilization percentiles."""

    _concurrent_trades: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {
            "dip_buyer": [],
            "turnover_0.5": [],
            "turnover_1.0": [],
            "two_percent": [],
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

        counts = [entry["count"] for entry in self._concurrent_trades[strategy]]
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

        if utilization < 0.5:
            return self.max_multiplier
        elif utilization < 0.75:
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


class OverflowProtection:
    """Safeguards against excessive total portfolio exposure."""

    def __init__(self, max_total_exposure: float = 1.0) -> None:
        self.max_total_exposure = max_total_exposure

    def apply_limits(
        self,
        new_trades: list[dict[str, object]],
        current_exposure: float,
        capital: float,
    ) -> list[dict[str, object]]:
        """
        Adjusts proposed trade sizes if total exposure exceeds limit.

        Args:
            new_trades: List of dicts checks with 'position_size' key.
            current_exposure: Current active market exposure ratio (0.0 to 1.0+).
            capital: Total portfolio equity.

        Returns:
            List of modified trades with adjusted sizes.
        """
        total_proposed_value = sum(t["expected_size"] for t in new_trades)
        proposed_exposure_ratio = total_proposed_value / capital

        projected_total_exposure = current_exposure + proposed_exposure_ratio

        if projected_total_exposure > self.max_total_exposure:
            # We are over limit. Calculate reduction factor.
            # Available room = Max - Current
            # If current is already over max, room is 0 (allow no new trades? or minimal?)
            # Logic: Proportional reduction of NEW trades.

            available_exposure_room = max(
                0.0, self.max_total_exposure - current_exposure
            )

            if available_exposure_room == 0:
                # Hard block: No new trades allowed
                reduction_factor = 0.0
            else:
                reduction_factor = available_exposure_room / proposed_exposure_ratio

            # Apply reduction
            for trade in new_trades:
                original_size = trade["expected_size"]
                trade["expected_size"] = original_size * reduction_factor
                trade["note"] = (
                    f"Reduced by {(1 - reduction_factor) * 100:.1f}% "
                    f"(Exposure {projected_total_exposure:.2f} > {self.max_total_exposure})"
                )

        return new_trades
