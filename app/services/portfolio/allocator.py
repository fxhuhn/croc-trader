import logging
from dataclasses import dataclass
from typing import Final
from ...const import Strategies, STRATEGY_ALIASES

logger = logging.getLogger(__name__)

# Constants
BUDGET_DIP_BUYER: Final[float] = 2000.0
RISK_AMOUNT_HOLD_TARGET: Final[float] = 100.0


@dataclass(frozen=True)
class AllocationResult:
    size: int
    budget_used: float
    risk_amount: float
    reason: str


class PortfolioAllocator:
    """
    Calculates the position size based on Strategy Rules.
    """

    def __init__(self, portfolio_config: dict | None = None):
        """Initializes allocator with optional custom sizing limits."""
        self.portfolio_config = portfolio_config or {}

    def _get_budget_for_strategy(self, strategy: Strategies, default: float) -> float:
        """Retrieves custom budget (quantity) from config or returns default."""
        if not self.portfolio_config or "strategies" not in self.portfolio_config:
            return default

        # Strategy name in YAML matches enum value (e.g. 'dip_buyer')
        strat_key = strategy.value

        for entry in self.portfolio_config["strategies"]:
            if strat_key in entry:
                # Find 'quantity' in properties list
                properties = entry[strat_key]
                for prop in properties:
                    if "quantity" in prop:
                        return float(prop["quantity"])

        return default

    def allocate(self, trade: dict) -> AllocationResult:
        # Resolve Strategy Name
        raw_strategy = trade.get("strategy", "").lower()
        strategy_enum = STRATEGY_ALIASES.get(raw_strategy)

        # Fallback: Check if it's already a valid Enum value
        if not strategy_enum:
            try:
                strategy_enum = Strategies(raw_strategy)
            except ValueError as val_error:
                logger.debug(
                    "Fallback Strategies resolution failed for '%s': %s",
                    raw_strategy,
                    val_error,
                )

        if not strategy_enum:
            # Try prefix matching for Turnover variants (e.g. turnover_timing_1.0)
            # This is a bit looser, but might be needed if exact match fails
            for s in Strategies:
                if raw_strategy.startswith(s.value):
                    strategy_enum = s
                    break

        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = float(trade.get("entry_price") or 0.0)

        if entry_price <= 0:
            return AllocationResult(0, 0.0, 0.0, "Invalid Entry Price")

        if not strategy_enum:
            logger.warning(
                f"[{symbol}] Unknown Strategy for Allocation: {raw_strategy}"
            )
            return AllocationResult(0, 0.0, 0.0, "Unknown Strategy")

        # 1. Dip Buyer (Fixed Budget)
        if strategy_enum == Strategies.DipBuyer:
            budget = self._get_budget_for_strategy(
                Strategies.DipBuyer, BUDGET_DIP_BUYER
            )
            size = int(budget / entry_price)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=budget,
                risk_amount=0.0,  # Not defined for DipBuyer
                reason=f"DipBuyer Budget ({budget})",
            )

        # 2. Hold Target (Fixed Risk)
        if strategy_enum in (
            Strategies.HoldTarget,
            Strategies.CrocSetup,
            Strategies.SplitTarget,
        ):
            stop_loss = float(trade.get("current_stop_loss") or 0.0)

            # Sanity Check for SL
            if stop_loss <= 0 or stop_loss >= entry_price:
                logger.warning(
                    f"[{symbol}] Invalid SL ({stop_loss}) for Risk Calculation. Entry: {entry_price}"
                )
                return AllocationResult(0, 0.0, 0.0, "Invalid Stop Loss")

            risk_per_share = entry_price - stop_loss
            size = int(RISK_AMOUNT_HOLD_TARGET / risk_per_share)

            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Risk/Share > Risk Amount")

            total_budget = size * entry_price

            return AllocationResult(
                size=size,
                budget_used=total_budget,
                risk_amount=RISK_AMOUNT_HOLD_TARGET,
                reason=f"HoldTarget Fixed Risk ({RISK_AMOUNT_HOLD_TARGET})",
            )

        # 3. Turnover Timing (Treat same as DipBuyer - Budget Based)
        if strategy_enum in (
            Strategies.TurnOverTiming,
            Strategies.TurnOverTiming_05,
            Strategies.TurnOverTiming_10,
        ):
            # Resolve specific variant budget if available, otherwise fallback to base Turnover default
            budget = self._get_budget_for_strategy(strategy_enum, BUDGET_DIP_BUYER)

            size = int(budget / entry_price)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=budget,
                risk_amount=0.0,
                reason=f"Turnover Budget ({budget})",
            )

        # 4. Two Percent Strategy (Fixed Budget)
        if strategy_enum == Strategies.TwoPercent:
            budget = self._get_budget_for_strategy(
                Strategies.TwoPercent, BUDGET_DIP_BUYER
            )
            size = int(budget / entry_price)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=budget,
                risk_amount=0.0,
                reason=f"TwoPercent Budget ({budget})",
            )

        # 5. NDX Momentum (Fixed Budget)
        if strategy_enum == Strategies.NDXMomentum:
            budget = self._get_budget_for_strategy(
                Strategies.NDXMomentum, BUDGET_DIP_BUYER
            )
            size = int(budget / entry_price)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=budget,
                risk_amount=0.0,
                reason=f"NDXMomentum Budget ({budget})",
            )

        # 6. Default / Fallback
        logger.warning(f"[{symbol}] Unhandled Strategy Enum: {strategy_enum}")
        return AllocationResult(0, 0.0, 0.0, "Unhandled Strategy")
