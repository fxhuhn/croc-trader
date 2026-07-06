import json
import logging
from dataclasses import dataclass
from decimal import Decimal

from ...config import PortfolioConfig
from ...const import STRATEGY_ALIASES, Strategies

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationResult:
    size: int
    budget_used: float
    risk_amount: float
    reason: str


class PortfolioAllocator:
    """Calculates the position size based on Strategy Rules."""

    def __init__(self, portfolio_config: PortfolioConfig | None = None) -> None:
        """Initializes allocator with optional custom sizing limits."""
        self.portfolio_config = portfolio_config or PortfolioConfig()

    def allocate(self, trade: dict[str, object]) -> AllocationResult:
        # Resolve Strategy Name
        raw_strategy = trade.get("strategy", "").lower()
        strategy_enum = STRATEGY_ALIASES.get(raw_strategy)

        # Fallback: Check if it's already a valid Enum value
        if not strategy_enum:
            try:
                strategy_enum = Strategies(raw_strategy)
            except ValueError as value_error:
                logger.debug(
                    "Fallback Strategies resolution failed for '%s': %s",
                    raw_strategy,
                    value_error,
                )

        if not strategy_enum:
            # Try prefix matching for Turnover variants (e.g. turnover_timing_1.0)
            # This is a bit looser, but might be needed if exact match fails
            for strategy_enum_option in Strategies:
                if raw_strategy.startswith(strategy_enum_option.value):
                    strategy_enum = strategy_enum_option
                    break

        symbol = trade.get("symbol", "UNKNOWN")
        entry_price_decimal = Decimal(str(trade.get("entry_price") or 0.0))

        if entry_price_decimal <= 0:
            return AllocationResult(0, 0.0, 0.0, "Invalid Entry Price")

        if not strategy_enum:
            logger.warning(
                f"[{symbol}] Unknown Strategy for Allocation: {raw_strategy}"
            )
            return AllocationResult(0, 0.0, 0.0, "Unknown Strategy")

        # 1. Dip Buyer (Fixed Budget)
        if strategy_enum == Strategies.DipBuyer:
            budget_decimal = Decimal(
                str(self.portfolio_config.get_budget(strategy_enum.value))
            )
            size = int(budget_decimal / entry_price_decimal)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=float(budget_decimal),
                risk_amount=0.0,  # Not defined for DipBuyer
                reason=f"DipBuyer Budget ({budget_decimal})",
            )

        # 2. Hold Target (Fixed Risk)
        if strategy_enum in (
            Strategies.HoldTarget,
            Strategies.CrocSetup,
            Strategies.SplitTarget,
        ):
            stop_loss_decimal = Decimal(str(trade.get("current_stop_loss") or 0.0))

            # Sanity Check for SL (D-04 Short check)
            direction = "long"
            raw_context = trade.get("signal_context", "")
            if isinstance(raw_context, str) and raw_context:
                try:
                    context = json.loads(raw_context)
                    direction = context.get("direction", "long")
                except (json.JSONDecodeError, TypeError) as decode_error:
                    logger.warning(
                        "[%s] Failed to decode signal_context: %s",
                        symbol,
                        decode_error,
                    )
            elif isinstance(raw_context, dict):
                direction = raw_context.get("direction", "long")

            if direction == "short":
                is_invalid_sl = (
                    stop_loss_decimal <= 0 or stop_loss_decimal <= entry_price_decimal
                )
            else:
                is_invalid_sl = (
                    stop_loss_decimal <= 0 or stop_loss_decimal >= entry_price_decimal
                )

            if is_invalid_sl:
                logger.warning(
                    f"[{symbol}] Invalid SL ({stop_loss_decimal}) for Risk Calculation. Entry: {entry_price_decimal}, Direction: {direction}"
                )
                return AllocationResult(0, 0.0, 0.0, "Invalid Stop Loss")

            risk_per_share_decimal = abs(entry_price_decimal - stop_loss_decimal)
            strategy_key = strategy_enum.value
            if strategy_key in ("croc_setup", "split_target"):
                strategy_key = "hold_target"
            risk_amount_decimal = Decimal(
                str(self.portfolio_config.get_risk_amount(strategy_key))
            )
            if risk_amount_decimal <= 0:
                risk_amount_decimal = Decimal("100.0")

            size = int(risk_amount_decimal / risk_per_share_decimal)

            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Risk/Share > Risk Amount")

            total_budget_decimal = Decimal(str(size)) * entry_price_decimal

            return AllocationResult(
                size=size,
                budget_used=float(total_budget_decimal),
                risk_amount=float(risk_amount_decimal),
                reason=f"HoldTarget Fixed Risk ({risk_amount_decimal})",
            )

        # 3. Turnover Timing (Treat same as DipBuyer - Budget Based)
        if strategy_enum in (
            Strategies.TurnOverTiming,
            Strategies.TurnOverTiming_05,
            Strategies.TurnOverTiming_10,
        ):
            budget_decimal = Decimal(
                str(self.portfolio_config.get_budget(strategy_enum.value))
            )
            size = int(budget_decimal / entry_price_decimal)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=float(budget_decimal),
                risk_amount=0.0,
                reason=f"Turnover Budget ({budget_decimal})",
            )

        # 4. Two Percent Strategy (Fixed Budget)
        if strategy_enum == Strategies.TwoPercent:
            budget_decimal = Decimal(
                str(self.portfolio_config.get_budget(strategy_enum.value))
            )
            size = int(budget_decimal / entry_price_decimal)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=float(budget_decimal),
                risk_amount=0.0,
                reason=f"TwoPercent Budget ({budget_decimal})",
            )

        # 5. NDX Momentum (Fixed Budget)
        if strategy_enum == Strategies.NDXMomentum:
            budget_decimal = Decimal(
                str(self.portfolio_config.get_budget(strategy_enum.value))
            )
            size = int(budget_decimal / entry_price_decimal)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")

            return AllocationResult(
                size=size,
                budget_used=float(budget_decimal),
                risk_amount=0.0,
                reason=f"NDXMomentum Budget ({budget_decimal})",
            )

        # 6. Default / Fallback
        logger.warning("[%s] Unhandled Strategy Enum: %s", symbol, strategy_enum)
        return AllocationResult(0, 0.0, 0.0, "Unhandled Strategy")
