import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

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

    def allocate(self, trade: dict[str, Any]) -> AllocationResult:
        """Allocates position sizing and risk budget based on trade strategy."""
        raw_strategy = str(trade.get("strategy", "")).lower()
        strategy_enum = self._resolve_strategy_enum(raw_strategy)
        symbol = str(trade.get("symbol", "UNKNOWN"))
        entry_price = Decimal(str(trade.get("entry_price") or 0.0))

        if entry_price <= 0:
            return AllocationResult(0, 0.0, 0.0, "Invalid Entry Price")

        if not strategy_enum:
            logger.warning(
                "[%s] Unknown Strategy for Allocation: %s", symbol, raw_strategy
            )
            return AllocationResult(0, 0.0, 0.0, "Unknown Strategy")

        if strategy_enum in (
            Strategies.HoldTarget,
            Strategies.CrocSetup,
            Strategies.SplitTarget,
        ):
            return self._allocate_risk_strategy(
                trade, symbol, strategy_enum, entry_price
            )

        strategy_labels = {
            Strategies.DipBuyer: "DipBuyer",
            Strategies.TurnOverTiming: "Turnover",
            Strategies.TurnOverTiming_05: "Turnover",
            Strategies.TurnOverTiming_10: "Turnover",
            Strategies.TwoPercent: "TwoPercent",
            Strategies.NDXMomentum: "NDXMomentum",
            Strategies.TGIM: "TGIM",
            Strategies.BridgeScout: "BridgeScout",
            Strategies.BounceBandit: "BounceBandit",
        }

        label = strategy_labels.get(strategy_enum)
        if label:
            return self._allocate_budget_strategy(strategy_enum, entry_price, label)

        logger.warning("[%s] Unhandled Strategy Enum: %s", symbol, strategy_enum)
        return AllocationResult(0, 0.0, 0.0, "Unhandled Strategy")

    def _resolve_strategy_enum(self, raw_strategy: str) -> Strategies | None:
        """Resolves strategy string to canonical Strategies enum."""
        strategy_enum = STRATEGY_ALIASES.get(raw_strategy)
        if strategy_enum:
            return strategy_enum

        try:
            return Strategies(raw_strategy)
        except ValueError:
            pass

        for option in Strategies:
            if raw_strategy.startswith(option.value):
                return option

        return None

    def _allocate_budget_strategy(
        self,
        strategy_enum: Strategies,
        entry_price: Decimal,
        label: str,
    ) -> AllocationResult:
        """Allocates sizing for budget-based strategies."""
        budget = Decimal(str(self.portfolio_config.get_budget(strategy_enum.value)))
        size = int(budget / entry_price)
        if size < 1:
            return AllocationResult(0, 0.0, 0.0, "Price > Budget")

        return AllocationResult(
            size=size,
            budget_used=float(budget),
            risk_amount=0.0,
            reason=f"{label} Budget ({budget})",
        )

    def _allocate_risk_strategy(
        self,
        trade: dict[str, Any],
        symbol: str,
        strategy_enum: Strategies,
        entry_price: Decimal,
    ) -> AllocationResult:
        """Allocates sizing for fixed-risk strategies (HoldTarget, CrocSetup, SplitTarget)."""
        stop_loss = Decimal(str(trade.get("current_stop_loss") or 0.0))
        direction = self._extract_direction(trade, symbol)

        if direction == "short":
            is_invalid_sl = stop_loss <= 0 or stop_loss <= entry_price
        else:
            is_invalid_sl = stop_loss <= 0 or stop_loss >= entry_price

        if is_invalid_sl:
            logger.warning(
                "[%s] Invalid SL (%s) for Risk Calculation. Entry: %s, Direction: %s",
                symbol,
                stop_loss,
                entry_price,
                direction,
            )
            return AllocationResult(0, 0.0, 0.0, "Invalid Stop Loss")

        risk_per_share = abs(entry_price - stop_loss)
        strategy_key = (
            "hold_target"
            if strategy_enum.value in ("croc_setup", "split_target")
            else strategy_enum.value
        )
        risk_amount = Decimal(str(self.portfolio_config.get_risk_amount(strategy_key)))
        if risk_amount <= 0:
            risk_amount = Decimal("100.0")

        size = int(risk_amount / risk_per_share)
        if size < 1:
            return AllocationResult(0, 0.0, 0.0, "Risk/Share > Risk Amount")

        total_budget = Decimal(str(size)) * entry_price
        return AllocationResult(
            size=size,
            budget_used=float(total_budget),
            risk_amount=float(risk_amount),
            reason=f"HoldTarget Fixed Risk ({risk_amount})",
        )

    def _extract_direction(self, trade: dict[str, Any], symbol: str) -> str:
        """Extracts order direction from trade context."""
        raw_context = trade.get("signal_context", "")
        if isinstance(raw_context, dict):
            return str(raw_context.get("direction", "long"))
        if isinstance(raw_context, str) and raw_context:
            try:
                context = json.loads(raw_context)
                if isinstance(context, dict):
                    return str(context.get("direction", "long"))
            except (json.JSONDecodeError, TypeError) as decode_error:
                logger.warning(
                    "[%s] Failed to decode signal_context: %s",
                    symbol,
                    decode_error,
                )
        return "long"
