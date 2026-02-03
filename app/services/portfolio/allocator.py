import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# Constants
BUDGET_DIP_BUYER: Final[float] = 2000.0
RISK_AMOUNT_HOLD_TARGET: Final[float] = 100.0

@dataclass
class AllocationResult:
    size: int
    budget_used: float
    risk_amount: float
    reason: str

class PortfolioAllocator:
    """
    Calculates the position size based on Strategy Rules.
    """
    
    def allocate(self, trade: dict) -> AllocationResult:
        strategy_name = trade.get('strategy', '').lower()
        symbol = trade.get('symbol', 'UNKNOWN')
        entry_price = float(trade.get('entry_price') or 0.0)
        
        if entry_price <= 0:
            return AllocationResult(0, 0.0, 0.0, "Invalid Entry Price")

        # 1. Dip Buyer (Fixed Budget)
        if "dipbuyer" in strategy_name:
            size = int(BUDGET_DIP_BUYER / entry_price)
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")
                
            return AllocationResult(
                size=size,
                budget_used=BUDGET_DIP_BUYER,
                risk_amount=0.0, # Not defined for DipBuyer
                reason=f"DipBuyer Fixed Budget ({BUDGET_DIP_BUYER})"
            )

        # 2. Hold Target (Fixed Risk)
        if "holdtarget" in strategy_name or "croc" in strategy_name:
            stop_loss = float(trade.get('current_stop_loss') or 0.0)
            
            # Sanity Check for SL
            if stop_loss <= 0 or stop_loss >= entry_price:
                logger.warning(f"[{symbol}] Invalid SL ({stop_loss}) for Risk Calculation. Entry: {entry_price}")
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
                reason=f"HoldTarget Fixed Risk ({RISK_AMOUNT_HOLD_TARGET})"
            )

        # 3. Turnover Timing (Treat same as DipBuyer - Budget Based)
        if "turnover" in strategy_name:
            # Similar to DipBuyer
            size = int(BUDGET_DIP_BUYER / entry_price) # Reuse 2000 constant for now
            if size < 1:
                return AllocationResult(0, 0.0, 0.0, "Price > Budget")
                
            return AllocationResult(
                size=size,
                budget_used=BUDGET_DIP_BUYER,
                risk_amount=0.0,
                reason=f"Turnover Fixed Budget ({BUDGET_DIP_BUYER})"
            )
            
        # 4. Default / Fallback
        logger.warning(f"[{symbol}] Unknown Strategy for Allocation: {strategy_name}")
        return AllocationResult(0, 0.0, 0.0, "Unknown Strategy")
