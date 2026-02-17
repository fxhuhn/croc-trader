import logging
from ...database.repositories.trade import TradeRepository
from ...types import TradeStatus
from .allocator import PortfolioAllocator
import json

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Central Logic for Capital Allocation.
    Process: Screener -> [PortfolioManager] -> TradeManager(Execution)
    """
    
    def __init__(self, trade_repository: TradeRepository, portfolio_config: dict | None = None):
        self.trade_repository = trade_repository
        self.allocator = PortfolioAllocator(portfolio_config=portfolio_config)
        
    def process_daily_signals(self) -> int:
        """
        Fetches all CREATED trades, allocates size, and updates them.
        Returns: Number of allocated trades.
        """
        logger.info("PortfolioManager: Starting Daily Allocation...")
        
        try:
            # 1. Fetch Candidates (CREATED and size 0)
            # We fetch all CREATED types. If size is already set, we might skip or re-evaluate.
            # Design Decision: If size > 0, assume manual override or previous run. Skip.
            candidates = self.trade_repository.get_by_status(TradeStatus.CREATED)
            
            allocated_count = 0
            
            for trade in candidates:
                symbol = trade['symbol']
                current_size = float(trade.get('initial_size') or 0.0)
                
                if current_size > 0:
                    logger.debug(f"[{symbol}] Skipping Allocation (Size already {current_size})")
                    continue
                    
                # 2. Allocate
                allocation = self.allocator.allocate(trade)
                
                if allocation.size > 0:
                    # 3. Update DB (Store metadata in Context)
                    context = json.loads(trade.get('signal_context') or "{}")
                    context['budget'] = allocation.budget_used
                    context['risk_amount'] = allocation.risk_amount
                    
                    self.trade_repository.update_trade(trade['id'], {
                        "initial_size": allocation.size,
                        "current_size": allocation.size, # Synced for initial state
                        "signal_context": json.dumps(context)
                    }, reason=f"Portfolio Allocated: {allocation.reason}")
                    
                    logger.info(f"[{symbol}] Allocated: {allocation.size} shares ({allocation.reason})")
                    allocated_count += 1
                else:
                    logger.warning(f"[{symbol}] Allocation Failed: {allocation.reason}")
                    # Optional: Mark as SKIPPED or leave as CREATED (0 size) to be ignored by OrderGen
            
            logger.info(f"PortfolioManager: Finished. Allocated {allocated_count} new trades.")
            return allocated_count

        except Exception as exception:
            logger.error(f"PortfolioManager Error: {exception}", exc_info=True)
            return 0
