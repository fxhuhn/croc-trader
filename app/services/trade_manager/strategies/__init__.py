from .abstract import BaseTradeStrategy
from .bounce_bandit import BounceBanditTradeStrategy
from .bridge_scout import BridgeScoutTradeStrategy
from .dip_buyer import DipBuyerStrategy
from .hold_target import HoldTargetStrategy
from .ndx_momentum import NDXMomentumTradeStrategy
from .tgim import TGIMTradeStrategy
from .turnover_timing import TurnoverTimingStrategy
from .two_percent_strategy import TwoPercentStrategy

# Uniform naming aliases for consistency
DipBuyerTradeStrategy = DipBuyerStrategy
HoldTargetTradeStrategy = HoldTargetStrategy
TurnoverTimingTradeStrategy = TurnoverTimingStrategy
TwoPercentTradeStrategy = TwoPercentStrategy

__all__ = [
    "BaseTradeStrategy",
    "BounceBanditTradeStrategy",
    "BridgeScoutTradeStrategy",
    "DipBuyerStrategy",
    "DipBuyerTradeStrategy",
    "HoldTargetStrategy",
    "HoldTargetTradeStrategy",
    "NDXMomentumTradeStrategy",
    "TGIMTradeStrategy",
    "TurnoverTimingStrategy",
    "TurnoverTimingTradeStrategy",
    "TwoPercentStrategy",
    "TwoPercentTradeStrategy",
]
