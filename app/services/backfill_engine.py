"""Generic execution engine for historical trade backfills."""

import json
import logging
from typing import Any, Protocol

import pandas as pd

from ..database.repositories.market_data_provider import MarketDataProvider
from ..database.repositories.trade import TradeRepository
from ..database.session import DatabaseSession
from ..types import TradeStatus
from .screener.strategies.bounce_bandit import BounceBanditStrategy
from .screener.strategies.bridge_scout import BridgeScoutStrategy
from .screener.strategies.tgim import TGIMStrategy
from .trade_manager.strategies.bounce_bandit import BounceBanditTradeStrategy
from .trade_manager.strategies.bridge_scout import BridgeScoutTradeStrategy
from .trade_manager.strategies.tgim import TGIMTradeStrategy

logger = logging.getLogger(__name__)


class ScreenerStrategyProtocol(Protocol):
    """Protocol for screener strategies used in backfilling."""

    def __init__(
        self, repository: TradeRepository, data_provider: MarketDataProvider
    ) -> None: ...

    def run(self, days: int = 0, analysis_date: str | None = None) -> int: ...


class TradeStrategyEngineProtocol(Protocol):
    """Protocol for trade execution and lifecycle management strategies."""

    def check_entry(
        self, trade: dict[str, Any], candle: pd.Series, df_sim: pd.DataFrame
    ) -> Any: ...

    def manage_active_trade(
        self, trade: dict[str, Any], df_sim: pd.DataFrame
    ) -> Any: ...


STRATEGY_MAP: dict[
    str, tuple[type[ScreenerStrategyProtocol], type[TradeStrategyEngineProtocol], int]
] = {
    "tgim": (TGIMStrategy, TGIMTradeStrategy, 30),
    "thank_god_its_monday": (TGIMStrategy, TGIMTradeStrategy, 30),
    "bridge_scout": (BridgeScoutStrategy, BridgeScoutTradeStrategy, 60),
    "bridgescout": (BridgeScoutStrategy, BridgeScoutTradeStrategy, 60),
    "bridge scout": (BridgeScoutStrategy, BridgeScoutTradeStrategy, 60),
    "qqq_eom": (BridgeScoutStrategy, BridgeScoutTradeStrategy, 60),
    "bounce_bandit": (BounceBanditStrategy, BounceBanditTradeStrategy, 350),
    "bouncebandit": (BounceBanditStrategy, BounceBanditTradeStrategy, 350),
    "bounce bandit": (BounceBanditStrategy, BounceBanditTradeStrategy, 350),
    "qqq_meanrev": (BounceBanditStrategy, BounceBanditTradeStrategy, 350),
}


def run_generic_backfill(
    *,
    stocks_session: DatabaseSession,
    signals_session: DatabaseSession,
    strategy_name: str,
    screener_class: type[ScreenerStrategyProtocol],
    strategy_engine_class: type[TradeStrategyEngineProtocol],
    lookback_days: int,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    budget: float = 10000.0,
    clear_existing: bool = True,
) -> dict[str, object]:
    """Executes a backfill simulation for a given strategy from start_date to end_date.

    Args:
        stocks_session: DatabaseSession for market historical prices.
        signals_session: DatabaseSession for trade persistence.
        strategy_name: Identifier of the strategy (e.g. 'bridge_scout').
        screener_class: Class implementing ScreenerStrategyProtocol.
        strategy_engine_class: Class implementing TradeStrategyEngineProtocol.
        lookback_days: Number of historical days required for indicator calculation.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD), defaults to current date.
        budget: Capital allocation budget per trade.
        clear_existing: If True, clears existing trades of strategy before running.

    Returns:
        dict[str, object]: Backfill summary metrics and list of closed trades.
    """
    trade_repository = TradeRepository(signals_session)
    data_provider = MarketDataProvider(stocks_session)

    canonical_strategy = strategy_name.lower().replace(" ", "_")

    if clear_existing:
        with signals_session.connect() as conn:
            conn.execute(
                "DELETE FROM trade_logs WHERE trade_id IN "
                "(SELECT id FROM trades WHERE LOWER(strategy) = ?)",
                (canonical_strategy,),
            )
            conn.execute(
                "DELETE FROM trades WHERE LOWER(strategy) = ?",
                (canonical_strategy,),
            )
            conn.commit()

    screener = screener_class(trade_repository, data_provider)
    strategy_engine = strategy_engine_class()

    data_provider.clear_cache()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    date_range = pd.date_range(start=start_ts, end=end_ts, freq="B")

    stats = {"signals": 0, "filled": 0, "closed": 0}

    for current_ts in date_range:
        current_date_str = current_ts.strftime("%Y-%m-%d")
        sim_date = current_ts.date()

        # 1. Screening
        hits = screener.run(days=0, analysis_date=current_date_str)
        stats["signals"] += hits

        # 2. Execution / Management
        active_trades = [
            t
            for t in trade_repository.get_by_status(
                [TradeStatus.CREATED, TradeStatus.ACTIVE]
            )
            if canonical_strategy in str(t.get("strategy")).lower()
        ]

        if not active_trades:
            continue

        syms = list({t["symbol"] for t in active_trades})
        market_data = data_provider.get_batch_history(
            syms, days=lookback_days, end_date=current_date_str
        )

        for trade in active_trades:
            sym = trade["symbol"]
            if sym not in market_data:
                continue

            df_hist = market_data[sym]
            df_sim = df_hist[df_hist["date"].dt.date <= sim_date].copy()
            if df_sim.empty:
                continue

            candle = df_sim.iloc[-1]
            if candle["date"].date() != sim_date:
                continue

            try:
                ctx = json.loads(trade.get("signal_context") or "{}")
                sig_date_str = ctx.get("date") or ctx.get("setup_date")
                if sig_date_str and sim_date < pd.Timestamp(sig_date_str).date():
                    continue
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                logger.warning("Failed to parse signal context: %s.", error)

            if trade["status"] == TradeStatus.CREATED.value:
                trade["budget"] = budget
                transition = strategy_engine.check_entry(trade, candle, df_sim)
                if transition and transition.updates:
                    trade_repository.update_trade(
                        int(trade["id"]),
                        transition.updates,
                        reason=transition.reason,
                    )
                    stats["filled"] += 1

            elif trade["status"] == TradeStatus.ACTIVE.value:
                transition = strategy_engine.manage_active_trade(trade, df_sim)
                if transition and transition.updates:
                    trade_repository.update_trade(
                        int(trade["id"]),
                        transition.updates,
                        reason=transition.reason,
                    )
                    stats["closed"] += 1

    closed = [
        t
        for t in trade_repository.get_by_status(TradeStatus.CLOSED)
        if canonical_strategy in str(t.get("strategy")).lower()
    ]
    total_pnl = sum(float(t.get("realized_pnl", 0) or 0) for t in closed)
    win_cnt = sum(1 for t in closed if float(t.get("realized_pnl", 0) or 0) > 0)
    win_rate = (win_cnt / len(closed) * 100) if closed else 0.0

    return {
        "start_date": start_ts.strftime("%Y-%m-%d"),
        "end_date": end_ts.strftime("%Y-%m-%d"),
        "signals_generated": stats["signals"],
        "trades_filled": stats["filled"],
        "trades_closed": len(closed),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "closed_trades": [
            {
                "id": t.get("id"),
                "symbol": t.get("symbol"),
                "entry_date": str(t.get("entry_date"))[:10],
                "exit_date": str(t.get("exit_date"))[:10],
                "entry_price": float(t.get("entry_price") or 0.0),
                "exit_price": float(t.get("exit_price") or 0.0),
                "realized_pnl": float(t.get("realized_pnl") or 0.0),
                "exit_reason": t.get("exit_reason"),
            }
            for t in closed
        ],
    }


def run_strategy_backfill(
    *,
    stocks_session: DatabaseSession,
    signals_session: DatabaseSession,
    strategy_name: str,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    budget: float = 10000.0,
    clear_existing: bool = True,
) -> dict[str, object]:
    """Dispatches backfill simulation for a registered strategy by name.

    Args:
        stocks_session: DatabaseSession for market historical prices.
        signals_session: DatabaseSession for trade persistence.
        strategy_name: Name of the strategy to backfill (e.g. 'tgim', 'bridge_scout').
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        budget: Capital allocation budget per trade.
        clear_existing: If True, clears existing strategy trades before running.

    Returns:
        dict[str, object]: Backfill summary metrics and list of closed trades.

    Raises:
        ValueError: If strategy_name is not registered in STRATEGY_MAP.
    """
    key = strategy_name.lower()
    if key not in STRATEGY_MAP:
        raise ValueError(
            f"Unknown strategy for backfill: '{strategy_name}'. "
            f"Available strategies: {list(STRATEGY_MAP.keys())}"
        )

    screener_cls, engine_cls, lookback = STRATEGY_MAP[key]

    return run_generic_backfill(
        stocks_session=stocks_session,
        signals_session=signals_session,
        strategy_name=key,
        screener_class=screener_cls,
        strategy_engine_class=engine_cls,
        lookback_days=lookback,
        start_date=start_date,
        end_date=end_date,
        budget=budget,
        clear_existing=clear_existing,
    )
