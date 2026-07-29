"""Service for executing historical trade backfills for the Bounce Bandit strategy."""

import json
import logging

import pandas as pd

from ..database.repositories.market_data_provider import MarketDataProvider
from ..database.repositories.trade import TradeRepository
from ..database.session import DatabaseSession
from ..types import TradeStatus
from .screener.strategies.bounce_bandit import BounceBanditStrategy
from .trade_manager.strategies.bounce_bandit import BounceBanditTradeStrategy

logger = logging.getLogger(__name__)


def run_bounce_bandit_backfill(
    stocks_session: DatabaseSession,
    signals_session: DatabaseSession,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    budget: float = 10000.0,
    clear_existing: bool = True,
) -> dict[str, object]:
    """Executes a backfill simulation for Bounce Bandit strategy from start_date to end_date.

    Args:
        stocks_session: DatabaseSession for market historical prices.
        signals_session: DatabaseSession for trade persistence.
        start_date: Start date string (YYYY-MM-DD), defaults to "2025-01-01".
        end_date: End date string (YYYY-MM-DD), defaults to current date.
        budget: Capital allocation budget per trade.
        clear_existing: If True, clears existing bounce_bandit trades before running.

    Returns:
        dict[str, object]: Backfill summary metrics and list of closed trades.
    """
    trade_repository = TradeRepository(signals_session)
    data_provider = MarketDataProvider(stocks_session)

    if clear_existing:
        with signals_session.connect() as conn:
            conn.execute(
                "DELETE FROM trade_logs WHERE trade_id IN (SELECT id FROM trades WHERE strategy='bounce_bandit')"
            )
            conn.execute("DELETE FROM trades WHERE strategy='bounce_bandit'")
            conn.commit()

    screener = BounceBanditStrategy(trade_repository, data_provider)
    strategy_engine = BounceBanditTradeStrategy()

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
                [
                    TradeStatus.CREATED,
                    TradeStatus.ACTIVE,
                ]
            )
            if "bounce_bandit" in str(t.get("strategy")).lower()
        ]

        if active_trades:
            syms = list({t["symbol"] for t in active_trades})
            market_data = data_provider.get_batch_history(
                syms, days=350, end_date=current_date_str
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
        if "bounce_bandit" in str(t.get("strategy")).lower()
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
