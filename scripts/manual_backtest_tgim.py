"""Manual Backtest Runner for the TGIM Strategy.

Simulates the EOD screening loop and subsequent execution (checking entries and exits)
for TGIM signals over a historical range of dates (e.g. 2026-01-01 to Present).
Directly reads market data from the stocks database and manages trade results in signals.db.

Usage:
    python scripts/manual_backtest_tgim.py --start 2026-01-01 [--end YYYY-MM-DD] [--budget 10000.0]

Side Effects:
    Modifies trades and trade_logs in the active signals database (data/signals.db).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.market_data_provider import (  # noqa: E402
    MarketDataProvider,
)
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.services.screener.strategies.tgim import TGIMStrategy  # noqa: E402
from app.services.trade_manager.strategies.tgim import (  # noqa: E402
    TGIMTradeStrategy,
)
from app.types import TradeStatus  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("TGIMBacktest")
logger.setLevel(logging.INFO)
logging.getLogger("app.services.screener.strategies.tgim").setLevel(logging.ERROR)


def main() -> None:
    """Executes historical backtest for TGIM strategy."""
    parser = argparse.ArgumentParser(
        description="Run TGIM strategy historical backtest."
    )
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--budget", type=float, default=10000.0)
    args = parser.parse_args()

    stocks_session = DatabaseSession(str(settings.get_path("stocks")))
    signals_session = DatabaseSession(str(settings.get_path("signals")))
    data_provider = MarketDataProvider(stocks_session)
    trade_repository = TradeRepository(signals_session)

    # Clear prior TGIM trade records for clean backtest run
    with signals_session.connect() as conn:
        conn.execute(
            "DELETE FROM trade_logs WHERE trade_id IN (SELECT id FROM trades WHERE strategy='tgim')"
        )
        conn.execute("DELETE FROM trades WHERE strategy='tgim'")
        conn.commit()

    screener = TGIMStrategy(trade_repository, data_provider)
    strategy_engine = TGIMTradeStrategy()

    data_provider.clear_cache()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    logger.info(
        "🚀 TGIM Backtest: %s bis %s | Budget: $%.2f",
        start_date.date(),
        end_date.date(),
        args.budget,
    )
    stats = {"signals": 0, "filled": 0, "closed": 0}

    for current_ts in tqdm(date_range, desc="Simulation"):
        current_date_str = current_ts.strftime("%Y-%m-%d")
        sim_date = current_ts.date()

        # 1. Screening
        hits = screener.run(days=0, analysis_date=current_date_str)
        stats["signals"] += hits

        # 2. Trading / Management
        active_trades = [
            t
            for t in trade_repository.get_by_status(
                [
                    TradeStatus.CREATED,
                    TradeStatus.ACTIVE,
                ]
            )
            if "tgim" in str(t.get("strategy")).lower()
        ]

        if active_trades:
            syms = list({t["symbol"] for t in active_trades})
            market_data = data_provider.get_batch_history(
                syms, days=30, end_date=current_date_str
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
                    logging.warning("Failed to parse signal context: %s.", error)

                if trade["status"] == TradeStatus.CREATED.value:
                    trade["budget"] = args.budget
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

    # Report
    closed = [
        t
        for t in trade_repository.get_by_status(TradeStatus.CLOSED)
        if "tgim" in str(t.get("strategy")).lower()
    ]
    total_pnl = sum(float(t.get("realized_pnl", 0) or 0) for t in closed)
    win_cnt = sum(1 for t in closed if float(t.get("realized_pnl", 0) or 0) > 0)
    win_rate = (win_cnt / len(closed) * 100) if closed else 0.0

    print("\n" + "=" * 60)
    print("🏁 TGIM BACKTEST ERGEBNIS")
    print("=" * 60)
    print(f"   Signale:       {stats['signals']}")
    print(f"   Ausgeführt:    {stats['filled']}")
    print(f"   Geschlossen:   {len(closed)}")
    print("-" * 60)
    print(f"   💰 GESAMT PnL: ${total_pnl:,.2f}")
    print(f"   📈 Win Rate:   {win_rate:.1f}%")
    print("=" * 60 + "\n")

    # Detailed trade breakdown table
    if closed:
        print("Trade Details:")
        print(
            f"{'ID':<5} {'Symbol':<8} {'Entry Date':<12} {'Exit Date':<12} {'Entry Price':<12} {'Exit Price':<12} {'PnL ($)':<10} {'Reason':<15}"
        )
        print("-" * 90)
        for t in closed:
            print(
                f"{t.get('id', ''):<5} {t.get('symbol', ''):<8} {t.get('entry_date', '')[:10]:<12} {t.get('exit_date', '')[:10]:<12} {float(t.get('entry_price') or 0):<12.2f} {float(t.get('exit_price') or 0):<12.2f} {float(t.get('realized_pnl') or 0):<10.2f} {t.get('exit_reason', ''):<15}"
            )
        print("-" * 90)


if __name__ == "__main__":
    main()
