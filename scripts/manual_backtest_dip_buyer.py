"""Manual Backtest Runner for the DipBuyer Strategy.

Simulates the EOD screening loop and subsequent execution (checking entries and exits)
for DipBuyer signals over a historical range of dates. Directly reads market data from
the stocks database and writes/manages trade results in the active signals database.

Usage:
    python script/manual_backtest_dip_buyer.py --start YYYY-MM-DD [--end YYYY-MM-DD] [--budget 2000.0]

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
from app.database.repositories.market_data_provider import (
    MarketDataProvider,  # noqa: E402
)
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402

# FIX 1: Import mit Alias, da die Klasse in der Datei nur 'DipBuyerStrategy' heißt
from app.services.screener.strategies.dip_buyer import (  # noqa: E402
    DipBuyerStrategy as DipBuyerScreenerStrategy,
)
from app.services.trade_manager.strategies.dip_buyer import (
    DipBuyerStrategy,  # noqa: E402
)
from app.types import TradeStatus  # noqa: E402

# Logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("DipBacktest")
logger.setLevel(logging.INFO)
# Unterdrückt Warnungen aus der Strategie für saubereren Output
logging.getLogger("app.services.screener.strategies.dip_buyer").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--budget", type=float, default=2000.0
    )  # Standard 2000$ pro Aktie
    args = parser.parse_args()

    stocks_session = DatabaseSession(str(settings.get_path("stocks")))
    signals_session = DatabaseSession(str(settings.get_path("signals")))
    data_provider = MarketDataProvider(stocks_session)
    trade_repository = TradeRepository(signals_session)
    # signal_repository wird vom Screener laut Code nicht benötigt, daher hier optional

    # FIX 2: Korrekte Initialisierung
    # Der Screener erwartet (trade_repository, data_provider, telegram_bot=None, config=Default)
    # Wir übergeben KEIN signal_repository und KEIN None für config (sonst Crash)
    screener = DipBuyerScreenerStrategy(trade_repository, data_provider)

    strategy_engine = DipBuyerStrategy()

    data_provider.clear_cache()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    logger.info(
        f"🚀 DipBuyer Backtest: {start_date.date()} bis {end_date.date()} | Budget: {args.budget}$"
    )
    stats = {"signals": 0, "filled": 0, "closed": 0}

    for current_ts in tqdm(date_range, desc="Simulation"):
        current_date_str = current_ts.strftime("%Y-%m-%d")
        sim_date = current_ts.date()

        # 1. Screening
        hits = screener.run(days=0, analysis_date=current_date_str)
        stats["signals"] += hits

        # 2. Trading
        active_trades = [
            t
            for t in trade_repository.get_by_status(
                [
                    TradeStatus.CREATED,
                    TradeStatus.ACTIVE,
                ]
            )
            if "DipBuyer" in t["strategy"]
        ]

        if active_trades:
            syms = list({t["symbol"] for t in active_trades})
            market_data = data_provider.get_batch_history(
                syms, days=60, end_date=current_date_str
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

                # Datums-Check
                try:
                    ctx = json.loads(trade.get("signal_context") or "{}")
                    sig_date_str = ctx.get("date") or ctx.get("setup_date")
                    if sig_date_str:
                        if sim_date <= pd.Timestamp(sig_date_str).date():
                            continue
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    logging.warning(
                        "Failed to determine trade date from context: %s. Proceeding with fallback.",
                        error,
                    )

                trader = strategy_engine

                if trade["status"] == TradeStatus.CREATED.value:
                    # BUDGET ÜBERGEBEN
                    trade["budget"] = args.budget

                    res = trader.check_entry(trade, candle, df_sim, trade_repository)
                    if res and "FILLED" in str(res):
                        stats["filled"] += 1

                elif trade["status"] == TradeStatus.ACTIVE.value:
                    res = trader.manage_active_trade(trade, df_sim, trade_repository)
                    if res:
                        stats["closed"] += 1

    # Report
    closed = [
        t
        for t in trade_repository.get_by_status(TradeStatus.CLOSED)
        if "DipBuyer" in t["strategy"]
    ]
    total_pnl = sum(float(t.get("realized_pnl", 0) or 0) for t in closed)
    win_cnt = sum(1 for t in closed if float(t.get("realized_pnl", 0) or 0) > 0)
    wr = (win_cnt / len(closed) * 100) if closed else 0.0

    print("\n" + "=" * 60)
    print("🏁 DIP BUYER ERGEBNIS")
    print("=" * 60)
    print(f"   Signale:       {stats['signals']}")
    print(f"   Geschlossen:   {len(closed)}")
    print("-" * 60)
    print(f"   💰 GESAMT PnL: ${total_pnl:,.2f}")
    print(f"   📈 Win Rate:   {wr:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
