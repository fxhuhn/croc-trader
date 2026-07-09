"""Post-Mortem Execution Analysis for Croc and Split Strategy Signals.

Retrieves trade proposals from the signals database and matches them against subsequent-day
historical candles in the stocks database. Calculates the proportion of trades that would have
triggered as breakout entries (High >= Entry) versus limit entries (Low <= Entry). Helps developers
diagnose filling logic issues and historical signal execution differences.

Usage:
    python script/debug_croc_fills.py

Side Effects:
    None (performs read-only queries on signals and stocks databases).
"""

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent
DB_SIGNALS = BASE_DIR / "data" / "signals.db"
DB_STOCKS = BASE_DIR / "data" / "stocks.db"


def main():
    if not DB_SIGNALS.exists() or not DB_STOCKS.exists():
        print(f"❌ Datenbanken nicht gefunden in {BASE_DIR}/instance/")
        return

    conn_sig = sqlite3.connect(DB_SIGNALS)
    conn_stk = sqlite3.connect(DB_STOCKS)

    print("\n🔍 ANALYSE CROC SETUP (Sample Check)")
    print("=" * 100)
    print(
        f"{'DATUM':<12} {'SYMBOL':<6} {'TYP':<8} {'ENTRY':<10} {'NEXT HIGH':<10} {'NEXT LOW':<10} {'BREAKOUT?':<10} {'LIMIT?':<10} {'STATUS'}"
    )
    print("-" * 100)

    # 1. Hole eine Stichprobe von Croc Trades (CREATED, MISSED, CLOSED)
    df_trades = pd.read_sql(
        """
        SELECT id, symbol, entry_price, created_at, status, signal_context
        FROM trades
        WHERE strategy LIKE 'Croc%' OR strategy LIKE 'Split%'
        LIMIT 100
    """,
        conn_sig,
    )

    if df_trades.empty:
        print("Keine Croc Trades gefunden.")
        return

    valid_breakouts = 0
    valid_limits = 0
    total_checked = 0

    for _, trade in df_trades.iterrows():
        try:
            # Datum des Signals ermitteln
            created_at = pd.Timestamp(trade["created_at"])
            sig_date_str = created_at.strftime("%Y-%m-%d")

            # Kontext prüfen (optional)
            if pd.notna(trade["signal_context"]):
                try:
                    ctx = json.loads(str(trade["signal_context"]))
                    if isinstance(ctx, dict) and "date" in ctx:
                        sig_date_str = ctx["date"]
                except (json.JSONDecodeError, KeyError) as error:
                    logger.debug("Could not parse signal_context JSON: %s", error)

            # Marktdaten für die Tage NACH dem Signal holen
            query = """
                SELECT date, open, high, low, close
                FROM market_prices
                WHERE symbol = ?
                AND date > ?
                ORDER BY date ASC LIMIT 1
            """
            df_price = pd.read_sql(
                query, conn_stk, params=(trade["symbol"], sig_date_str)
            )

            if df_price.empty:
                continue

            # Der Tag der Ausführung (Next Day)
            next_day = df_price.iloc[0]
            next_date = pd.Timestamp(next_day["date"]).strftime("%Y-%m-%d")

            entry = float(trade["entry_price"])
            high = float(next_day["high"])
            low = float(next_day["low"])

            # CHECK: War es ein Breakout? (Preis stieg über Entry)
            is_breakout = high >= entry

            # CHECK: War es ein Limit/Dip? (Preis fiel unter Entry)
            is_limit = low <= entry

            # Typ bestimmen (Breakout Setup vs Dip Setup)
            # Wir nehmen an: Wenn Entry > Close vom Vortag -> Breakout. Hier vereinfacht.

            print(
                f"{next_date:<12} {trade['symbol']:<6} {'CROC':<8} {entry:<10.2f} {high:<10.2f} {low:<10.2f} {str(is_breakout):<10} {str(is_limit):<10} {trade['status']}"
            )

            if is_breakout:
                valid_breakouts += 1
            if is_limit:
                valid_limits += 1
            total_checked += 1

        except Exception as error:
            logger.error("Failed to process trade %s: %s", trade.get("symbol"), error)
            continue

    print("=" * 100)
    print(f"ANALYSE ERGEBNIS ({total_checked} Trades geprüft):")
    print(
        f"👉 Wäre als BREAKOUT (High >= Entry) gefüllt worden: {valid_breakouts} ({valid_breakouts / total_checked * 100:.1f}%)"
    )
    print(
        f"👉 Wäre als LIMIT (Low <= Entry) gefüllt worden:    {valid_limits} ({valid_limits / total_checked * 100:.1f}%)"
    )
    print(
        f"👉 Tatsächlich ausgeführte Trades im Backtest:      {len(df_trades[df_trades['status'] != 'CREATED'])}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
