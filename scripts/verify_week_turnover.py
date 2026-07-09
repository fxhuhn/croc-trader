"""Specific Scenario Verifier for TurnoverTiming Strategy.

Simulates a specific week of trading (Feb 6th - Feb 10th, 2026) for TurnoverTiming strategies on a list of
test symbols (AAPL, MU, NVDA, TSLA, MSFT). Runs the screener to check Friday signals, runs the Monday entry checks,
and runs the Tuesday exit checks, printing tables of intermediate details for developer logic verification.

Usage:
    python script/verify_week_turnover.py

Side Effects:
    Deletes existing TurnoverTiming signals for '2026-02-06' in data/signals.db to ensure a clean scenario run.
"""

import logging
import os
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.append(os.getcwd())

from app.config import settings
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.trade_manager.strategies.turnover_timing import (
    TurnoverTimingStrategy as TradeManagerTurnover,
)
from app.types import TradeStatus

# Setup Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("VERIFY")
logger.setLevel(logging.INFO)

console = Console()


def verify_scenario():
    # 1. Setup Dependencies
    # settings is already instantiated in config.py

    # Use real databases
    signal_session = DatabaseSession(settings.get_db_path("signals"))
    market_session = DatabaseSession(settings.get_db_path("stocks"))

    trade_repo = TradeRepository(signal_session)
    SignalRepository(signal_session)
    market_provider = MarketDataProvider(market_session)

    screener = TurnoverTimingStrategy(trade_repo, market_provider)
    manager = TradeManagerTurnover()

    # DEBUG: Check Data Availability
    console.rule("[bold yellow]DEBUG: Data Check")
    aapl_hist = market_provider.get_symbol_history("AAPL", days=10)
    if not aapl_hist.empty:
        aapl_hist["date"] = pd.to_datetime(aapl_hist["date"])
        mask = (aapl_hist["date"] >= "2026-02-01") & (aapl_hist["date"] <= "2026-02-10")
        console.print(aapl_hist[mask])
    else:
        console.print("[red]No data found for AAPL[/red]")

    # DEBUG: Analyze specific symbols
    console.rule("[bold yellow]DEBUG: Single Symbol Analysis")
    for symbol in ["AAPL", "MU", "NVDA", "TSLA", "MSFT"]:
        try:
            # analyze_single_symbol uses self.data_provider directly
            # We need to make sure the screener instance has it working.
            # Note: analyze_single_symbol usually fetches fresh data.
            # We might need to monkeypatch "days" inside it if it hardcodes 400 and we have 800?
            # No, get_symbol_history handles it.

            # IMPORTANT: verify_week_turnover uses "TurnoverTimingStrategy" which has "analyze_single_symbol"
            res = screener.analyze_single_symbol(symbol)
            console.print(f"{symbol}: {res}")
        except Exception as e:
            console.print(f"[red]Error analyzing {symbol}: {e}[/red]")

    # DEBUG: Cleanup existing signals to ensure fresh run
    console.rule("[bold red]DEBUG: Cleaning up existing trades for 2026-02-06")

    # Need DB connection for raw delete
    with signal_session.connect() as conn:
        conn.execute(
            "DELETE FROM trades WHERE strategy LIKE 'TurnoverTiming%' AND signal_context LIKE '%2026-02-06%'"
        )
    console.print("[green]Cleanup complete.[/green]")

    # Enable Screener Debugging logic
    screener_logger = logging.getLogger(
        "app.services.screener.strategies.turnover_timing"
    )
    screener_logger.setLevel(logging.INFO)
    if not screener_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        screener_logger.addHandler(handler)

    # ------------------------------------------------------------------
    # STEP 1: Run Screener for Friday, Feb 6th 2026
    # ------------------------------------------------------------------
    console.rule("[bold blue]STEP 1: Screening for Friday 2026-02-06")

    # We want to see if specific tickers are picked.
    # The user listed AAPL, MU, NVDA, TSLA.
    # Let's clean up existing signals for this date/strategy to ensure fresh run if needed,
    # but the screener skips existing. Let's just run it.

    try:
        screener.run(analysis_date="2026-02-06")
    except Exception as e:
        console.print(f"[red]Screener Failed: {e}[/red]")

    # Verify Signals
    table = Table(title="Generated Signals (2026-02-06)")
    table.add_column("Ticker")
    table.add_column("Strategy")
    table.add_column("Entry Price")
    table.add_column("Setup Close")
    table.add_column("ATR")

    # Fetch all created trades for this date context
    created_trades = trade_repo.get_by_status(TradeStatus.CREATED)
    # Filter for our strategy and date
    relevant_trades = []

    targets = ["AAPL", "MU", "NVDA", "TSLA"]

    for trade in created_trades:
        ctx = trade.get("signal_context") or {}
        if isinstance(ctx, str):
            import json

            ctx = json.loads(ctx)

        if (
            ctx.get("setup_date") == "2026-02-06"
            and "TurnoverTiming" in trade["strategy"]
        ):
            relevant_trades.append(trade)
            if trade["symbol"] in targets:
                table.add_row(
                    trade["symbol"],
                    trade["strategy"],
                    f"{float(trade['entry_price']):.2f}",
                    f"{float(ctx.get('setup_close', 0)):.2f}",
                    f"{float(ctx.get('setup_atr', 0)):.2f}",
                )

    console.print(table)

    # ------------------------------------------------------------------
    # STEP 2: Process Monday, Feb 9th 2026 (Entries)
    # ------------------------------------------------------------------
    console.rule("[bold blue]STEP 2: Processing Monday 2026-02-09")

    monday = pd.Timestamp("2026-02-09")

    # Mocking what Engine does: Process CREATED trades
    for trade in relevant_trades:
        symbol = trade["symbol"]
        # Get Data for Monday
        df_hist = market_provider.get_symbol_history(symbol, days=300)
        if df_hist.empty:
            continue

        df_hist["date"] = pd.to_datetime(df_hist["date"])
        df_hist = df_hist.set_index("date").sort_index()

        # Slice up to Monday
        df_slice = df_hist[df_hist.index <= monday]
        if df_slice.empty:
            continue

        # Reset index so 'date' becomes a column in the Series
        candle = df_slice.reset_index().iloc[-1]

        # Check Entry
        if trade["symbol"] in targets:
            console.print(
                f"Checking Entry for {symbol} ({trade['strategy']}) on {monday.date()}..."
            )
            console.print(
                f"Monday Candle: Open={candle['open']}, High={candle['high']}, Low={candle['low']}, Close={candle['close']}"
            )

            result = manager.check_entry(
                trade, candle, df_slice.reset_index(), trade_repo
            )
            if result:
                console.print(f"[green]RESULT: {result}[/green]")
            else:
                console.print("[yellow]RESULT: No Entry[/yellow]")

    # ------------------------------------------------------------------
    # STEP 3: Check Active Status & Exit for Tuesday, Feb 10th 2026
    # ------------------------------------------------------------------
    console.rule("[bold blue]STEP 3: Processing Tuesday 2026-02-10 (Exits)")

    tuesday = pd.Timestamp("2026-02-10")

    active_trades = trade_repo.get_by_status(TradeStatus.ACTIVE)
    console.print(
        f"[bold yellow]DEBUG: Found {len(active_trades)} ACTIVE trades in DB.[/bold yellow]"
    )
    for t in active_trades:
        console.print(f" - {t['symbol']} {t['strategy']} {t['status']}")

    target_active = [
        t
        for t in active_trades
        if t["symbol"] in targets and "TurnoverTiming" in t["strategy"]
    ]

    table_active = Table(
        title="Active Trades Status (After Monday / On Tuesday Morning)"
    )
    table_active.add_column("Ticker")
    table_active.add_column("Strategy")
    table_active.add_column("Entry")
    table_active.add_column("Current Close (Mon)")
    table_active.add_column("Unrealized %")
    table_active.add_column("Action Needed?")

    for trade in target_active:
        symbol = trade["symbol"]
        df_hist = market_provider.get_symbol_history(symbol, days=300)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        df_hist = df_hist.set_index("date").sort_index()

        # Slice including Tuesday (current candle) and Monday (prev)
        # manage_active_trade needs history up to current candle (Tuesday)
        df_slice_tue = df_hist[df_hist.index <= tuesday]

        # Calculate Monday statistics
        df_slice_mon = df_hist[df_hist.index <= monday]
        mon_close = df_slice_mon.iloc[-1]["close"]
        entry_price = float(trade["entry_price"])
        pnl_pct = ((mon_close - entry_price) / entry_price) * 100

        # Check Exit Logic using Tuesday Data
        # We invoke manage_active_trade with Tuesday as "Current High/Low/Open"
        # The logic checks PREVIOUS candles (Mon, Fri).

        action = "NO ACTION"

        # Dry run exit check
        # We need to capture the side effect or return value.
        # But we don't want to actually close it in DB for this verify script unless we mock it.
        # Let's just print what it WOULD do.

        # To strictly verify logic, we can look at Monday (prev_1) and Friday (prev_2)
        # Monday = 2026-02-09
        # Friday = 2026-02-06

        try:
            prev_1 = df_slice_tue.iloc[-2]  # Monday
            prev_2 = df_slice_tue.iloc[-3]  # Friday

            is_green_1 = prev_1["close"] > prev_1["open"]
            is_green_2 = prev_2["close"] > prev_2["open"]

            if is_green_1 and is_green_2:
                action = "EXIT MKT OPG TOMORROW (Triggered Today)"
        except IndexError:
            action = "Not enough history"

        table_active.add_column("Mon Green?", justify="center")
        table_active.add_column("Fri Green?", justify="center")

        table_active.add_row(
            symbol,
            trade["strategy"],
            f"{entry_price:.2f}",
            f"{mon_close:.2f}",
            f"{pnl_pct:.2f}%",
            action,
            str(is_green_1),
            str(is_green_2),
        )

    console.print(table_active)


if __name__ == "__main__":
    verify_scenario()
