import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Assume these imports exist in your project structure
from app.core.protocols import SignalRepository
from core.database import OHLCVRepository

logger = logging.getLogger(__name__)


class PaperTrader:
    def __init__(
        self,
        signal_repo: SignalRepository,
        market_repo: OHLCVRepository,
        risk_per_trade: float = 100.0,
        tp_multiple: float = 3.0,
    ):
        self.signal_repo = signal_repo
        self.market_repo = market_repo
        self.risk = risk_per_trade
        self.tp_mult = tp_multiple

    def run(self):
        """Main execution method to update all pending/active trades."""
        # 1. Fetch all relevant trades
        all_trades = self.signal_repo.get_all_trades(limit=10000)
        active_trades = [
            t for t in all_trades if t.get("state") in ("pending", "active")
        ]

        if not active_trades:
            logger.info("No active trades to process.")
            return

        logger.info(f"Processing {len(active_trades)} active/pending trades...")

        # 2. Optimization: Batch fetch market data for all symbols at once
        # Find the earliest date needed to minimize data requests
        unique_symbols = {t["symbol"] for t in active_trades}
        min_date_str = min(
            t.get("entry_time") or t.get("signal_timestamp", datetime.now().isoformat())
            for t in active_trades
        ).split("T")[0]

        # Fetch data dictionary: { "AAPL": DataFrame, "TSLA": DataFrame }
        market_data_map = self._batch_fetch_market_data(
            list(unique_symbols), min_date_str
        )

        # 3. Process trades using cached data
        for trade in active_trades:
            self._process_trade(trade, market_data_map.get(trade["symbol"]))

    def _batch_fetch_market_data(
        self, symbols: list[str], start_date: str
    ) -> dict[str, pd.DataFrame]:
        """Fetches data for multiple symbols and returns a dict of DataFrames."""
        if not symbols:
            return {}

        # Assuming market_repo can handle bulk fetch or we loop here efficiently
        # If your repo only supports one-by-one, at least we do it once per symbol, not per trade
        data_map = {}
        for symbol in symbols:
            df = self.market_repo.get_data_after_date(
                symbols=[symbol], after_date=start_date, inclusive=False
            )
            if not df.empty:
                # Ensure index is sorted for performance
                data_map[symbol] = df.sort_index(level="date")
        return data_map

    def _process_trade(self, trade: dict[str, Any], df: pd.DataFrame | None):
        if df is None or df.empty:
            return

        symbol = trade["symbol"]
        # Normalize signal date
        raw_ts = trade.get("signal_timestamp") or trade.get("timestamp")
        if not raw_ts:
            logger.warning(f"Trade {trade['id']} missing timestamp. Skipping.")
            return

        signal_date_str = raw_ts.split("T")[0]

        # PHASE 1: Initialization
        if not all(k in trade for k in ("buy_limit", "stop_loss", "quantity")):
            if not self._initialize_trade_params(trade, signal_date_str):
                return

        # PHASE 2: Simulation
        # Determine where to start checking in the dataframe
        start_check_date = (
            trade["entry_time"] if trade["state"] == "active" else signal_date_str
        )

        # Filter dataframe to only relevant future rows
        # df index is likely (symbol, date) or just date. Adjust access accordingly.
        try:
            # fast slicing using the MultiIndex or Date index
            relevant_data = (
                df.xs(symbol, level="symbol", drop_level=False)
                if "symbol" in df.index.names
                else df
            )
            relevant_data = relevant_data[
                relevant_data.index.get_level_values("date") > start_check_date
            ]
        except KeyError:
            return

        for idx, row in relevant_data.iterrows():
            if trade["state"] in ("closed", "rejected"):
                break

            # idx is (symbol, date) or date depending on repo implementation
            current_date = idx[1] if isinstance(idx, tuple) else idx
            day_str = current_date.strftime("%Y-%m-%d")

            # Mark-to-Market Update
            self._update_current_price(trade, row["close"])

            if trade["state"] == "pending":
                self._check_entry(trade, row, day_str)
            elif trade["state"] == "active":
                self._check_exit(trade, row, day_str)

    def _update_current_price(self, trade: dict, price: float):
        # Only update DB if price changed significantly or periodically?
        # For now, we keep it simple but separate logic from loop.
        self.signal_repo.update_trade(
            trade["id"],
            {
                "current_price": price,
                "updated_at": datetime.now().isoformat(),
            },
        )

    def _initialize_trade_params(self, trade: dict[str, Any], signal_date: str) -> bool:
        candle = self.market_repo.get_candle(trade["symbol"], signal_date)
        if not candle:
            logger.warning(f"No signal candle for {trade['symbol']} on {signal_date}")
            return False

        high, low = candle["high"], candle["low"]
        risk_range = high - low

        if risk_range <= 0:
            return False

        buy_limit = high
        stop_loss = low
        take_profit = high + (self.tp_mult * risk_range)

        risk_amount = buy_limit - stop_loss
        qty = int(self.risk / risk_amount) if risk_amount > 0 else 0

        updates = {
            "buy_limit": buy_limit,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": qty,
            "updated_at": datetime.now().isoformat(),
        }

        self.signal_repo.update_trade(trade["id"], updates)
        trade.update(updates)
        return True

    def _check_entry(self, trade: dict, row: pd.Series, day_str: str):
        buy_limit = trade["buy_limit"]
        stop_loss = trade["stop_loss"]

        # Rule: Low < Stop before entry = Rejected
        if row["low"] <= stop_loss:
            self._close_trade(trade, day_str, stop_loss, "rejected")
            return

        # Rule: High >= Buy Limit = Entry
        if row["high"] >= buy_limit:
            # Gap up logic: Fill at Open if Open > BuyLimit
            fill_price = max(row["open"], buy_limit)

            updates = {
                "state": "active",
                "entry_time": day_str,
                "entry_price": fill_price,
            }
            self.signal_repo.update_trade(trade["id"], updates)
            trade.update(updates)

    def _check_exit(self, trade: dict, row: pd.Series, day_str: str):
        stop_loss = trade["stop_loss"]
        take_profit = trade["take_profit"]

        # Gap Checks (Open price priority)
        if row["open"] <= stop_loss:
            self._close_trade(trade, day_str, row["open"], "loss")
            return
        if row["open"] >= take_profit:
            self._close_trade(trade, day_str, row["open"], "win")
            return

        # Intraday Checks
        if row["low"] <= stop_loss:
            self._close_trade(trade, day_str, stop_loss, "loss")
            return
        if row["high"] >= take_profit:
            self._close_trade(trade, day_str, take_profit, "win")
            return

    def _close_trade(self, trade: dict, date_str: str, price: float, state: str):
        updates = {
            "state": state,
            "exit_time": date_str,
            "exit_price": price,
        }
        self.signal_repo.update_trade(trade["id"], updates)
        trade.update(updates)
        logger.info(f"Trade {trade['symbol']} finished: {state.upper()} at {price}")
