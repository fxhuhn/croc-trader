import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd

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
        # Limit set high to catch all active trades
        all_trades = self.signal_repo.get_all_trades(limit=10000)
        active_trades = [t for t in all_trades if t["state"] in ("pending", "active")]

        logger.info(f"Processing {len(active_trades)} active/pending trades...")

        for trade in active_trades:
            self._process_trade(trade)

    def _process_trade(self, trade: Dict[str, Any]):
        symbol = trade["symbol"]
        # Use signal_timestamp (falling back to timestamp if missing)
        signal_ts = trade.get("signal_timestamp") or trade.get("timestamp")
        if not signal_ts:
            logger.warning(f"Trade {trade['id']} has no timestamp. Skipping.")
            return

        # Normalize date string (handle ISO T separator)
        signal_date = signal_ts.split("T")[0]

        # --- PHASE 1: Initialization (Calculate missing params) ---

        if (
            not trade.get("buy_limit")
            or not trade.get("stop_loss")
            or trade.get("quantity") is None
        ):  # <--- Check for missing quantity
            if not self._initialize_trade_params(trade, signal_date):
                return

        # --- PHASE 2: Simulation (Daily updates) ---
        # Determine start date for simulation
        # If pending, we start looking from the day AFTER the signal
        # If active, we start looking from the day AFTER the entry
        current_ref_date = (
            trade["entry_time"] if trade["state"] == "active" else signal_date
        )

        # Only fetch FUTURE data (> current_ref_date)
        df = self.market_repo.get_data_after_date(
            symbols=[symbol], after_date=current_ref_date.split("T")[0], inclusive=False
        )

        if df.empty:
            return

        df = df.sort_index(level="date")

        for (sym, day_date), row in df.iterrows():
            if trade["state"] in ("closed", "rejected"):
                break

            day_str = day_date.strftime("%Y-%m-%d")

            # Update Mark-to-Market Price
            self.signal_repo.update_trade(
                trade["id"],
                {
                    "current_price": row["close"],
                    "updated_at": datetime.now().isoformat(),
                },
            )

            if trade["state"] == "pending":
                self._check_entry(trade, row, day_str)

            elif trade["state"] == "active":
                self._check_exit(trade, row, day_str)

    def _initialize_trade_params(self, trade: Dict[str, Any], signal_date: str) -> bool:
        candle = self.market_repo.get_candle(trade["symbol"], signal_date)
        if not candle:
            logger.warning(
                f"Signal candle not found for {trade['symbol']} on {signal_date}"
            )
            return False

        high = candle["high"]
        low = candle["low"]

        # Recalculate levels (idempotent, safe to redo)
        buy_limit = high
        stop_loss = low
        risk_range = high - low

        if risk_range <= 0:
            return False

        take_profit = high + (self.tp_mult * risk_range)

        # Calculate Quantity
        risk_amount = buy_limit - stop_loss
        qty = 0
        if risk_amount > 0:
            qty = int(self.risk / risk_amount)

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

    def _check_entry(self, trade: Dict[str, Any], row: pd.Series, day_str: str):
        buy_limit = trade["buy_limit"]
        stop_loss = trade["stop_loss"]

        open_px = row["open"]
        high_px = row["high"]
        low_px = row["low"]

        # Rule: Reject if Low < Stop happens before entry
        # On daily data, we assume worst case if both happen same day: Stop hit first.
        if low_px <= stop_loss:
            # But if Open > Stop and we could have entered...
            # The prompt is strict: "if there is a low below the stopp ... reject"
            # It implies checking the low is a rejection condition.
            self._close_trade(trade, day_str, stop_loss, "rejected")
            return

        # Check Entry
        if high_px >= buy_limit:
            # We entered.
            # If Open > BuyLimit, we fill at Open (Gap Up). Else at BuyLimit.
            fill_price = open_px if open_px > buy_limit else buy_limit

            self.signal_repo.update_trade(
                trade["id"],
                {
                    "state": "active",
                    "entry_time": day_str,
                    "entry_price": fill_price,
                },
            )
            trade["state"] = "active"
            trade["entry_price"] = fill_price

    def _check_exit(self, trade: Dict[str, Any], row: pd.Series, day_str: str):
        stop_loss = trade["stop_loss"]
        take_profit = trade["take_profit"]

        open_px = row["open"]
        high_px = row["high"]
        low_px = row["low"]

        # 1. Check Gaps (Open price determines outcome)
        if open_px <= stop_loss:
            self._close_trade(trade, day_str, open_px, "loss")  # Gap Down
            return
        if open_px >= take_profit:
            self._close_trade(trade, day_str, open_px, "win")  # Gap Up
            return

        # 2. Check Intraday Hits
        # Standard conservative backtest: If Low <= SL, we are out.
        if low_px <= stop_loss:
            self._close_trade(trade, day_str, stop_loss, "loss")
            return

        if high_px >= take_profit:
            self._close_trade(trade, day_str, take_profit, "win")
            return

    def _close_trade(self, trade, date_str, price, state):
        """
        Closes the trade with a specific state (win/loss/rejected).
        """
        self.signal_repo.update_trade(
            trade["id"],
            {
                "state": state,  # Now specific: 'win', 'loss', or 'rejected'
                "exit_time": date_str,
                "exit_price": price,
            },
        )
        trade["state"] = state
        logger.info(f"Trade {trade['symbol']} finished: {state.upper()} at {price}")

    def _close_trade(self, trade, date_str, price, state):
        self.signal_repo.update_trade(
            trade["id"], {"state": state, "exit_time": date_str, "exit_price": price}
        )
        trade["state"] = state
        logger.info(
            f"Trade {trade['id']} for {trade['symbol']} {state} on {date_str} at {price}"
        )
