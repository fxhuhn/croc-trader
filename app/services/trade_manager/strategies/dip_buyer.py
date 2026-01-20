import logging
import sqlite3
from typing import Optional, override

import pandas as pd

from ...database import SignalDatabase
from ..types import Order, OrderLeg, TradeParams
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class DipBuyerStrategy(BaseTradeStrategy):
    @override
    def get_current_params(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[TradeParams]:
        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_target = entry_price + (0.8 * atr)

        prev_high = 0.0
        if len(df_history) >= 2:
            prev_high = float(df_history.iloc[-2]["high"])

        return TradeParams(
            stop_loss=0.0, tp_1=tp_target, extras={"threshold_loc": prev_high}
        )

    @override
    def check_entry(
        self,
        trade: dict,
        candle: pd.Series,
        df_history: pd.DataFrame,
        db: SignalDatabase,
    ) -> Optional[str]:
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])
        current_date_ts = pd.Timestamp(candle["date"])

        # 1. Wait for T+1
        if current_date_ts <= signal_date_ts:
            return None

        # 2. Identify the Valid Trading Day (T+1 relative to Signal)
        future_history = df_history[df_history["date"] > signal_date_ts]
        if future_history.empty:
            return None
        next_valid_day = future_history.iloc[0]["date"]

        # 3. Expiration Logic
        # A) If we skipped past the valid day (e.g. Gap or data skip), Expire.
        if current_date_ts > next_valid_day:
            self._expire_trade(db, trade["id"])
            return (
                f"❌ **EXPIRED**: {trade['symbol']} Entry missed (Date > Valid Date)."
            )

        # B) If this IS the valid day, we check conditions.
        # If conditions fail, we EXPIRE it at the end of this function.
        is_valid_day = current_date_ts == next_valid_day

        entry_price = float(trade["entry_price"])

        # 4. Entry Trigger Check
        if candle["low"] <= entry_price <= candle["high"]:
            fill_date_str = candle["date"].strftime("%Y-%m-%d")
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'ACTIVE', entry_date = ? WHERE id = ?",
                        (fill_date_str, trade["id"]),
                    )
                    conn.commit()
            except sqlite3.IntegrityError:
                # Handle Collision
                try:
                    with db._get_conn() as conn:
                        conn.execute(
                            "UPDATE active_trades SET status = 'ACTIVE' WHERE id = ?",
                            (trade["id"],),
                        )
                        conn.commit()
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"DB Error on Fill {trade['symbol']}: {e}")

            return (
                f"✅ **FILLED**: {trade['symbol']} am {fill_date_str} zu {entry_price}"
            )

        # 5. Strict Expiration on Valid Day
        # If we are on the valid day (Backfill candle = closed day) and didn't fill above:
        # The opportunity is gone. Expire immediately.
        if is_valid_day:
            self._expire_trade(db, trade["id"])
            return f"❌ **EXPIRED**: {trade['symbol']} Price did not reach entry on valid day."

        return None

    def _expire_trade(self, db: SignalDatabase, trade_id: int):
        try:
            with db._get_conn() as conn:
                conn.execute(
                    "UPDATE active_trades SET status = 'MISSED', exit_reason = 'EXPIRED' WHERE id = ?",
                    (trade_id,),
                )
                conn.commit()
        except Exception:
            pass

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )
        if df_since.empty:
            return None

        trading_days = len(df_since) - 1
        current_candle = df_since.iloc[-1]

        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_target = entry_price + (0.8 * atr)

        if len(df_history) < 2:
            return None
        prev_day_high = float(df_history.iloc[-2]["high"])

        exit_reason = None
        exit_price = None

        if trading_days >= 1:
            if current_candle["high"] >= tp_target:
                exit_reason, exit_price = "TAKE_PROFIT_ATR", tp_target
            elif current_candle["close"] > prev_day_high:
                exit_reason, exit_price = "LOC_PROFIT", current_candle["close"]
            elif trading_days >= 7:
                exit_reason, exit_price = "TIME_STOP", current_candle["close"]

        if exit_reason:
            exit_date_str = current_candle["date"].strftime("%Y-%m-%d")
            self._close_trade_in_db(
                db, trade["id"], exit_reason, exit_price, exit_date=exit_date_str
            )
            pct = round((exit_price - entry_price) / entry_price * 100, 2)
            return f"💰 **EXIT {trade['symbol']}**: {exit_reason} @ {exit_price:.2f} ({pct}%)"

        return None

    @override
    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        status = trade["status"]
        symbol = trade["symbol"]

        if status == "CREATED":
            entry_price = float(trade["entry_price"])
            qty = max(1, int(budget / entry_price))
            db.update_trade_quantity(trade["id"], qty)
            return Order(
                id=f"{symbol}_DIP_ENTRY",
                symbol=symbol,
                qty=qty,
                mode="BRACKET",
                entry=OrderLeg(action="BUY", type="LMT", price=round(entry_price, 2)),
                exits=[],
            )

        elif status == "ACTIVE":
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            df_since = df_history[df_history["date"] >= entry_date_ts]
            trading_days = len(df_since) - 1
            next_day_idx = trading_days + 1
            if next_day_idx > 7:
                return None

            entry_price = float(trade["entry_price"])
            atr = float(trade["atr_at_entry"])
            qty = int(trade["quantity"])
            tp_target = round(entry_price + (0.8 * atr), 2)
            prev_day_high = round(float(df_history.iloc[-1]["high"]), 2)

            exits = []
            exits.append(OrderLeg(action="SELL", type="LMT", price=tp_target, qty=qty))

            if next_day_idx >= 7:
                exits.append(OrderLeg(action="SELL", type="MOC", price=0.0, qty=qty))
            else:
                exits.append(
                    OrderLeg(action="SELL", type="LOC", price=prev_day_high, qty=qty)
                )

            return Order(
                id=f"{symbol}_MGMT_D{next_day_idx}",
                symbol=symbol,
                qty=qty,
                mode="MANAGE",
                entry=None,
                exits=exits,
            )

        return None
