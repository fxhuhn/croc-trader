import logging
import sqlite3
from typing import Optional, Tuple, override

import pandas as pd

from ...database import SignalDatabase
from ..types import Order, OrderLeg, TradeParams
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class SplitTargetStrategy(BaseTradeStrategy):
    @override
    def get_current_params(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[TradeParams]:
        try:
            sl, tp1, tp3 = self._get_risk_params(trade, df_history)
            current_reason = str(trade.get("exit_reason") or "")
            if "TP1_LOCKED" in current_reason:
                sl = float(trade["entry_price"])

            return TradeParams(stop_loss=sl, tp_1=tp1, tp_2=tp3)
        except Exception:
            return None

    def _get_risk_params(
        self, trade: dict, df_history: pd.DataFrame
    ) -> Tuple[float, float, float]:
        entry_price = float(trade["entry_price"])
        target_date_val = trade.get("signal_date") or trade["entry_date"]
        target_date_ts = pd.Timestamp(str(target_date_val).split(" ")[0])

        signal_candle = df_history[df_history["date"] == target_date_ts]

        if not signal_candle.empty:
            sl_price = float(signal_candle.iloc[0]["low"])
        else:
            sl_price = entry_price * 0.99

        if sl_price >= entry_price:
            sl_price = entry_price * 0.99

        risk_per_share = entry_price - sl_price
        tp1_price = entry_price + risk_per_share
        tp3_price = entry_price + (3 * risk_per_share)

        return round(sl_price, 2), round(tp1_price, 2), round(tp3_price, 2)

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

        # Wait for T+1
        if candle["date"] <= signal_date_ts:
            return None

        entry_price = float(trade["entry_price"])

        # --- INVALIDATION CHECK ---
        # Calculate where the SL would be. If we hit it before filling, the setup is dead.
        sl_price, _, _ = self._get_risk_params(trade, df_history)

        if candle["low"] <= sl_price:
            # Setup Broken
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'MISSED', exit_reason = 'SETUP_INVALIDATED' WHERE id = ?",
                        (trade["id"],),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"DB Error Invalidating Trade {trade['symbol']}: {e}")

            return f"❌ **INVALIDATED**: {trade['symbol']} Low ({candle['low']}) broke Setup Low ({sl_price})."

        # --- ENTRY CHECK ---
        if candle["high"] >= entry_price:
            fill_date_str = candle["date"].strftime("%Y-%m-%d")
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'ACTIVE', entry_date = ? WHERE id = ?",
                        (fill_date_str, trade["id"]),
                    )
                    conn.commit()
            except sqlite3.IntegrityError:
                logger.warning(
                    f"[{trade['symbol']}] Date Collision on Fill. Marking ACTIVE."
                )
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
                logger.error(f"DB Update Error: {e}")

            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price} am {fill_date_str}"

        return None

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        # ... (Manage logic remains same) ...
        # Copied from previous correct version for completeness
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )

        if df_since.empty:
            return None
        trading_days = len(df_since) - 1
        current_candle = df_since.iloc[-1]
        exit_date_str = current_candle["date"].strftime("%Y-%m-%d")
        trade_id = trade["id"]
        symbol = trade["symbol"]

        sl_price, tp1_price, tp3_price = self._get_risk_params(trade, df_history)
        current_reason = str(trade.get("exit_reason") or "")
        tp1_locked_price: Optional[float] = None

        if "TP1_LOCKED" in current_reason:
            try:
                tp1_locked_price = float(current_reason.split(":")[1])
            except:
                pass

        if tp1_locked_price is None:
            if current_candle["low"] <= sl_price:
                self._close_trade_in_db(
                    db, trade_id, "STOP_LOSS", sl_price, exit_date=exit_date_str
                )
                return f"🛑 **STOP LOSS**: {symbol} @ {sl_price:.2f}"
            if current_candle["high"] >= tp1_price:
                new_state = f"TP1_LOCKED:{tp1_price}"
                self._update_trade_state(db, trade_id, new_state)
                return f"💰 **TP1 HIT**: {symbol} @ {tp1_price:.2f}"
        else:
            be_sl = float(trade["entry_price"])
            if current_candle["high"] >= tp3_price:
                avg = (tp1_locked_price + tp3_price) / 2
                self._close_trade_in_db(
                    db, trade_id, "TP1_TP3_WIN", avg, exit_date=exit_date_str
                )
                return f"🚀 **TP3 HIT**: {symbol}"
            if current_candle["low"] <= be_sl:
                avg = (tp1_locked_price + be_sl) / 2
                self._close_trade_in_db(
                    db, trade_id, "TP1_BE_STOP", avg, exit_date=exit_date_str
                )
                return f"🛡️ **BE STOP**: {symbol}"

        if trading_days >= 10:
            final = current_candle["close"]
            reason = "TIME_STOP"
            if tp1_locked_price:
                final = (tp1_locked_price + final) / 2
                reason = "TIME_STOP_PARTIAL"
            self._close_trade_in_db(
                db, trade_id, reason, final, exit_date=exit_date_str
            )
            return f"⏰ **TIME STOP**: {symbol}"
        return None

    def _update_trade_state(self, db: SignalDatabase, trade_id: int, reason: str):
        with db._get_conn() as conn:
            conn.execute(
                "UPDATE active_trades SET exit_reason = ? WHERE id = ?",
                (reason, trade_id),
            )
            conn.commit()

    @override
    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        # ... (Generate orders remains same) ...
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        sl_price, tp1, tp3 = self._get_risk_params(trade, df_history)
        risk_per_share = entry_price - sl_price
        current_reason = str(trade.get("exit_reason") or "")
        is_phase_2 = "TP1_LOCKED" in current_reason
        current_qty = int(trade["quantity"])
        active_qty = current_qty if not is_phase_2 else max(1, current_qty // 2)

        if trade["status"] == "CREATED":
            risk_budget = 100.0
            qty_total = int(risk_budget / risk_per_share) if risk_per_share > 0 else 1
            qty_total = max(2, qty_total)
            db.update_trade_quantity(trade["id"], qty_total)
            exits = [OrderLeg(action="SELL", type="STP", price=sl_price, qty=None)]
            return Order(
                id=f"{symbol}_SPLIT_INIT",
                symbol=symbol,
                qty=qty_total,
                mode="BRACKET",
                entry=OrderLeg(action="BUY", type="STP", price=entry_price),
                exits=exits,
            )
        elif trade["status"] == "ACTIVE":
            exits = []
            if not is_phase_2:
                qty_tp1 = active_qty // 2
                exits.append(
                    OrderLeg(action="SELL", type="STP", price=sl_price, qty=active_qty)
                )
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp1, qty=qty_tp1)
                )
                exits.append(
                    OrderLeg(
                        action="SELL", type="LMT", price=tp3, qty=active_qty - qty_tp1
                    )
                )
            else:
                exits.append(
                    OrderLeg(
                        action="SELL", type="STP", price=entry_price, qty=active_qty
                    )
                )
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp3, qty=active_qty)
                )
            return Order(
                id=f"{symbol}_SPLIT_MGMT",
                symbol=symbol,
                qty=active_qty,
                mode="MANAGE",
                entry=None,
                exits=exits,
            )
        return None
