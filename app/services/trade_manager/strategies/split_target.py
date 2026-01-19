import logging
from typing import Optional, Tuple, override

import pandas as pd

from ...database import SignalDatabase
from ..types import Order, OrderLeg
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class SplitTargetStrategy(BaseTradeStrategy):
    """
    Split Strategie (Smart Blending):
    - Signal Kerze: Tag 0 (Basis für Setup)
    - Entry: Stop Buy @ High der Signal Kerze
    - Stop Loss: Low der Signal Kerze (Initial)
    """

    def _get_risk_params(
        self, trade: dict, df_history: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Berechnet SL, TP1 und TP3 basierend auf der Signalkerze.
        Returns: (sl_price, tp1_price, tp3_price)
        """
        entry_price = float(trade["entry_price"])

        # WICHTIG: Die Parameter basieren auf der SIGNAL-Kerze.
        # Wenn der Trade gefüllt ist, ist entry_date != signal_date.
        # Wir müssen also signal_date nutzen, um die Kerze zu finden.
        target_date_val = trade.get("signal_date") or trade["entry_date"]
        target_date_ts = pd.Timestamp(str(target_date_val).split(" ")[0])

        # Wir suchen die Zeile in der History, die dem signal_date entspricht
        signal_candle = df_history[df_history["date"] == target_date_ts]

        if not signal_candle.empty:
            sl_price = float(signal_candle.iloc[0]["low"])
        else:
            logger.warning(
                f"[{trade['symbol']}] Signalkerze ({target_date_ts}) nicht in History gefunden! Nutze 1% Fallback."
            )
            sl_price = entry_price * 0.99

        if sl_price >= entry_price:
            sl_price = entry_price * 0.99

        risk_per_share = entry_price - sl_price
        tp1_price = entry_price + risk_per_share  # 1R
        tp3_price = entry_price + (3 * risk_per_share)  # 3R

        return round(sl_price, 2), round(tp1_price, 2), round(tp3_price, 2)

    @override
    def check_entry(
        self, trade: dict, candle: pd.Series, db: SignalDatabase
    ) -> Optional[str]:
        # Nur prüfen, wenn Kerze NACH Signal Datum (T+1)
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])

        if candle["date"] <= signal_date_ts:
            return None

        entry_price = float(trade["entry_price"])

        # STOP BUY LOGIK: High >= Entry
        if candle["high"] >= entry_price:
            # DB Update: Fill Datum setzen
            fill_date_str = candle["date"].strftime("%Y-%m-%d")
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'ACTIVE', entry_date = ? WHERE id = ?",
                        (fill_date_str, trade["id"]),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"DB Update Error: {e}")

            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price} am {fill_date_str}"

        return None

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        # Management ab Fill Date
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )

        if df_since.empty:
            return None

        # Zählweise: Tag 0 ist Fill Tag.
        trading_days = len(df_since) - 1

        current_candle = df_since.iloc[-1]
        exit_date_str = current_candle["date"].strftime("%Y-%m-%d")
        trade_id = trade["id"]
        symbol = trade["symbol"]

        # --- DYNAMISCHE PREISBERECHNUNG ---
        sl_price, tp1_price, tp3_price = self._get_risk_params(trade, df_history)

        # --- STATE MANAGEMENT ---
        current_reason = str(trade.get("exit_reason") or "")
        tp1_locked_price: Optional[float] = None

        if "TP1_LOCKED" in current_reason:
            try:
                parts = current_reason.split(":")
                if len(parts) > 1:
                    tp1_locked_price = float(parts[1])
            except ValueError:
                pass

        # --- LOGIK ---

        if tp1_locked_price is None:
            # === PHASE 1: Full Position ===

            # 1. Stop Loss (Initial SL: Low der Signalkerze)
            if current_candle["low"] <= sl_price:
                self._close_trade_in_db(
                    db, trade_id, "STOP_LOSS", sl_price, exit_date=exit_date_str
                )
                return f"🛑 **STOP LOSS**: {symbol} @ {sl_price:.2f} (Signal Low unterschritten)"

            # 2. TP1 (1R)
            if current_candle["high"] >= tp1_price:
                new_state = f"TP1_LOCKED:{tp1_price}"
                self._update_trade_state(db, trade_id, new_state)
                return f"💰 **TP1 HIT**: {symbol} @ {tp1_price:.2f} (1R erreicht)."

        else:
            # === PHASE 2: Half Position ===
            # SL auf Break Even (Entry)
            entry_price = float(trade["entry_price"])
            be_sl = entry_price

            # 1. TP3 (3R)
            if current_candle["high"] >= tp3_price:
                avg_exit_price = (tp1_locked_price + tp3_price) / 2
                self._close_trade_in_db(
                    db, trade_id, "TP1_TP3_WIN", avg_exit_price, exit_date=exit_date_str
                )
                return f"🚀 **TP3 HIT**: {symbol}. Blended Exit: {avg_exit_price:.2f}"

            # 2. Break Even Stop
            if current_candle["low"] <= be_sl:
                avg_exit_price = (tp1_locked_price + be_sl) / 2
                self._close_trade_in_db(
                    db, trade_id, "TP1_BE_STOP", avg_exit_price, exit_date=exit_date_str
                )
                return f"🛡️ **BE STOP**: {symbol} Rest ausgestoppt. Blended Exit: {avg_exit_price:.2f}"

        # Time Stop (10 Tage nach Entry)
        if trading_days >= 10:
            current_price = current_candle["close"]
            final_price = current_price
            reason = "TIME_STOP"

            if tp1_locked_price:
                final_price = (tp1_locked_price + current_price) / 2
                reason = "TIME_STOP_PARTIAL"

            self._close_trade_in_db(
                db, trade_id, reason, final_price, exit_date=exit_date_str
            )
            return f"⏰ **TIME STOP**: {symbol} geschlossen nach 10 Tagen."

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
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])

        # Parameter holen
        sl_price, tp1_price, tp3_price = self._get_risk_params(trade, df_history)
        risk_per_share = entry_price - sl_price

        current_reason = str(trade.get("exit_reason") or "")
        is_phase_2 = "TP1_LOCKED" in current_reason
        current_qty = int(trade["quantity"])

        active_qty = current_qty
        if is_phase_2:
            active_qty = max(1, current_qty // 2)

        if trade["status"] == "CREATED":
            # Risk Based Quantity (100$ Risk)
            risk_budget = 100.0
            if risk_per_share > 0:
                qty_total = int(risk_budget / risk_per_share)
            else:
                qty_total = 1
            qty_total = max(2, qty_total)
            db.update_trade_quantity(trade["id"], qty_total)

            exits = [
                OrderLeg(action="SELL", type="STP", price=sl_price, qty=None),
            ]

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
                qty_tp3 = active_qty - qty_tp1

                exits.append(
                    OrderLeg(action="SELL", type="STP", price=sl_price, qty=active_qty)
                )
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp1_price, qty=qty_tp1)
                )
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp3_price, qty=qty_tp3)
                )
            else:
                be_sl = entry_price
                exits.append(
                    OrderLeg(action="SELL", type="STP", price=be_sl, qty=active_qty)
                )
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp3_price, qty=active_qty)
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
