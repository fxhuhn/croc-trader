import logging
from typing import Optional

import pandas as pd

from ...database import SignalDatabase
from ..types import Order, OrderLeg
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class DipBuyerStrategy(BaseTradeStrategy):
    def check_entry(
        self, trade: dict, candle: pd.Series, db: SignalDatabase
    ) -> Optional[str]:
        # Nur prüfen, wenn Kerze NACH Entry Datum
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        if candle["date"] <= entry_date_ts:
            return None

        entry_price = float(trade["entry_price"])

        # Klassischer Limit Check (Low <= Limit <= High)
        if candle["low"] <= entry_price <= candle["high"]:
            db.update_trade_status(trade["id"], "ACTIVE")
            return f"✅ **FILLED**: {trade['symbol']} am {candle['date'].strftime('%Y-%m-%d')} zu {entry_price}"
        else:
            db.update_trade_status(trade["id"], "MISSED", "LIMIT_NOT_REACHED")
            return f"❌ **MISSED**: {trade['symbol']} - Limit nicht erreicht."

    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )

        if df_since.empty:
            return None

        current_candle = df_since.iloc[-1]  # Gestern
        trading_days = len(df_since)

        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_price = entry_price + (0.8 * atr)

        exit_reason = None
        exit_price = None

        # 1. Take Profit Check (Intraday High)
        if current_candle["high"] >= tp_price:
            exit_reason = "TAKE_PROFIT"
            exit_price = tp_price

        # 2. LOC Check (Close > High Vortag)
        elif trading_days >= 2:
            prev_candle = df_since.iloc[-2]
            if current_candle["close"] > prev_candle["high"]:
                exit_reason = "LOC_PROFIT"
                exit_price = current_candle["close"]

        # 3. Time Stop (7 Handelstage)
        if not exit_reason and trading_days >= 7:
            exit_reason = "TIME_STOP"
            exit_price = current_candle["close"]

        if exit_reason:
            self._close_trade_in_db(db, trade["id"], exit_reason, exit_price)
            pct = round((exit_price - entry_price) / entry_price * 100, 2)
            return f"💰 **EXIT {trade['symbol']}**: {exit_reason} @ {exit_price:.2f} ({pct}%)"

        return None

    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        status = trade["status"]
        symbol = trade["symbol"]

        # --- A) Entry Order (CREATED) ---
        if status == "CREATED":
            entry_price = float(trade["entry_price"])
            # atr = float(trade["atr_at_entry"])
            # target_price = entry_price + (0.8 * atr)

            # NEU: Vortages-High aus der Historie holen (letzter Eintrag in df_history ist Tag 0)
            # Fallback, falls df_history leer ist (sollte nicht passieren)
            prev_day_high = (
                df_history.iloc[-1]["high"]
                if not df_history.empty
                else (entry_price * 1.05)
            )

            # Quantity berechnen
            qty = max(1, int(budget / entry_price))

            # Update Quantity in DB (WICHTIG: Sofort speichern!)
            db.update_trade_quantity(trade["id"], qty)

            return Order(
                id=f"{symbol}_DIP",
                symbol=symbol,
                qty=qty,
                mode="BRACKET",
                entry=OrderLeg(action="BUY", type="LMT", price=round(entry_price, 2)),
                exits=[
                    # ANPASSUNG: Hier prev_day_high statt target_price verwenden
                    OrderLeg(action="SELL", type="LOC", price=round(prev_day_high, 2))
                ],
            )

        # --- B) Management Order (ACTIVE) ---
        elif status == "ACTIVE":
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            df_since = df_history[df_history["date"] >= entry_date_ts]

            days_passed = len(df_since)
            current_day = days_passed + 1

            if current_day > 7:
                return None  # Sollte schon geschlossen sein

            entry_price = float(trade["entry_price"])
            atr = float(trade["atr_at_entry"])
            qty = int(trade["quantity"])

            # --- SELF HEALING: Falls Qty noch 1 ist (Fehlerbehebung) ---
            if qty <= 1:
                qty = max(1, int(budget / entry_price))
                logger.info(
                    f"[{symbol}] Fixing DipBuyer quantity 1 -> {qty} (Self-Healing)"
                )
                db.update_trade_quantity(trade["id"], qty)

            tp_price = round(entry_price + (0.8 * atr), 2)
            prev_day_high = round(df_since.iloc[-1]["high"], 2)  # High von gestern

            exits = []
            # Order 1: Immer Take Profit (LMT Order basierend auf ATR vom Einstieg)
            exits.append(OrderLeg(action="SELL", type="LMT", price=tp_price, qty=qty))

            # Order 2: Management (LOC oder MOC)
            if current_day == 7:
                # Letzter Tag: Market on Close
                exits.append(OrderLeg(action="SELL", type="MOC", price=0.0, qty=qty))
            else:
                # Tag 2-6: LOC mit Trigger = Prev High
                exits.append(
                    OrderLeg(action="SELL", type="LOC", price=prev_day_high, qty=qty)
                )
            return Order(
                id=f"{symbol}_MGMT_D{current_day}",
                symbol=symbol,
                qty=qty,
                mode="MANAGE",
                entry=None,
                exits=exits,
            )

        return None
