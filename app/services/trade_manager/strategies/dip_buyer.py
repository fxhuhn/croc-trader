import logging
from typing import Optional, override

import pandas as pd

from ...database import SignalDatabase
from ..types import Order, OrderLeg
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class DipBuyerStrategy(BaseTradeStrategy):
    @override
    def check_entry(
        self, trade: dict, candle: pd.Series, db: SignalDatabase
    ) -> Optional[str]:
        # Nur prüfen, wenn Kerze NACH Entry Datum (Tag 0)
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

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        """
        Management Logik (Korrigierte Zählung).
        - Tag 0: Signal Tag (entry_date)
        - Tag 1: Execution Tag (entry_date + 1 Handelstag) -> Nur LOC
        - Tag 2: Erster "Haltedag" -> TP erlaubt + LOC
        """
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        # df_since enthält [SignalTag, Tag1, Tag2, ...]
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )

        if df_since.empty:
            return None

        # WICHTIG: Wir ziehen 1 ab, da der erste Eintrag der Signal-Tag (Tag 0) ist.
        # Wenn df_since 2 Einträge hat (Signal + Heute), ist trading_days = 1.
        trading_days = len(df_since) - 1

        # Wir können erst managen, wenn wir mindestens in Tag 1 sind
        if trading_days < 1:
            return None

        current_candle = df_since.iloc[-1]

        # Preise
        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_price = entry_price + (0.8 * atr)

        # High des Vortages für LOC (Current Candle ist -1, Vortag ist -2)
        # An Tag 1 ist der Vortag der Signal-Tag. Das ist korrekt.
        prev_day_high = float(df_since.iloc[-2]["high"])

        exit_reason = None
        exit_price = None

        # --- REGELWERK ---

        # 1. Take Profit (Erst ab Tag 2!)
        # An Tag 1 (trading_days=1) ist diese Bedingung FALSE.
        if trading_days >= 2:
            if current_candle["high"] >= tp_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = tp_price

        # 2. LOC Check (Ab Tag 1 aktiv)
        # Close > High des Vortages (beim Einstieg ist das High der Signalkerze)
        if not exit_reason:
            if current_candle["close"] > prev_day_high:
                exit_reason = "LOC_PROFIT"
                exit_price = current_candle["close"]

        # 3. Time Stop (Tag 7 Close)
        if not exit_reason and trading_days >= 7:
            exit_reason = "TIME_STOP"
            exit_price = current_candle["close"]

        if exit_reason:
            # Datum für die Datenbank
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

        # --- A) Entry Order (CREATED -> Order für Tag 1) ---
        if status == "CREATED":
            entry_price = float(trade["entry_price"])

            # Vortages-High (Das ist hier das High der Signalkerze = Tag 0)
            prev_day_high = (
                float(df_history.iloc[-1]["high"])
                if not df_history.empty
                else (entry_price * 1.05)
            )

            qty = max(1, int(budget / entry_price))
            db.update_trade_quantity(trade["id"], qty)

            # Order für Tag 1: NUR LOC Exit, kein TP
            return Order(
                id=f"{symbol}_DIP_ENTRY",
                symbol=symbol,
                qty=qty,
                mode="BRACKET",
                entry=OrderLeg(action="BUY", type="LMT", price=round(entry_price, 2)),
                exits=[
                    OrderLeg(action="SELL", type="LOC", price=round(prev_day_high, 2))
                ],
            )

        # --- B) Management Order (ACTIVE -> Order für Tag X) ---
        elif status == "ACTIVE":
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            df_since = df_history[df_history["date"] >= entry_date_ts]

            # Zählung: df_since enthält [Tag0, Tag1 ... Heute]
            # Beispiel: Wir haben Tag 1 abgeschlossen (Len=2).
            # Wir generieren die Order für MORGEN (Tag 2).
            # Len 2 entspricht also genau dem "nächsten" Trading Tag Index.
            next_trading_day = len(df_since)

            if next_trading_day > 7:
                return None  # Überfällig

            entry_price = float(trade["entry_price"])
            atr = float(trade["atr_at_entry"])
            qty = int(trade["quantity"])

            if qty <= 1:
                qty = max(1, int(budget / entry_price))
                db.update_trade_quantity(trade["id"], qty)

            tp_price = round(entry_price + (0.8 * atr), 2)
            prev_day_high = round(float(df_history.iloc[-1]["high"]), 2)

            exits = []

            # 1. Take Profit (Ab Tag 2 aktiv)
            # Da wir die Order für Tag 2+ schreiben, ist das hier immer TRUE.
            # (Denn ACTIVE heißt, wir haben Tag 1 schon hinter uns).
            if next_trading_day >= 2:
                exits.append(
                    OrderLeg(action="SELL", type="LMT", price=tp_price, qty=qty)
                )

            # 2. LOC / Time Stop
            if next_trading_day == 7:
                exits.append(OrderLeg(action="SELL", type="MOC", price=0.0, qty=qty))
            else:
                exits.append(
                    OrderLeg(action="SELL", type="LOC", price=prev_day_high, qty=qty)
                )

            return Order(
                id=f"{symbol}_MGMT_D{next_trading_day}",
                symbol=symbol,
                qty=qty,
                mode="MANAGE",
                entry=None,
                exits=exits,
            )

        return None
