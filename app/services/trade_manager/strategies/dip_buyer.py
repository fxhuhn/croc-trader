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
        """
        Prüft, ob der Trade gefüllt wurde.
        Basis: Signal Date. Entry muss strikt NACH Signal Date erfolgen.
        """
        # 1. Signal Datum ermitteln (Fallback auf entry_date für alte Daten)
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])

        # CHECK: Kerze muss NACH dem Signal-Tag liegen (T+1)
        if candle["date"] <= signal_date_ts:
            return None

        entry_price = float(trade["entry_price"])

        # Klassischer Limit Check (Low <= Limit <= High)
        # Annahme: Buy Limit Order
        if candle["low"] <= entry_price <= candle["high"]:
            # DB UPDATE: Status ACTIVE und Entry Date = Heute (Fill Date)
            # Wir nutzen direkt SQL, um entry_date zu überschreiben
            fill_date_str = candle["date"].strftime("%Y-%m-%d")

            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'ACTIVE', entry_date = ? WHERE id = ?",
                        (fill_date_str, trade["id"]),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"DB Update Error on Fill {trade['symbol']}: {e}")

            return (
                f"✅ **FILLED**: {trade['symbol']} am {fill_date_str} zu {entry_price}"
            )

        else:
            # Kein Fill heute
            # Optional: Prüfen ob Limit zu weit weg -> Missed?
            # Hier lassen wir es offen, Manager kümmert sich via Stale Cleanup.
            return None

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        """
        Management Logik.
        df_history enthält alle Daten. Wir schneiden ab Entry Date (Fill Date).
        """
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])

        # df_since enthält den Fill-Tag (Tag 0) und alle folgenden Tage
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )

        if df_since.empty:
            return None

        # Trading Days Zählung:
        # Tag 0 = Fill Tag.
        # Tag 1 = Erster Tag nach Fill.
        trading_days = len(df_since) - 1

        current_candle = df_since.iloc[-1]

        # Preise
        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_price = entry_price + (0.8 * atr)

        # High des Vortages für LOC (Limit on Close)
        # Wenn trading_days == 0 (Fill Tag), gibt es keine "Vortages-Kerze" IM Trade.
        # Wir müssen auf die Historie VOR dem Trade zugreifen.
        # df_history enthält alles. current_candle ist df_history.iloc[-1]
        # prev_day ist df_history.iloc[-2]

        # Safety Check: Haben wir genug Historie?
        if len(df_history) < 2:
            return None

        prev_day_candle = df_history.iloc[-2]
        prev_day_high = float(prev_day_candle["high"])

        exit_reason = None
        exit_price = None

        # --- REGELWERK ---

        # 1. Take Profit
        # Erlaubt ab Tag 0 (Fill Tag), wenn High > TP (Intraday Fill & TP möglich)
        if current_candle["high"] >= tp_price:
            exit_reason = "TAKE_PROFIT"
            exit_price = tp_price

        # 2. LOC Check (Ab Tag 1 aktiv = Mindestens 1 Overnight)
        # Wir wollen verhindern, dass der Trade am selben Tag via LOC geschlossen wird,
        # wenn er gerade erst gefüllt wurde (außer er rennt direkt ins Ziel).
        if not exit_reason and trading_days >= 1:
            if current_candle["close"] > prev_day_high:
                exit_reason = "LOC_PROFIT"
                exit_price = current_candle["close"]

        # 3. Time Stop (Tag 7 Close)
        if not exit_reason and trading_days >= 7:
            exit_reason = "TIME_STOP"
            exit_price = current_candle["close"]

        if exit_reason:
            # Datum für DB
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

        # --- A) Entry Order (CREATED -> Order für T+1) ---
        if status == "CREATED":
            entry_price = float(trade["entry_price"])

            # Vortages-High für LOC Exit Order (Falls am Fill-Tag schon relevant?)
            # Hier nehmen wir das High der letzten bekannten Kerze (Signal-Tag)
            prev_day_high = (
                float(df_history.iloc[-1]["high"])
                if not df_history.empty
                else (entry_price * 1.05)
            )

            qty = max(1, int(budget / entry_price))
            db.update_trade_quantity(trade["id"], qty)

            # Order Setup: Limit Entry + LOC Exit (Bracket)
            # Hinweis: Manche Broker unterstützen LOC nicht als Bracket-Exit.
            # Hier gehen wir davon aus, dass es geht oder die Engine es splittet.
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

        # --- B) Management Order (ACTIVE) ---
        elif status == "ACTIVE":
            # entry_date ist jetzt das echte Fill Date
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])

            # Wie viele Tage sind wir schon im Trade?
            df_since = df_history[df_history["date"] >= entry_date_ts]
            trading_days = len(df_since) - 1  # Tag 0 = Fill

            # Wir generieren Orders für MORGEN.
            next_trading_day_idx = trading_days + 1

            if next_trading_day_idx > 7:
                return None  # Überfällig / TimeStop greift eh

            entry_price = float(trade["entry_price"])
            atr = float(trade["atr_at_entry"])
            qty = int(trade["quantity"])

            if qty <= 1:
                qty = max(1, int(budget / entry_price))
                db.update_trade_quantity(trade["id"], qty)

            tp_price = round(entry_price + (0.8 * atr), 2)
            prev_day_high = round(float(df_history.iloc[-1]["high"]), 2)

            exits = []

            # 1. Take Profit (Immer aktiv)
            exits.append(OrderLeg(action="SELL", type="LMT", price=tp_price, qty=qty))

            # 2. LOC / Time Stop
            # Ab Tag 1 (Overnight) LOC aktivieren
            if next_trading_day_idx >= 1:
                # Am 7. Tag MOC (Market on Close) Exit erzwingen
                if next_trading_day_idx >= 7:
                    exits.append(
                        OrderLeg(action="SELL", type="MOC", price=0.0, qty=qty)
                    )
                else:
                    exits.append(
                        OrderLeg(
                            action="SELL", type="LOC", price=prev_day_high, qty=qty
                        )
                    )

            return Order(
                id=f"{symbol}_MGMT_D{next_trading_day_idx}",
                symbol=symbol,
                qty=qty,
                mode="MANAGE",
                entry=None,
                exits=exits,
            )

        return None
