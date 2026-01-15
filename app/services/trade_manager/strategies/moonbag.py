import logging
from typing import Optional

import pandas as pd

from ...database import SignalDatabase
from ..types import CrocContext, Order, OrderLeg
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class MoonbagStrategy(BaseTradeStrategy):
    RISK_PER_TRADE = 100.0  # Fixe Konstante

    def check_entry(
        self, trade: dict, candle: pd.Series, db: SignalDatabase
    ) -> Optional[str]:
        # Stop Buy Logic: Wir kaufen, wenn Kurs >= Entry (High berührt Entry)
        # Entry ist hier das High der Signalkerze
        entry_price = float(trade["entry_price"])

        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        if candle["date"] <= entry_date_ts:
            return None

        # Wenn High >= Entry, dann wurde der Stop Buy ausgelöst
        if candle["high"] >= entry_price:
            db.update_trade_status(trade["id"], "ACTIVE")
            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price}"

        # TimeStop für Entry (z.B. nach 1 Tag nicht abgeholt -> Missed)
        # Hier einfache Logik: Wenn nicht heute gefüllt, dann vorbei (optional)
        db.update_trade_status(trade["id"], "MISSED", "NO_MOMENTUM")
        return f"❌ **MISSED**: {trade['symbol']} - Stop Buy nicht ausgelöst."

    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        # Moonbag hat aktuell im Code nur den Time-Stop Fallback (7 Tage Kalender)
        # Exits liegen als Bracket beim Broker.
        # Hier implementieren wir den einfachen Time-Stop basierend auf Handelstagen.
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts]

        if len(df_since) >= 7:
            price = df_since.iloc[-1]["close"]
            self._close_trade_in_db(db, trade["id"], "TIME_STOP", price)
            return f"⏰ **TIME STOP**: {trade['symbol']} wird geschlossen."

        return None

    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        if trade["status"] != "CREATED":
            return None  # Active Moonbags werden vom Broker gemanaged (Bracket)

        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        entry_date = trade["entry_date"]

        # Kontext (Low für Stop Loss) nachladen
        context = self._fetch_croc_context(symbol, entry_date, db)
        if not context:
            logger.warning(
                f"[{symbol}] Moonbag übersprungen: Kein Low in screener_croc gefunden."
            )
            return None

        stop_loss = context.low
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return None

        # Position Size Calculation
        qty = int(self.RISK_PER_TRADE / risk_per_share)
        qty = max(1, qty)

        target_1r = entry_price + risk_per_share
        qty_half = int(qty * 0.5)

        # Update Quantity in DB
        with db._get_conn() as conn:
            conn.execute(
                "UPDATE active_trades SET quantity = ? WHERE id = ?", (qty, trade["id"])
            )
            conn.commit()

        exits = []
        # 1. Stop Loss (Full)
        exits.append(
            OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=None)
        )

        # 2. Take Profit (Half)
        if qty_half > 0:
            exits.append(
                OrderLeg(
                    action="SELL", type="LMT", price=round(target_1r, 2), qty=qty_half
                )
            )

        return Order(
            id=f"{symbol}_MNBG",
            symbol=symbol,
            qty=qty,
            mode="BRACKET",
            entry=OrderLeg(action="BUY", type="STP", price=round(entry_price, 2)),
            exits=exits,
        )

    def _fetch_croc_context(
        self, symbol: str, date_val, db: SignalDatabase
    ) -> Optional[CrocContext]:
        date_str = str(date_val).split(" ")[0]
        sql = (
            "SELECT high, low FROM screener_croc WHERE symbol = ? AND date = ? LIMIT 1"
        )
        try:
            with db._get_conn() as conn:
                row = conn.execute(sql, (symbol, date_str)).fetchone()
                if row:
                    return CrocContext(high=float(row["high"]), low=float(row["low"]))
        except Exception:
            pass
        return None
