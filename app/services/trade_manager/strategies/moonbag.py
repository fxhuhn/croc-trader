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
        entry_price = float(trade["entry_price"])
        entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])

        if candle["date"] <= entry_date_ts:
            return None

        # Wenn High >= Entry, dann wurde der Stop Buy ausgelöst
        if candle["high"] >= entry_price:
            db.update_trade_status(trade["id"], "ACTIVE")
            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price}"

        # TimeStop für Entry (z.B. nach 1 Tag nicht abgeholt -> Missed)
        db.update_trade_status(trade["id"], "MISSED", "NO_MOMENTUM")
        return f"❌ **MISSED**: {trade['symbol']} - Stop Buy nicht ausgelöst."

    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        """
        Verwaltet aktive Moonbag Trades:
        1. Stop Loss Prüfung (auf Basis des Lows aus screener_croc)
        2. Time Stop Prüfung (7 Handelstage)
        """
        entry_date = trade["entry_date"]
        entry_date_ts = pd.Timestamp(str(entry_date).split(" ")[0])
        symbol = trade["symbol"]

        # Historie seit Entry
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )
        if df_since.empty:
            return None

        current_candle = df_since.iloc[-1]

        # --- 1. STOP LOSS PRÜFUNG ---
        # Wir brauchen das Low vom Setup-Tag als Stop Loss
        context = self._fetch_croc_context(symbol, entry_date, db)

        if context:
            stop_loss_price = context.low

            # Hat das heutige Low den SL berührt/unterschritten?
            if current_candle["low"] <= stop_loss_price:
                self._close_trade_in_db(db, trade["id"], "STOP_LOSS", stop_loss_price)
                pct = round(
                    (stop_loss_price - float(trade["entry_price"]))
                    / float(trade["entry_price"])
                    * 100,
                    2,
                )
                return f"🛑 **STOP LOSS**: {symbol} @ {stop_loss_price} ({pct}%)"
        else:
            logger.warning(
                f"[{symbol}] Konnte SL-Preis (Context) für Active Trade Management nicht laden."
            )

        # --- 2. TIME STOP PRÜFUNG ---
        if len(df_since) >= 7:
            exit_price = current_candle["close"]
            self._close_trade_in_db(db, trade["id"], "TIME_STOP", exit_price)
            return f"⏰ **TIME STOP**: {symbol} wird geschlossen."

        return None

    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        status = trade["status"]
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        entry_date = trade["entry_date"]

        # --- Kontext laden (für SL Berechnung notwendig) ---
        context = self._fetch_croc_context(symbol, entry_date, db)
        if not context:
            logger.warning(
                f"[{symbol}] Order-Gen abgebrochen: Kein Low in screener_croc gefunden."
            )
            return None

        stop_loss = context.low
        risk_per_share = entry_price - stop_loss

        # Sicherheitscheck
        if risk_per_share <= 0:
            return None

        # --- A) Entry Order (CREATED) ---
        if status == "CREATED":
            # Position Size Calculation
            qty = int(self.RISK_PER_TRADE / risk_per_share)
            qty = max(1, qty)

            target_1r = entry_price + risk_per_share
            qty_half = int(qty * 0.5)

            # Update Quantity in DB
            with db._get_conn() as conn:
                conn.execute(
                    "UPDATE active_trades SET quantity = ? WHERE id = ?",
                    (qty, trade["id"]),
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
                        action="SELL",
                        type="LMT",
                        price=round(target_1r, 2),
                        qty=qty_half,
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

        # --- B) Management Order (ACTIVE) ---
        elif status == "ACTIVE":
            # Orders beibehalten/erneuern für den nächsten Tag
            qty = int(trade["quantity"])
            target_1r = entry_price + risk_per_share
            qty_half = int(qty * 0.5)

            # Prüfung TimeStop (keine Orders mehr wenn Tag 7 vorbei)
            entry_date_ts = pd.Timestamp(str(entry_date).split(" ")[0])
            if not df_history.empty:
                df_since = df_history[df_history["date"] >= entry_date_ts]
                if len(df_since) >= 7:
                    return None

            exits = []

            # 1. Stop Loss (100% der Position)
            # Wir senden qty=None (oder qty), was oft als "Restposition" interpretiert wird,
            # hier explizit qty für Klarheit in der YAML.
            exits.append(
                OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=qty)
            )

            # 2. Take Profit (50% der Position)
            if qty_half > 0:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LMT",
                        price=round(target_1r, 2),
                        qty=qty_half,
                    )
                )

            return Order(
                id=f"{symbol}_MNBG_MGMT",
                symbol=symbol,
                qty=qty,
                mode="MANAGE",
                entry=None,  # Reine Management Order
                exits=exits,
            )

        return None

    def _fetch_croc_context(
        self, symbol: str, date_val, db: SignalDatabase
    ) -> Optional[CrocContext]:
        date_str = str(date_val).split(" ")[0]
        # SQL sucht nach dem passenden Eintrag in der Screener Tabelle
        sql = (
            "SELECT high, low FROM screener_croc WHERE symbol = ? AND date = ? LIMIT 1"
        )
        try:
            with db._get_conn() as conn:
                row = conn.execute(sql, (symbol, date_str)).fetchone()
                if row:
                    return CrocContext(high=float(row["high"]), low=float(row["low"]))
        except Exception as e:
            logger.error(f"Fehler beim Laden des Croc-Context für {symbol}: {e}")
        return None
