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

        # FIX: Nutzung von signal_date für T+1 Check
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])

        # Einstieg erst am Tag NACH dem Signal erlaubt
        if candle["date"] <= signal_date_ts:
            return None

        # Wenn High >= Entry, dann wurde der Stop Buy ausgelöst
        if candle["high"] >= entry_price:
            # DB UPDATE: Setze entry_date auf den ECHTEN Fill-Tag
            fill_date_str = candle["date"].strftime("%Y-%m-%d")
            try:
                with db._get_conn() as conn:
                    conn.execute(
                        "UPDATE active_trades SET status = 'ACTIVE', entry_date = ? WHERE id = ?",
                        (fill_date_str, trade["id"]),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Moonbag Fill Error: {e}")

            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price} am {fill_date_str}"

        # TimeStop für Entry (z.B. nach 3 Tagen nicht abgeholt -> Missed)
        # Hier optional, aktuell im Code noch nicht strikt forciert, außer Manager cleanup.
        return None

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

        # Historie seit Entry (Fill Date)
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )
        if df_since.empty:
            return None

        current_candle = df_since.iloc[-1]
        exit_date_str = current_candle["date"].strftime("%Y-%m-%d")

        # --- 1. STOP LOSS PRÜFUNG ---
        # WICHTIG: SL kommt aus dem Screener Context vom SIGNAL DATE
        # Daher müssen wir das Signal Date kennen.
        signal_date = trade.get("signal_date") or entry_date  # Fallback

        context = self._fetch_croc_context(symbol, signal_date, db)

        if context:
            stop_loss_price = context.low

            # Hat das heutige Low den SL berührt/unterschritten?
            if current_candle["low"] <= stop_loss_price:
                self._close_trade_in_db(
                    db,
                    trade["id"],
                    "STOP_LOSS",
                    stop_loss_price,
                    exit_date=exit_date_str,
                )
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

        # --- 2. TIME STOP PRÜFUNG (7 Tage ab Fill) ---
        if len(df_since) >= 7:
            exit_price = current_candle["close"]
            self._close_trade_in_db(
                db, trade["id"], "TIME_STOP", exit_price, exit_date=exit_date_str
            )
            return f"⏰ **TIME STOP**: {symbol} wird geschlossen."

        return None

    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        status = trade["status"]
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])

        # Context laden via Signal Date
        signal_date = trade.get("signal_date") or trade["entry_date"]
        context = self._fetch_croc_context(symbol, signal_date, db)

        if not context:
            logger.warning(
                f"[{symbol}] Order-Gen abgebrochen: Kein Low in screener_croc gefunden."
            )
            return None

        stop_loss = context.low
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return None

        # --- A) Entry Order (CREATED) ---
        if status == "CREATED":
            qty = int(self.RISK_PER_TRADE / risk_per_share)
            qty = max(1, qty)

            target_1r = entry_price + risk_per_share
            qty_half = int(qty * 0.5)

            db.update_trade_quantity(trade["id"], qty)

            exits = []
            exits.append(
                OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=None)
            )

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
            qty = int(trade["quantity"])
            if qty <= 1:
                qty = int(self.RISK_PER_TRADE / risk_per_share)
                qty = max(1, qty)
                db.update_trade_quantity(trade["id"], qty)

            target_1r = entry_price + risk_per_share
            qty_half = int(qty * 0.5)

            # TimeStop Check ab Fill Date
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            if not df_history.empty:
                df_since = df_history[df_history["date"] >= entry_date_ts]
                if len(df_since) >= 7:
                    return None

            exits = []
            exits.append(
                OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=qty)
            )

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
                entry=None,
                exits=exits,
            )

        return None

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
        except Exception as e:
            logger.error(f"Fehler beim Laden des Croc-Context für {symbol}: {e}")
        return None
