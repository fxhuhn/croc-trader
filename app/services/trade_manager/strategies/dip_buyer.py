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
        Basis: Signal Date. Entry muss strikt NACH Signal Date erfolgen (T+1).
        """
        # 1. Datum ermitteln: Wenn signal_date vorhanden (neu), nutze das.
        # Fallback auf entry_date (alt), aber das ist unsicher.
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])

        # CHECK: Kerze muss NACH dem Signal-Tag liegen
        if candle["date"] <= signal_date_ts:
            return None

        entry_price = float(trade["entry_price"])

        # Klassischer Limit Check (Low <= Limit <= High)
        if candle["low"] <= entry_price <= candle["high"]:
            # DB UPDATE: Status ACTIVE
            # WICHTIG: entry_date wird jetzt auf den ECHTEN FILL-Tag aktualisiert!
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

        return None

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        """
        Management Logik.
        df_history enthält alle Daten. Wir schneiden ab entry_date (das jetzt das Fill-Date ist).
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
        trading_days = len(df_since) - 1
        current_candle = df_since.iloc[-1]

        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])
        tp_price = entry_price + (0.8 * atr)

        # Safety Check
        if len(df_history) < 2:
            return None

        # Previous Day High vom Gesamt-DF (für LOC Exit wichtig)
        prev_day_candle = df_history.iloc[-2]
        prev_day_high = float(prev_day_candle["high"])

        exit_reason = None
        exit_price = None

        # --- REGELWERK ---

        # 1. Take Profit (ab Tag 0 möglich)
        if current_candle["high"] >= tp_price:
            exit_reason = "TAKE_PROFIT"
            exit_price = tp_price

        # 2. LOC Check (Ab Tag 1 aktiv)
        if not exit_reason and trading_days >= 1:
            if current_candle["close"] > prev_day_high:
                exit_reason = "LOC_PROFIT"
                exit_price = current_candle["close"]

        # 3. Time Stop (Tag 7 Close)
        if not exit_reason and trading_days >= 7:
            exit_reason = "TIME_STOP"
            exit_price = current_candle["close"]

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

        # --- A) Entry Order (CREATED) ---
        if status == "CREATED":
            entry_price = float(trade["entry_price"])
            # Fallback für Bracket-Exit (Signal-Tag High)
            prev_day_high = (
                float(df_history.iloc[-1]["high"])
                if not df_history.empty
                else (entry_price * 1.05)
            )

            qty = max(1, int(budget / entry_price))
            db.update_trade_quantity(trade["id"], qty)

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
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            df_since = df_history[df_history["date"] >= entry_date_ts]
            trading_days = len(df_since) - 1

            next_trading_day_idx = trading_days + 1
            if next_trading_day_idx > 7:
                return None

            entry_price = float(trade["entry_price"])
            atr = float(trade["atr_at_entry"])
            qty = int(trade["quantity"])

            if qty <= 1:
                qty = max(1, int(budget / entry_price))
                db.update_trade_quantity(trade["id"], qty)

            tp_price = round(entry_price + (0.8 * atr), 2)
            prev_day_high = round(float(df_history.iloc[-1]["high"]), 2)

            exits = []
            exits.append(OrderLeg(action="SELL", type="LMT", price=tp_price, qty=qty))

            if next_trading_day_idx >= 1:
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
