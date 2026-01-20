import logging
import sqlite3
from typing import Optional, override

import pandas as pd

from ...database import SignalDatabase
from ..types import CrocContext, Order, OrderLeg, TradeParams
from .abstract import BaseTradeStrategy

logger = logging.getLogger(__name__)


class MoonbagStrategy(BaseTradeStrategy):
    RISK_PER_TRADE = 100.0

    @override
    def get_current_params(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[TradeParams]:
        symbol = trade["symbol"]
        signal_date = trade.get("signal_date") or trade["entry_date"]
        context = self._fetch_croc_context(symbol, signal_date, db)
        if not context:
            return None
        entry_price = float(trade["entry_price"])
        stop_loss = context.low
        risk = entry_price - stop_loss
        tp1 = entry_price + risk if risk > 0 else None
        return TradeParams(stop_loss=stop_loss, tp_1=tp1, tp_2=None)

    @override
    def check_entry(
        self,
        trade: dict,
        candle: pd.Series,
        df_history: pd.DataFrame,
        db: SignalDatabase,
    ) -> Optional[str]:
        entry_price = float(trade["entry_price"])
        signal_date_str = trade.get("signal_date") or trade["entry_date"]
        signal_date_ts = pd.Timestamp(str(signal_date_str).split(" ")[0])

        if candle["date"] <= signal_date_ts:
            return None

        # --- INVALIDATION CHECK ---
        context = self._fetch_croc_context(trade["symbol"], signal_date_str, db)
        if context:
            if candle["low"] <= context.low:
                try:
                    with db._get_conn() as conn:
                        conn.execute(
                            "UPDATE active_trades SET status = 'MISSED', exit_reason = 'SETUP_INVALIDATED' WHERE id = ?",
                            (trade["id"],),
                        )
                        conn.commit()
                except Exception:
                    pass
                return f"❌ **INVALIDATED**: {trade['symbol']} Low broke Setup Low."

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
                logger.error(f"Moonbag Fill Error: {e}")
            return f"✅ **FILLED (Stop Buy)**: {trade['symbol']} @ {entry_price} am {fill_date_str}"

        return None

    @override
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        # ... (Remains same) ...
        entry_date = trade["entry_date"]
        entry_date_ts = pd.Timestamp(str(entry_date).split(" ")[0])
        df_since = df_history[df_history["date"] >= entry_date_ts].reset_index(
            drop=True
        )
        if df_since.empty:
            return None
        current_candle = df_since.iloc[-1]
        exit_date_str = current_candle["date"].strftime("%Y-%m-%d")

        signal_date = trade.get("signal_date") or entry_date
        context = self._fetch_croc_context(trade["symbol"], signal_date, db)

        if context:
            stop_loss_price = context.low
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
                return (
                    f"🛑 **STOP LOSS**: {trade['symbol']} @ {stop_loss_price} ({pct}%)"
                )

        if len(df_since) >= 7:
            exit_price = current_candle["close"]
            self._close_trade_in_db(
                db, trade["id"], "TIME_STOP", exit_price, exit_date=exit_date_str
            )
            return f"⏰ **TIME STOP**: {trade['symbol']}"
        return None

    @override
    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        # ... (Remains same) ...
        status = trade["status"]
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        signal_date = trade.get("signal_date") or trade["entry_date"]
        context = self._fetch_croc_context(symbol, signal_date, db)
        if not context:
            return None
        stop_loss = context.low
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return None

        if status == "CREATED":
            qty = max(1, int(self.RISK_PER_TRADE / risk_per_share))
            target_1r = entry_price + risk_per_share
            db.update_trade_quantity(trade["id"], qty)
            exits = [
                OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=None)
            ]
            if qty * 0.5 >= 1:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LMT",
                        price=round(target_1r, 2),
                        qty=int(qty * 0.5),
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
        elif status == "ACTIVE":
            qty = int(trade["quantity"])
            entry_date_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            if not df_history.empty:
                if len(df_history[df_history["date"] >= entry_date_ts]) >= 7:
                    return None
            exits = [
                OrderLeg(action="SELL", type="STP", price=round(stop_loss, 2), qty=qty)
            ]
            if qty * 0.5 >= 1:
                exits.append(
                    OrderLeg(
                        action="SELL",
                        type="LMT",
                        price=round(entry_price + risk_per_share, 2),
                        qty=int(qty * 0.5),
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
        try:
            with db._get_conn() as conn:
                row = conn.execute(
                    "SELECT high, low FROM screener_croc WHERE symbol = ? AND date = ? LIMIT 1",
                    (symbol, date_str),
                ).fetchone()
                if row:
                    return CrocContext(high=float(row["high"]), low=float(row["low"]))
        except Exception:
            pass
        return None
