import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import settings
from ...mapping import mapper
from ..database import SignalDatabase
from .strategies.abstract import BaseTradeStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.moonbag import MoonbagStrategy
from .strategies.split_target import SplitTargetStrategy

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Orchestrator für Trade-Management.
    Delegiert Logik an Strategien (Strategy Pattern).
    """

    def __init__(self, db_path: Path, stocks_db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

        # Datenbank-Schema prüfen
        self._ensure_db_schema()

        # Strategie-Register
        self.strategies: dict[str, BaseTradeStrategy] = {
            "dipbuyer": DipBuyerStrategy(),
            "moonbag": MoonbagStrategy(),
            "moonshot": MoonbagStrategy(),
            "crocsetup": MoonbagStrategy(),
            "split": SplitTargetStrategy(),
            "webhook": SplitTargetStrategy(),
            "tp1_tp3": SplitTargetStrategy(),
        }

    def _ensure_db_schema(self):
        try:
            db = SignalDatabase(self.db_path)
            with db._get_conn() as conn:
                cursor = conn.execute("PRAGMA table_info(active_trades)")
                columns = [row["name"] for row in cursor.fetchall()]

                if "signal_date" not in columns:
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN signal_date TEXT"
                    )
                    conn.commit()
                if "screener_id" not in columns:
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN screener_id INTEGER"
                    )
                    conn.commit()

                conn.execute(
                    "UPDATE active_trades SET signal_date = entry_date WHERE signal_date IS NULL OR signal_date = ''"
                )
                conn.commit()

                trades = conn.execute(
                    "SELECT id, symbol, signal_date, entry_date, strategy FROM active_trades WHERE screener_id IS NULL AND status IN ('ACTIVE', 'CLOSED')"
                ).fetchall()
                updates = []
                for t in trades:
                    ref_date = t["signal_date"] or t["entry_date"]
                    if not ref_date:
                        continue
                    ref_date_str = str(ref_date).split(" ")[0]
                    strategy = str(t["strategy"]).lower()
                    found_id = None

                    if any(s in strategy for s in ["moonbag", "split", "croc"]):
                        row = conn.execute(
                            "SELECT id FROM screener_croc WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                            (t["symbol"], ref_date_str),
                        ).fetchone()
                        if row:
                            found_id = row[0]
                    elif "dip" in strategy:
                        row = conn.execute(
                            "SELECT id FROM screener_dip_buyer WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                            (t["symbol"], ref_date_str),
                        ).fetchone()
                        if row:
                            found_id = row[0]

                    if found_id:
                        updates.append((found_id, t["id"]))

                if updates:
                    conn.executemany(
                        "UPDATE active_trades SET screener_id = ? WHERE id = ?", updates
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"DB Schema Check Failed: {e}")

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        try:
            self._cleanup_stale_trades()
            self._update_positions_status()
            self._run_daily_logging()
            self._export_orders_to_yaml(investment_per_trade)
        except Exception as error:
            logger.error(f"TradeManager Crash: {error}", exc_info=True)
            if self.telegram:
                self.telegram.send(
                    f"⚠️ **CRITICAL ERROR**: TradeManager ist abgestürzt!\n`{error}`"
                )

    def _cleanup_stale_trades(self):
        try:
            db = SignalDatabase(self.db_path)
            with db._get_conn() as conn:
                created_trades = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM active_trades WHERE status = 'CREATED'"
                    ).fetchall()
                ]
                active_trades = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM active_trades WHERE status = 'ACTIVE'"
                    ).fetchall()
                ]
            if not created_trades and not active_trades:
                return
            all_symbols = list(
                set([t["symbol"] for t in created_trades + active_trades])
            )
            market_dates = {}
            with sqlite3.connect(self.stocks_db_path) as conn:
                start_date = (datetime.now() - pd.Timedelta(days=60)).strftime(
                    "%Y-%m-%d"
                )
                placeholders = ",".join("?" for _ in all_symbols)
                rows = conn.execute(
                    f"SELECT symbol, date FROM market_prices WHERE symbol IN ({placeholders}) AND date >= ? AND timeframe='1D'",
                    all_symbols + [start_date],
                ).fetchall()
                for sym, date_str in rows:
                    if sym not in market_dates:
                        market_dates[sym] = []
                    market_dates[sym].append(pd.Timestamp(date_str))
            for sym in market_dates:
                market_dates[sym].sort()
            updates = []
            for trade in created_trades:
                sym = trade["symbol"]
                strategy_name = str(trade.get("strategy", "")).lower()
                max_wait = 1 if "dip" in strategy_name else 5
                ref_date = pd.Timestamp(
                    str(trade.get("signal_date") or trade["entry_date"]).split(" ")[0]
                )
                if (
                    sym in market_dates
                    and len([d for d in market_dates[sym] if d > ref_date]) > max_wait
                ):
                    reason = "EXPIRED" if "dip" in strategy_name else "STALE_CLEANUP"
                    updates.append(("MISSED", reason, trade["id"]))
            for trade in active_trades:
                sym = trade["symbol"]
                ref_date = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
                if (
                    sym in market_dates
                    and len([d for d in market_dates[sym] if d > ref_date]) > 20
                ):
                    updates.append(("CLOSED", "STALE_CLEANUP", trade["id"]))
            if updates:
                with db._get_conn() as conn:
                    conn.executemany(
                        "UPDATE active_trades SET status = ?, exit_reason = ?, closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        updates,
                    )
                    conn.commit()
        except Exception as e:
            logger.error(f"Fehler bei _cleanup_stale_trades: {e}")

    def _run_daily_logging(self):
        logger.info("Updating Strategy Logs/State...")
        db = SignalDatabase(self.db_path)
        sql = "SELECT * FROM active_trades WHERE status = 'ACTIVE' OR (status = 'CLOSED' AND date(closed_at) >= date('now', '-1 day'))"
        with db._get_conn() as conn:
            trades = [dict(r) for r in conn.execute(sql).fetchall()]
        if not trades:
            return
        market_data = self._fetch_market_data_batch(trades)
        for trade in trades:
            if market_data.get(trade["symbol"]) is None:
                continue
            df_hist = market_data[trade["symbol"]]
            if df_hist.empty:
                continue
            self._update_strategy_state(trade, df_hist.iloc[-1], df_hist, db)

    def _update_strategy_state(self, trade, current_candle, df_history, db):
        strategy_name = str(trade.get("strategy", "")).lower()
        strat_impl = self._get_strategy(trade)
        if not strat_impl or not hasattr(strat_impl, "get_current_params"):
            return
        params = strat_impl.get_current_params(trade, df_history, db)
        if not params:
            return
        entry_price = float(trade["entry_price"])
        current_price = float(current_candle["close"])

        if any(s in strategy_name for s in ["moonbag", "split", "croc"]):
            signal_name = ""
            if trade.get("screener_id"):
                with db._get_conn() as conn:
                    r = conn.execute(
                        "SELECT signal FROM screener_croc WHERE id=?",
                        (trade["screener_id"],),
                    ).fetchone()
                    if r:
                        signal_name = r[0]
            if not signal_name:
                try:
                    ref_date_str = str(
                        trade.get("signal_date") or trade["entry_date"]
                    ).split(" ")[0]
                    with db._get_conn() as conn:
                        row = conn.execute(
                            "SELECT id, signal FROM screener_croc WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                            (trade["symbol"], ref_date_str),
                        ).fetchone()
                        if row:
                            signal_name = row[1]
                            conn.execute(
                                "UPDATE active_trades SET screener_id = ? WHERE id = ?",
                                (row[0], trade["id"]),
                            )
                            conn.commit()
                except:
                    pass
            if not signal_name:
                last_snap = db.get_latest_croc_snapshot(trade["id"])
                signal_name = last_snap["signal"] if last_snap else ""
            pnl_pct = 0.0
            if entry_price > 0:
                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
            risk_multiple = 0.0
            risk = entry_price - params.stop_loss
            if risk > 0:
                risk_multiple = round((current_price - entry_price) / risk, 2)
            db.log_croc_trade(
                {
                    "date": current_candle["date"].strftime("%Y-%m-%d"),
                    "symbol": trade["symbol"],
                    "exchange": mapper.get_exchange(trade["symbol"]),
                    "timeframe": "1D",
                    "signal": signal_name,
                    "recommended_strategy": trade.get("strategy"),
                    "entry": entry_price,
                    "stop": params.stop_loss,
                    "tp_1": params.tp_1,
                    "tp_2": params.tp_2,
                    "exit_reason": trade.get("exit_reason"),
                    "close": current_price,
                    "high": float(current_candle["high"]),
                    "low": float(current_candle["low"]),
                    "active_trade_id": trade["id"],
                    "pnl_percent": pnl_pct,
                    "quantity": int(trade.get("quantity", 1)),
                    "risk_multiple": risk_multiple,
                }
            )
        elif "dip" in strategy_name:
            if not trade.get("screener_id"):
                try:
                    ref_date_str = str(
                        trade.get("signal_date") or trade["entry_date"]
                    ).split(" ")[0]
                    with db._get_conn() as conn:
                        row = conn.execute(
                            "SELECT id FROM screener_dip_buyer WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1",
                            (trade["symbol"], ref_date_str),
                        ).fetchone()
                        if row:
                            conn.execute(
                                "UPDATE active_trades SET screener_id = ? WHERE id = ?",
                                (row[0], trade["id"]),
                            )
                            conn.commit()
                except:
                    pass
            pnl_pct = 0.0
            if entry_price > 0:
                pnl_pct = round(((current_price - entry_price) / entry_price) * 100, 2)
            entry_ts = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])
            df_held = df_history[df_history["date"] >= entry_ts]
            days_held = len(df_held) - 1 if not df_held.empty else 0
            atr_val = float(trade.get("atr_at_entry", 0.0))
            db.log_dip_trade(
                {
                    "date": current_candle["date"].strftime("%Y-%m-%d"),
                    "symbol": trade["symbol"],
                    "exchange": mapper.get_exchange(trade["symbol"]),
                    "timeframe": "1D",
                    "entry": entry_price,
                    "atr": atr_val,
                    "tp_target": params.tp_1,
                    "threshold_loc": params.extras.get("threshold_loc", 0.0),
                    "exit_reason": trade.get("exit_reason"),
                    "close": current_price,
                    "high": float(current_candle["high"]),
                    "low": float(current_candle["low"]),
                    "active_trade_id": trade["id"],
                    "pnl_percent": pnl_pct,
                    "quantity": int(trade.get("quantity", 1)),
                    "days_held": days_held,
                }
            )

    def run_backfill(self, default_budget: float = 2000.0) -> dict:
        logger.info("Starte Smart-Backfill-Prozess...")
        stats = {
            "processed": 0,
            "filled": 0,
            "missed": 0,
            "closed_active": 0,
            "errors": 0,
            "skipped_no_data": 0,
        }
        db = SignalDatabase(self.db_path)
        with db._get_conn() as conn:
            trades = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM active_trades WHERE status IN ('CREATED', 'ACTIVE') ORDER BY signal_date ASC"
                ).fetchall()
            ]
        if not trades:
            return stats
        market_cache = {}
        for trade in trades:
            stats["processed"] += 1
            symbol = trade["symbol"]
            start_date_str = trade.get("signal_date") or trade["entry_date"]
            if not start_date_str:
                continue
            try:
                strat_impl = self._get_strategy(trade)
                if not strat_impl:
                    continue
                if symbol not in market_cache:
                    with sqlite3.connect(self.stocks_db_path) as conn:
                        df = pd.read_sql_query(
                            "SELECT date, open, high, low, close FROM market_prices WHERE symbol = ? AND timeframe='1D' ORDER BY date ASC",
                            conn,
                            params=(symbol,),
                        )
                        df["date"] = pd.to_datetime(df["date"])
                        market_cache[symbol] = df
                df_hist = market_cache[symbol]
                if df_hist.empty:
                    stats["skipped_no_data"] += 1
                    continue
                start_ts = pd.Timestamp(str(start_date_str).split(" ")[0])
                future_candles = df_hist[df_hist["date"] >= start_ts].sort_values(
                    "date"
                )
                if future_candles.empty:
                    continue

                for _, current_candle in future_candles.iterrows():
                    current_status = trade["status"]
                    simulated_history = df_hist[
                        df_hist["date"] <= current_candle["date"]
                    ]

                    if current_status == "CREATED":
                        result_msg = strat_impl.check_entry(
                            trade, current_candle, simulated_history, db
                        )
                        if result_msg and "FILLED" in result_msg:
                            stats["filled"] += 1
                            trade["status"] = "ACTIVE"
                            trade["entry_date"] = current_candle["date"].strftime(
                                "%Y-%m-%d"
                            )
                            self._calculate_and_update_quantity(
                                trade, strat_impl, default_budget, db, df_hist, start_ts
                            )
                            self._update_strategy_state(
                                trade, current_candle, simulated_history, db
                            )
                        elif result_msg and (
                            "INVALIDATED" in result_msg or "EXPIRED" in result_msg
                        ):
                            trade["status"] = "MISSED"
                            stats["missed"] += 1
                            break
                        elif not result_msg:
                            pass
                        else:
                            trade["status"] = "MISSED"
                            stats["missed"] += 1
                            break
                    elif current_status == "ACTIVE":
                        mgmt_msg = strat_impl.manage_active_trade(
                            trade, simulated_history, db
                        )
                        if mgmt_msg:
                            with db._get_conn() as c:
                                row = c.execute(
                                    "SELECT status, exit_reason FROM active_trades WHERE id=?",
                                    (trade["id"],),
                                ).fetchone()
                                if row:
                                    trade["status"] = row["status"]
                                    trade["exit_reason"] = row["exit_reason"]
                        self._update_strategy_state(
                            trade, current_candle, simulated_history, db
                        )
                        if mgmt_msg and "CLOSED" in str(trade.get("status", "")):
                            stats["closed_active"] += 1
                            break
            except Exception as e:
                logger.error(f"Fehler Backfill {symbol}: {e}", exc_info=True)
                stats["errors"] += 1
        logger.info(f"Backfill beendet: {stats}")
        return stats

    def _calculate_and_update_quantity(
        self, trade, strat, budget, db, df_hist, signal_ts
    ):
        try:
            entry_price = float(trade["entry_price"])
            new_qty = 1
            if isinstance(strat, DipBuyerStrategy):
                new_qty = int(budget / entry_price)
            elif isinstance(strat, SplitTargetStrategy):
                sl_price = entry_price * 0.99
                signal_candle = df_hist[df_hist["date"] == signal_ts]
                if not signal_candle.empty:
                    sl_price = float(signal_candle.iloc[0]["low"])
                risk = entry_price - sl_price
                if risk > 0:
                    new_qty = int(100.0 / risk)
                else:
                    new_qty = int(budget / entry_price)
            else:
                new_qty = int(budget / entry_price)
            new_qty = max(1, new_qty)
            db.update_trade_quantity(trade["id"], new_qty)
            trade["quantity"] = new_qty
        except Exception:
            pass

    # ... (Rest of helpers remain same) ...
    def _update_positions_status(self):
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()
        if not trades:
            return
        market_data = self._fetch_market_data_batch(trades)
        alerts = []
        for trade in trades:
            if market_data.get(trade["symbol"]) is None:
                continue
            df_hist = market_data[trade["symbol"]]
            if df_hist.empty:
                continue
            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue
            msg = None
            if trade["status"] == "CREATED":
                msg = strat_impl.check_entry(trade, df_hist.iloc[-1], df_hist, db)
            elif trade["status"] == "ACTIVE":
                msg = strat_impl.manage_active_trade(trade, df_hist, db)
            if msg:
                alerts.append(msg)
        if alerts and self.telegram:
            self.telegram.send("⚡ **Trade Updates**\n" + "\n".join(alerts))

    def _export_orders_to_yaml(self, budget: float):
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()
        if not trades:
            return
        market_data = self._fetch_market_data_batch(trades)
        orders_by_date = {}
        today_str = datetime.now().strftime("%Y-%m-%d")
        for trade in trades:
            if market_data.get(trade["symbol"]) is None:
                continue
            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue
            df_hist = market_data[trade["symbol"]]
            if df_hist.empty and trade["status"] == "ACTIVE":
                continue
            try:
                order = strat_impl.generate_orders(trade, df_hist, budget, db)
                if order:
                    if today_str not in orders_by_date:
                        orders_by_date[today_str] = []
                    orders_by_date[today_str].append(self._dataclass_to_dict(order))
            except Exception:
                pass
        self._write_yaml_files(orders_by_date)

    def _get_strategy(self, trade):
        raw = str(trade.get("strategy", "")).lower().replace(" ", "")
        for key, impl in self.strategies.items():
            if key in raw:
                return impl
        return None

    def _fetch_market_data_batch(self, trades):
        symbols = list({t["symbol"] for t in trades})
        if not symbols:
            return {}
        start_date = (datetime.now() - pd.Timedelta(days=50)).strftime("%Y-%m-%d")
        placeholders = ",".join("?" for _ in symbols)
        cache = {}
        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                df = pd.read_sql_query(
                    f"SELECT date, symbol, open, high, low, close FROM market_prices WHERE symbol IN ({placeholders}) AND date >= ? AND timeframe = '1D' ORDER BY date ASC",
                    conn,
                    params=symbols + [start_date],
                )
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    for sym, group in df.groupby("symbol"):
                        cache[sym] = group.reset_index(drop=True)
        except Exception:
            pass
        return cache

    def _write_yaml_files(self, orders_map):
        for date_key, orders in orders_map.items():
            path = self.orders_dir / f"orders_{date_key}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(orders, f, sort_keys=False)
            if self.telegram:
                self.telegram.send(
                    f"📁 **Orders Generated**: {len(orders)} Orders für {date_key}"
                )

    def _dataclass_to_dict(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            excluded = ["last_status", "last_update"]
            return {
                k: self._dataclass_to_dict(v)
                for k, v in obj.__dict__.items()
                if v is not None and k not in excluded
            }
        if isinstance(obj, list):
            return [self._dataclass_to_dict(i) for i in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return round(float(obj), 2)
        return obj
