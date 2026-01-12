import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DipBuyerBacktester:
    def __init__(self, stocks_db_path: Path, strategies_db_path: Path):
        self.stocks_db_path = stocks_db_path
        self.strategies_db_path = strategies_db_path
        self._init_strategy_db()

    def _init_strategy_db(self):
        try:
            with sqlite3.connect(self.strategies_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT,
                        symbol TEXT,
                        signal_date TEXT,
                        entry_date TEXT,
                        exit_date TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        pct_change REAL,
                        exit_reason TEXT,
                        status TEXT,
                        holding_days INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                try:
                    conn.execute("ALTER TABLE backtest_logs ADD COLUMN status TEXT")
                except sqlite3.OperationalError:
                    pass
        except Exception as e:
            logger.error(f"Backtest DB Init Error: {e}")

    def run_backtest(self, start_year=2023, debug_symbol=None):
        start_date = f"{start_year}-01-01"
        logger.info(
            f"Starte DipBuyer Backtest ab {start_date} (TP + LOC + TimeStop)..."
        )

        # 400 Tage Vorlauf für Indikator-Konsistenz
        fetch_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                df_all = pd.read_sql_query(
                    f"SELECT date, symbol, open, high, low, close, volume FROM market_prices WHERE date >= '{fetch_start}' AND timeframe = '1D' ORDER BY date ASC",
                    conn,
                )
        except Exception as e:
            logger.error(f"DB Error: {e}")
            return None

        if df_all.empty:
            return None
        df_all["date"] = pd.to_datetime(df_all["date"])

        analysis_mask = df_all["date"] >= start_date
        if not analysis_mask.any():
            return {"error": "Keine Daten im Zeitraum."}

        data_universe = {
            "first_record": df_all.loc[analysis_mask, "date"]
            .min()
            .strftime("%Y-%m-%d"),
            "last_record": df_all.loc[analysis_mask, "date"].max().strftime("%Y-%m-%d"),
            "total_symbols": df_all["symbol"].nunique(),
        }

        symbols = df_all["symbol"].unique()
        trades = []

        logger.info(f"Verarbeite {len(symbols)} Symbole...")

        for symbol in symbols:
            df_sym = (
                df_all[df_all["symbol"] == symbol].copy().set_index("date").sort_index()
            )
            if len(df_sym) < 250:
                continue

            is_debug = symbol == debug_symbol
            sym_trades = self._process_symbol(
                symbol, df_sym, start_date, debug=is_debug
            )
            trades.extend(sym_trades)

        self._save_trades_to_db(trades)
        results = self._generate_report(trades, data_universe)

        filled = sum(1 for t in trades if t["status"] == "FILLED")
        logger.info(f"Fertig: {filled} Trades ausgeführt.")

        return results

    def _save_trades_to_db(self, trades):
        if not trades:
            return
        data_to_insert = []
        for t in trades:
            s_date = t["signal_date"].strftime("%Y-%m-%d") if t["signal_date"] else None
            e_date = t["entry_date"].strftime("%Y-%m-%d") if t["entry_date"] else None
            ex_date = t["exit_date"].strftime("%Y-%m-%d") if t["exit_date"] else None

            data_to_insert.append(
                (
                    "DipBuyer",
                    t["symbol"],
                    s_date,
                    e_date,
                    ex_date,
                    t["entry_price"],
                    t["exit_price"],
                    t["pct_change"],
                    t["exit_reason"],
                    t["status"],
                    t["holding_days"],
                )
            )
        try:
            with sqlite3.connect(self.strategies_db_path) as conn:
                conn.execute(
                    "DELETE FROM backtest_logs WHERE strategy_name = 'DipBuyer'"
                )
                conn.executemany(
                    """
                    INSERT INTO backtest_logs (strategy_name, symbol, signal_date, entry_date, exit_date, entry_price, exit_price, pct_change, exit_reason, status, holding_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    data_to_insert,
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB Save Error: {e}")

    def _process_symbol(self, symbol, df, valid_from_date, debug=False):
        # Indikatoren
        df["sma200"] = df["close"].rolling(window=200).mean()
        df["vol_sma20"] = df["volume"].rolling(window=20).mean()

        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr5"] = tr.ewm(span=9, adjust=False).mean()

        df["atr_r3"] = (df["close"] - df["close"].shift(3)) / df["atr5"]
        df["ibs"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["prev_close"] = df["close"].shift(1)
        df["prev_open"] = df["open"].shift(1)
        df["limit_price"] = df["close"] - df["atr5"]

        # Filter
        c_date = df.index >= valid_from_date
        c_vol = df["vol_sma20"] > 500_000
        c_price = df["close"] > 5.0
        c_trend = df["close"] > df["sma200"]
        c_dip = df["atr_r3"] < -1.0
        c_vola = (df["atr5"] / df["close"]) > 0.03
        c_red = df["close"] < df["open"]
        c_ibs = df["ibs"] < 0.2
        c_prev_red = df["prev_close"] < df["prev_open"]

        mask = (
            c_date
            & c_vol
            & c_price
            & c_trend
            & c_dip
            & c_vola
            & c_red
            & c_ibs
            & c_prev_red
        )

        if debug:
            debug_mask = (df.index >= "2025-12-28") & (df.index <= "2026-01-15")
            for date, row in df[debug_mask].iterrows():
                if mask.loc[date]:
                    logger.info(f"DEBUG {symbol} {date.date()}: ✅ SIGNAL")

        signal_dates = df[mask].index
        trades = []

        for signal_date in signal_dates:
            try:
                sig_idx = df.index.get_loc(signal_date)
                if sig_idx + 1 >= len(df):
                    continue

                # Setup Werte vom Signaltag
                limit_price = df.iloc[sig_idx]["limit_price"]
                atr_at_signal = df.iloc[sig_idx]["atr5"]

                # --- ENTRY PRÜFUNG (Tag 1) ---
                entry_day = df.iloc[sig_idx + 1]

                if entry_day["low"] <= limit_price:
                    # FILLED
                    entry_price = min(limit_price, entry_day["open"])
                    entry_date = entry_day.name
                    status = "FILLED"

                    # --- TP BERECHNUNG ---
                    # Entry + 0.8 * ATR (vom Signaltag/Setup)
                    target_price = entry_price + (0.8 * atr_at_signal)

                    exit_price = None
                    exit_date = None
                    exit_reason = "TIME_STOP"

                    # Exit Loop: Tag 1 bis Tag 7 (Handelstage)
                    # Wir iterieren durch den DataFrame, das SIND Handelstage (keine Wochenenden)
                    max_lookforward = min(8, len(df) - sig_idx)

                    for i in range(1, max_lookforward):
                        current_day = df.iloc[sig_idx + i]

                        # 1. CHECK: TAKE PROFIT (Intraday High)
                        # Passiert meist vor dem Close, daher Priorität 1
                        if current_day["high"] >= target_price:
                            exit_price = target_price
                            exit_date = current_day.name
                            exit_reason = "TAKE_PROFIT"
                            break

                        # 2. CHECK: LOC (Close > PrevHigh)
                        # PrevHigh ist das High vom Vortag (relativ zum Loop)
                        prev_day_high = df.iloc[sig_idx + i - 1]["high"]
                        if current_day["close"] > prev_day_high:
                            exit_price = current_day["close"]
                            exit_date = current_day.name
                            exit_reason = "LOC_PROFIT"
                            break

                    # TIME STOP (Close am 7. Handelstag)
                    if exit_price is None:
                        last_idx_offset = max_lookforward - 1
                        last_day_row = df.iloc[sig_idx + last_idx_offset]
                        exit_price = last_day_row["close"]
                        exit_date = last_day_row.name
                        if max_lookforward < 8:
                            exit_reason = "FORCE_CLOSE (End of Data)"

                    pct_change = (exit_price - entry_price) / entry_price * 100
                    holding_days = (
                        exit_date - entry_date
                    ).days  # Kalendertage für Statistik

                    trades.append(
                        {
                            "symbol": symbol,
                            "signal_date": signal_date,
                            "entry_date": entry_date,
                            "exit_date": exit_date,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pct_change": pct_change,
                            "year": entry_date.year,
                            "month": entry_date.month,
                            "exit_reason": exit_reason,
                            "status": status,
                            "holding_days": holding_days,
                        }
                    )
                else:
                    # MISSED
                    trades.append(
                        {
                            "symbol": symbol,
                            "signal_date": signal_date,
                            "entry_date": entry_day.name,
                            "exit_date": None,
                            "entry_price": limit_price,
                            "exit_price": None,
                            "pct_change": 0.0,
                            "year": entry_day.name.year,
                            "month": entry_day.name.month,
                            "exit_reason": "MISSED",
                            "status": "MISSED",
                            "holding_days": 0,
                        }
                    )

            except Exception:
                continue

        return trades

    def _generate_report(self, trades_list, data_universe):
        if not trades_list:
            return {
                "error": "Keine Signale gefunden.",
                "data_universe": data_universe,
                "metrics": {},
            }

        df = pd.DataFrame(trades_list)
        df_filled = df[df["status"] == "FILLED"].copy()

        total_signals = len(df)
        total_trades = len(df_filled)
        recent_trades = []

        if total_trades > 0:
            win_trades = df_filled[df_filled["pct_change"] > 0]
            loss_trades = df_filled[df_filled["pct_change"] <= 0]

            win_rate = len(win_trades) / total_trades * 100
            avg_return = df_filled["pct_change"].mean()

            gross_win = win_trades["pct_change"].sum()
            gross_loss = abs(loss_trades["pct_change"].sum())
            profit_factor = (
                round(gross_win / gross_loss, 2) if gross_loss != 0 else 99.99
            )

            df_sorted = df_filled.sort_values("entry_date")
            df_sorted["cum_return"] = df_sorted["pct_change"].cumsum()
            df_sorted["running_max"] = df_sorted["cum_return"].cummax()
            df_sorted["drawdown"] = df_sorted["cum_return"] - df_sorted["running_max"]
            max_drawdown = df_sorted["drawdown"].min()

            avg_holding = df_filled["holding_days"].mean()
            best_trade = df_filled["pct_change"].max()
            worst_trade = df_filled["pct_change"].min()
            exit_reasons = df_filled["exit_reason"].value_counts().to_dict()

            # --- LETZTE 20 TRADES ---
            # Wir formatieren sie direkt hier für die Anzeige
            last_20 = df_filled.sort_values("entry_date", ascending=False).head(20)
            for _, row in last_20.iterrows():
                recent_trades.append(
                    {
                        "date": row["entry_date"].strftime("%Y-%m-%d"),
                        "symbol": row["symbol"],
                        "entry": f"{row['entry_price']:.2f}",
                        "exit": f"{row['exit_price']:.2f}",
                        "pct": f"{row['pct_change']:.2f}%",
                        "reason": row["exit_reason"],
                        "class": "pos-text" if row["pct_change"] > 0 else "neg-text",
                    }
                )
        else:
            win_rate = 0
            avg_return = 0
            profit_factor = 0
            max_drawdown = 0
            avg_holding = 0
            best_trade = 0
            worst_trade = 0
            exit_reasons = {}

        fill_rate = (total_trades / total_signals * 100) if total_signals > 0 else 0

        monthly_perf = (
            df_filled.groupby(["year", "month"])["pct_change"]
            .mean()
            .unstack(level=1)
            .fillna(0)
        )
        monthly_count = (
            df_filled.groupby(["year", "month"])["pct_change"]
            .count()
            .unstack(level=1)
            .fillna(0)
        )

        current_year = datetime.now().year
        current_month = datetime.now().month
        month_avgs = df_filled.groupby("month")["pct_change"].mean()
        avg_for_this_month = month_avgs.get(current_month, 0)
        curr_perf = (
            monthly_perf.loc[current_year, current_month]
            if (
                current_year in monthly_perf.index
                and current_month in monthly_perf.columns
            )
            else 0
        )

        report = {
            "data_universe": data_universe,
            "metrics": {
                "total_trades": total_trades,
                "total_signals": total_signals,
                "fill_rate": round(fill_rate, 2),
                "win_rate": round(win_rate, 2),
                "avg_return_pct": round(avg_return, 2),
                "profit_factor": profit_factor,
                "max_drawdown": round(max_drawdown, 2),
                "avg_holding_days": round(avg_holding, 1),
                "best_trade": round(best_trade, 2),
                "worst_trade": round(worst_trade, 2),
                "exit_reasons": exit_reasons,
            },
            "comparison": {
                "current_month_name": datetime.now().strftime("%B"),
                "current_perf": round(curr_perf, 2),
                "historical_avg": round(avg_for_this_month, 2),
                "status": "BETTER" if curr_perf > avg_for_this_month else "WORSE",
            },
            "monthly_matrix": monthly_perf.round(2).to_dict(orient="index"),
            "monthly_counts": monthly_count.astype(int).to_dict(orient="index"),
            "years": sorted(df_filled["year"].unique().tolist(), reverse=True)
            if not df_filled.empty
            else [],
            "recent_trades": recent_trades,
        }
        return report
