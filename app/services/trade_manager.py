import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
import yfinance as yf

from ..config import settings
from .database import SignalDatabase

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Domain Types & Constants
# --------------------------------------------------------------------------

OrderAction = Literal["BUY", "SELL"]
OrderType = Literal["LMT", "MKT", "LOC", "STP"]
TimeInForce = Literal["DAY", "GTC"]
TradeStatus = Literal["CREATED", "ACTIVE", "CLOSED", "MISSED"]

# Festes Risiko pro Trade laut Anforderung
RISK_PER_TRADE_CROC_USD = 100.0


@dataclass
class OrderLeg:
    action: OrderAction
    type: OrderType  # ANPASSUNG: order_type -> type
    price: float
    qty: int | None = (
        None  # ANPASSUNG: quantity -> qty (None bedeutet: Restliche/Volle Position)
    )
    tif: TimeInForce = "DAY"


@dataclass
class Order:
    id: str
    symbol: str
    qty: int  # ANPASSUNG: total_quantity -> qty
    mode: str
    entry: OrderLeg
    exits: list[OrderLeg]
    last_status: str = "PendingSubmit"
    last_update: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class CrocContext:
    """Hält Kontext-Daten (Low für StopLoss), die nicht in active_trades stehen."""

    high: float
    low: float


# --------------------------------------------------------------------------
# Service Class
# --------------------------------------------------------------------------


class TradeManager:
    """
    Verwaltet den Lebenszyklus von Trades:
    1. Status-Updates (Fills/Exits) via Market Data prüfen.
    2. YAML-Orders für neue Signale ('CREATED') generieren.
    """

    def __init__(self, db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        """
        Orchestriert den täglichen Prozess.
        """
        try:
            self._update_positions_status()
            self._export_orders_to_yaml(investment_per_trade)
        except Exception as error:
            logger.error(f"Kritischer Fehler im TradeManager Prozess: {error}")

    # ----------------------------------------------------------------------
    # Teil A: Status Updates (Marktdaten-Check)
    # ----------------------------------------------------------------------

    def _update_positions_status(self) -> None:
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            return

        logger.info(f"Prüfe Status für {len(trades)} Trades...")

        market_data_cache = self._fetch_market_data_batch(trades)
        alerts: list[str] = []
        today = datetime.now().date()

        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in market_data_cache:
                continue

            df = market_data_cache[symbol]
            alert = self._evaluate_trade_status(db, trade, df, today)
            if alert:
                alerts.append(alert)

        if alerts and self.telegram:
            self.telegram.send("⚡ **STATUS UPDATES**\n" + "\n".join(alerts))

    def _fetch_market_data_batch(self, trades: list[dict]) -> dict[str, pd.DataFrame]:
        symbols = list({t["symbol"] for t in trades})
        cache = {}

        for symbol in symbols:
            yahoo_symbol = self._map_to_yahoo_ticker(symbol)
            try:
                df = yf.download(
                    yahoo_symbol, period="10d", progress=False, auto_adjust=True
                )
                if df.empty:
                    continue

                # Normalisierung
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()

                date_col = "date" if "date" in df.columns else "datetime"
                if date_col in df.columns:
                    df["date"] = pd.to_datetime(df[date_col])
                    cache[symbol] = df

            except Exception:
                logger.warning(f"Fehler beim Laden von Marktdaten für {symbol}")

        return cache

    def _evaluate_trade_status(
        self, db: SignalDatabase, trade: dict, df: pd.DataFrame, today
    ) -> str | None:
        trade_id = trade["id"]
        status: TradeStatus = trade["status"]
        entry_price = float(trade["entry_price"])

        entry_date_raw = trade["entry_date"]
        entry_date_str = (
            entry_date_raw
            if isinstance(entry_date_raw, str)
            else entry_date_raw.strftime("%Y-%m-%d")
        )
        signal_date_obj = pd.to_datetime(entry_date_str).date()

        match status:
            case "CREATED":
                # Prüfe auf Fill am Tag NACH dem Signal
                potential_days = df[df["date"].dt.date > signal_date_obj]
                if potential_days.empty:
                    return None

                row = potential_days.iloc[0]
                check_date = row["date"].strftime("%Y-%m-%d")

                # Generischer Fill-Check: Wurde Preislimit intraday berührt?
                price_touched = row["low"] <= entry_price <= row["high"]

                if price_touched:
                    db.update_trade_status(trade_id, "ACTIVE")
                    return f"✅ **FILLED**: {trade['symbol']} am {check_date}"
                else:
                    db.update_trade_status(trade_id, "MISSED", "LIMIT_NOT_REACHED")
                    return f"❌ **MISSED**: {trade['symbol']} am {check_date}"

            case "ACTIVE":
                # Time Stop (7 Tage)
                days_since = (today - signal_date_obj).days
                if days_since >= 7:
                    db.close_trade(trade_id, reason="TIME_STOP")
                    return f"⏰ **TIME STOP**: {trade['symbol']} wird geschlossen."

        return None

    # ----------------------------------------------------------------------
    # Teil B: Order Generierung (Strategy Dispatcher)
    # ----------------------------------------------------------------------

    def _export_orders_to_yaml(self, investment_amount: float) -> None:
        db = SignalDatabase(self.db_path)

        # Nur neue Trades verarbeiten
        new_trades = [
            t for t in db.get_all_managed_trades() if t["status"] == "CREATED"
        ]

        if not new_trades:
            logger.info("Keine neuen Trades (Status CREATED) gefunden.")
            return

        logger.info(f"Erstelle Orders für {len(new_trades)} Signale...")
        orders_by_date: dict[str, list[dict]] = {}

        for trade in new_trades:
            try:
                order = self._create_order_strategy_dispatch(trade, investment_amount)

                if not order:
                    continue

                date_key = trade["entry_date"]
                if date_key not in orders_by_date:
                    orders_by_date[date_key] = []

                orders_by_date[date_key].append(self._dataclass_to_dict(order))

            except ValueError as error:
                logger.error(f"Order-Fehler {trade['symbol']}: {error}")

        self._write_yaml_files(orders_by_date)

    def _create_order_strategy_dispatch(
        self, trade: dict, budget: float
    ) -> Order | None:
        """Wählt die Order-Logik anhand des Strategienamens."""
        raw_name = str(trade.get("strategy", ""))
        strategy_clean = raw_name.lower().replace(" ", "")
        symbol = trade["symbol"]

        # 1. Moonbag (Croc)
        if "moonbag(tp5)" in strategy_clean:
            return self._build_moonbag_order(trade)

        # 2. DipBuyer (Legacy)
        if "dipbuyer" in strategy_clean:
            return self._build_dip_buyer_order(trade, budget)

        logger.debug(f"[{symbol}] Strategie '{raw_name}' wird ignoriert.")
        return None

    def _build_dip_buyer_order(self, trade: dict, budget: float) -> Order:
        """Legacy Logik für DipBuyer."""
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])

        quantity = max(1, int(budget / entry_price))
        target_price = entry_price + (0.8 * atr)

        return Order(
            id=f"{symbol}_DIP",
            symbol=symbol,
            qty=quantity,  # ANPASSUNG: total_quantity -> qty
            mode="BRACKET",
            entry=OrderLeg(
                action="BUY", type="LMT", price=round(entry_price, 2)
            ),  # ANPASSUNG: order_type -> type
            exits=[
                OrderLeg(
                    action="SELL", type="LOC", price=round(target_price, 2)
                )  # ANPASSUNG: order_type -> type
            ],
        )

    def _build_moonbag_order(self, trade: dict) -> Order | None:
        """
        Neue Logik für 'Moonbag (TP5)':
        - Entry: Stop Buy @ High (aus active_trades)
        - Stop: @ Low (nachgeladen aus screener_croc)
        - Risk: 100 USD (fix)
        - Exit: 50% bei 1R
        """
        symbol = trade["symbol"]
        entry_date = trade["entry_date"]

        # Entry Price ist bei Moonbag das High der Signalkerze (Stop Buy)
        entry_price = float(trade["entry_price"])

        # Kontext (Low) aus Screener-Tabelle nachladen
        context = self._fetch_croc_context(symbol, entry_date)
        if not context:
            logger.warning(
                f"[{symbol}] Order übersprungen: Fehlendes Low in screener_croc."
            )
            return None

        stop_loss = context.low
        risk_per_share = entry_price - stop_loss

        # Sicherheitscheck
        if risk_per_share <= 0:
            logger.warning(f"[{symbol}] Ungültiges Risiko (High <= Low).")
            return None

        # Positionsgröße: 100$ Risk / (High - Low)
        quantity = int(RISK_PER_TRADE_CROC_USD / risk_per_share)
        if quantity < 1:
            quantity = 1  # Fallback

        # Ziel: 1R (Entry + Risk)
        target_1r = entry_price + risk_per_share
        qty_exit_1 = int(quantity * 0.5)

        # --- Order Konstruktion ---

        # 1. Entry: Stop Buy
        entry_leg = OrderLeg(
            action="BUY",
            type="STP",
            price=round(entry_price, 2),  # ANPASSUNG: order_type -> type
        )

        exits = []

        # 2. Stop Loss (für gesamte verbleibende Menge)
        exits.append(
            OrderLeg(
                action="SELL",
                type="STP",  # ANPASSUNG: order_type -> type
                price=round(stop_loss, 2),
                qty=None,  # ANPASSUNG: quantity -> qty
            )
        )

        # 3. Take Profit (1R) für 50%
        if qty_exit_1 > 0:
            exits.append(
                OrderLeg(
                    action="SELL",
                    type="LMT",  # ANPASSUNG: order_type -> type
                    price=round(target_1r, 2),
                    qty=qty_exit_1,  # ANPASSUNG: quantity -> qty
                )
            )

        return Order(
            id=f"{symbol}_MNBG",
            symbol=symbol,
            qty=quantity,  # ANPASSUNG: total_quantity -> qty
            mode="BRACKET",
            entry=entry_leg,
            exits=exits,
        )

    def _fetch_croc_context(self, symbol: str, date_val) -> CrocContext | None:
        """Holt Low/High für Risk-Berechnung."""
        # Datum sicher in String YYYY-MM-DD wandeln
        date_str = str(date_val).split(" ")[0]

        sql = """
            SELECT high, low
            FROM screener_croc
            WHERE symbol = ? AND date = ?
            LIMIT 1
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(sql, (symbol, date_str)).fetchone()

                if row:
                    return CrocContext(high=float(row["high"]), low=float(row["low"]))
        except Exception as e:
            logger.error(f"DB Fehler Context ({symbol}): {e}")

        return None

    def _write_yaml_files(self, orders_map: dict[str, list[dict]]) -> None:
        for date_key, orders_list in orders_map.items():
            file_name = f"orders_{date_key}.yaml"
            file_path = self.orders_dir / file_name

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(orders_list, f, sort_keys=False, allow_unicode=True)

                logger.info(
                    f"Order-File erstellt: {file_name} ({len(orders_list)} Orders)"
                )
                if self.telegram:
                    self.telegram.send(f"📁 **New Orders**: {file_name} generated.")

            except OSError as error:
                logger.error(f"Konnte YAML Datei {file_name} nicht schreiben: {error}")

    def _map_to_yahoo_ticker(self, symbol: str) -> str:
        mapping = {
            "ES1!": "ES=F",
            "NQ1!": "NQ=F",
            "YM1!": "YM=F",
            "RTY1!": "RTY=F",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "JEN": "JEN.DE",
            "VH2": "VH2.DE",
        }
        return mapping.get(symbol, symbol)

    def _dataclass_to_dict(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            # Filterung: Diese Felder sollen NICHT in der YAML auftauchen
            excluded = ["last_status", "last_update"]
            return {
                k: self._dataclass_to_dict(v)
                for k, v in obj.__dict__.items()
                if v is not None and k not in excluded
            }
        if isinstance(obj, list):
            return [self._dataclass_to_dict(i) for i in obj]
        return obj
