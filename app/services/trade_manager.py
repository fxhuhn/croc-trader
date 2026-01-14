import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import yaml
import yfinance as yf

from ..config import settings
from .database import SignalDatabase

logger = logging.getLogger(__name__)


@dataclass
class OrderLeg:
    action: Literal["BUY", "SELL"]
    type: Literal["LMT", "MKT", "LOC", "STP"]
    price: float
    tif: str = "DAY"


@dataclass
class Order:
    id: str
    symbol: str
    qty: int
    mode: str
    entry: OrderLeg
    exits: List[OrderLeg]
    ib_id: Optional[int] = None
    last_status: str = "PendingSubmit"
    last_update: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class TradeManager:
    """
    Verwaltet den Lebenszyklus von Trades:
    1. Status-Updates bestehender Positionen (via Market Data).
    2. Generierung von Order-Files für neue Signale.
    """

    def __init__(self, db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        """
        Führt den täglichen Prozess aus: Status prüfen -> Neue Orders schreiben.
        """
        # 1. Datenbank-Status aktualisieren (Fills, Exits, Time-Stops erkennen)
        self.update_positions_status()

        # 2. Orders für NEUE Signale generieren (DipBuyer)
        self.export_orders_to_yaml(investment_per_trade)

    def update_positions_status(self) -> None:
        """
        Prüft 'CREATED' Trades auf Fills und 'ACTIVE' Trades auf Exits/Time-Stops.
        Aktualisiert den Status in der Datenbank und sendet Alerts.
        """
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            logger.info("TradeManager: Keine aktiven Trades zur Prüfung.")
            return

        logger.info(f"Prüfe Status für {len(trades)} Trades...")

        # --- Daten laden (Batch Processing) ---
        symbols = list({t["symbol"] for t in trades})
        data_cache = {}

        for symbol in symbols:
            yahoo_symbol = self._get_yahoo_ticker(symbol)
            try:
                # Daten laden (letzte 10 Tage reichen für aktuelle Prüfungen)
                df = yf.download(
                    yahoo_symbol, period="10d", progress=False, auto_adjust=True
                )

                if not df.empty:
                    df = df.reset_index()
                    # Spalten bereinigen (MultiIndex flatten falls nötig)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()

                    # Datumsspalte normalisieren
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    elif "datetime" in df.columns:
                        df["date"] = pd.to_datetime(df["datetime"])

                    data_cache[symbol] = df
                else:
                    logger.warning(
                        f"Keine Daten für {symbol} (Yahoo: {yahoo_symbol}) gefunden."
                    )

            except Exception as e:
                logger.error(f"Fehler beim Laden von {symbol} -> {yahoo_symbol}: {e}")

        # --- Status Logik ---
        alerts = []
        today = datetime.now().date()

        for trade in trades:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            status = trade["status"]
            entry_price = trade["entry_price"]
            entry_date_str = trade["entry_date"]

            if symbol not in data_cache:
                continue

            df = data_cache[symbol]
            # Sicherstellen, dass entry_date als String vorliegt (Fallback)
            if not isinstance(entry_date_str, str):
                # Falls es wider Erwarten ein Timestamp ist
                entry_date_str = entry_date_str.strftime("%Y-%m-%d")

            signal_date_obj = pd.to_datetime(entry_date_str).date()

            # Fall A: Trade ist noch im Wartestand (CREATED) -> Prüfen ob gefillt
            if status == "CREATED":
                # Wir suchen Tage NACH dem Signal
                potential_days = df[df["date"].dt.date > signal_date_obj]

                if not potential_days.empty:
                    # Checke den ersten Tag nach Signal
                    row = potential_days.iloc[0]
                    check_date = row["date"].strftime("%Y-%m-%d")

                    # Limit Buy Logik: Wenn Low <= Limit, dann Fill
                    if row["low"] <= entry_price:
                        db.update_trade_status(trade_id, "ACTIVE")
                        alerts.append(f"✅ **FILLED**: {symbol} am {check_date}")
                    else:
                        # Setup verfallen (Gap Up o.ä.), wenn Bedingungen nicht erfüllt
                        # Hier vereinfacht: Wenn am Tag 1 nicht geholt, dann MISSED
                        db.update_trade_status(trade_id, "MISSED", "LIMIT_NOT_REACHED")
                        alerts.append(f"❌ **MISSED**: {symbol} am {check_date}")

            # Fall B: Trade läuft (ACTIVE) -> Prüfen auf Time Stop oder Exit-Signale
            elif status == "ACTIVE":
                last_row = df.iloc[-1]
                current_close = last_row["close"]
                days_since = (today - signal_date_obj).days

                # 1. TIME STOP (nach 7 Tagen)
                if days_since >= 7:
                    db.close_trade(trade_id, reason="TIME_STOP")
                    alerts.append(
                        f"⏰ **TIME STOP**: {symbol} wird geschlossen (DB Update)."
                    )
                    # Hinweis: Exit-Order wird hier laut Anforderung ignoriert (nur DB Pflege)

                # 2. LOC EXIT Check (nur Alerting, da Positionen ignoriert werden sollen)
                else:
                    prev_high = 999999
                    if len(df) >= 2:
                        prev_high = df.iloc[-2]["high"]

                    if current_close > prev_high:
                        alerts.append(f"📈 **LOC SIGNAL**: {symbol} (Close > PrevHigh)")

        # --- Telegram Benachrichtigung ---
        if alerts and self.telegram:
            msg = "⚡ **STATUS UPDATES**\n" + "\n".join(alerts)
            self.telegram.send(msg)

    def export_orders_to_yaml(self, investment_amount: float) -> None:
        """
        Erstellt YAML-Order-Files NUR für neue 'DipBuyer' Trades.
        Bestehende Positionen werden laut Anforderung ignoriert.
        """
        db = SignalDatabase(self.db_path)

        # Wir holen alle Trades, filtern aber in Python explizit
        all_trades = db.get_all_managed_trades()

        # Filter: Nur 'CREATED' (neu) und Strategie 'DipBuyer'
        new_dip_signals = [
            t
            for t in all_trades
            if t["status"] == "CREATED" and self._is_dip_buyer(t.get("strategy"))
        ]

        if not new_dip_signals:
            logger.info("Keine neuen DipBuyer-Signale für Order-Generierung.")
            return

        orders_by_date = {}

        for trade in new_dip_signals:
            try:
                order = self._create_dip_buyer_order(trade, investment_amount)

                # Gruppierung nach Datum (Entry Date)
                entry_date = trade["entry_date"]
                if entry_date not in orders_by_date:
                    orders_by_date[entry_date] = []

                orders_by_date[entry_date].append(asdict(order))

            except ValueError as e:
                logger.error(f"Fehler bei Order-Erstellung für {trade['symbol']}: {e}")
                continue

        self._write_yaml_files(orders_by_date)

    def _create_dip_buyer_order(self, trade: dict, budget: float) -> Order:
        """
        Erstellt das Order-Datenobjekt basierend auf der DipBuyer-Logik.
        Regel: Entry = Limit, Exit = LOC (Target), Qty = Budget / Preis.
        """
        symbol = trade["symbol"]
        entry_price = float(trade["entry_price"])
        atr = float(trade["atr_at_entry"])

        if entry_price <= 0:
            raise ValueError(f"Ungültiger Entry-Preis: {entry_price}")

        # Berechnung Quantity (abgerundet)
        qty = int(budget / entry_price)
        if qty < 1:
            logger.warning(
                f"{symbol}: Budget ({budget}) zu klein für Preis ({entry_price}). Setze Qty=1."
            )
            qty = 1

        # Logik: Entry LMT, Exit LOC (Target)
        target_price = entry_price + (0.8 * atr)

        # Unique ID generieren
        order_id = f"{symbol}_DIP_Buyer"

        # Order Legs definieren
        entry_leg = OrderLeg(
            action="BUY", type="LMT", price=round(entry_price, 2), tif="DAY"
        )

        exit_leg = OrderLeg(
            action="SELL",
            type="LOC",  # Limit On Close
            price=round(target_price, 2),
            tif="DAY",
        )

        return Order(
            id=order_id,
            symbol=symbol,
            qty=qty,
            mode="BRACKET",
            entry=entry_leg,
            exits=[exit_leg],
            ib_id=None,
            last_status="PendingSubmit",
        )

    def _write_yaml_files(self, orders_map: dict) -> None:
        """Schreibt die gesammelten Orders in YAML-Dateien."""
        for date_key, orders_list in orders_map.items():
            file_name = f"orders_{date_key}.yaml"
            file_path = self.orders_dir / file_name

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # sort_keys=False behält die Reihenfolge der Felder bei (wichtig für Lesbarkeit)
                    yaml.dump(orders_list, f, sort_keys=False, allow_unicode=True)

                logger.info(
                    f"Order-File erstellt: {file_name} ({len(orders_list)} Orders)"
                )

                if self.telegram:
                    self.telegram.send(f"📁 **New Orders**: {file_name} generated.")

            except OSError as e:
                logger.error(f"Konnte YAML Datei nicht schreiben: {e}")

    def _is_dip_buyer(self, strategy_name: Optional[str]) -> bool:
        """
        Hilfsmethode zur Identifikation der Strategie.
        Robust gegen Groß-/Kleinschreibung oder Varianten.
        """
        if not strategy_name:
            return False
        return "dipbuyer" in strategy_name.lower().replace("_", "").replace(" ", "")

    def _get_yahoo_ticker(self, symbol: str) -> str:
        """
        Übersetzt TradingView/Broker-Symbole in Yahoo Finance Symbole.
        """
        mapping = {
            # --- FUTURES ---
            "ES1!": "ES=F",
            "NQ1!": "NQ=F",
            "YM1!": "YM=F",
            "RTY1!": "RTY=F",
            "FDAX1!": "DX=F",
            "GC1!": "GC=F",
            "SI1!": "SI=F",
            "CL1!": "CL=F",
            "BTC1!": "BTC=F",
            # --- FOREX ---
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            # --- SPEZIAL ---
            "JEN": "JEN.DE",
            "VH2": "VH2.DE",
        }

        if symbol in mapping:
            return mapping[symbol]

        return symbol
