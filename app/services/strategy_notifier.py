import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from .database import SignalDatabase
from .telegram import TelegramBot

logger = logging.getLogger(__name__)


class StrategyNotifier:
    def __init__(
        self,
        db_path: Path,
        telegram_bot: TelegramBot,
        config_path: Path = Path("data/croc-strategie.yaml"),
    ):
        self.db_path = db_path
        self.telegram = telegram_bot
        self.config_path = config_path

    def _load_strategies(self):
        """Lädt die Strategien aus der YAML Datei."""
        if not self.config_path.exists():
            logger.error(f"Strategie-Datei nicht gefunden: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Strategien: {e}")
            return []

    def check_and_notify(self, lookback_days=1, title_prefix="Tagesabschluss"):
        """
        1. Holt Screener Ergebnisse der letzten X Tage.
        2. Wendet die YAML-Filter an.
        3. Sendet Ergebnisse per Telegram.
        """
        db = SignalDatabase(self.db_path)
        strategies = self._load_strategies()

        if not strategies:
            return

        # 1. Daten holen (Wir nehmen an, es gibt eine Methode get_screener_results_df)
        # WICHTIG: Die Tabelle muss 'signal', 'rsi', 'sma_200', 'close' enthalten!
        # Falls deine DB Methode anders heißt, hier anpassen.
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        # SQL Query simulieren, um DataFrame zu bekommen
        # Hier musst du sicherstellen, dass die Spalten RSI und SMA200 in 'screener_results' gespeichert wurden
        sql = f"""
            SELECT timestamp as date, symbol, close, rsi, sma_200, signal
            FROM signals
            WHERE date >= '{start_date}'
        """

        try:
            with db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"Datenbankfehler beim Laden der Signale: {e}")
            return

        if df.empty:
            logger.info("Keine Roh-Signale im Zeitraum gefunden.")
            return

        # Spaltennamen normalisieren (alles kleinschreiben für query)
        df.columns = df.columns.str.lower()

        # 2. Filter anwenden
        final_hits = []

        for strat in strategies:
            name = strat.get("name", "Unbekannt")
            logic = strat.get("logic", "")

            try:
                # Pandas Magic: Filtert den DataFrame basierend auf dem String
                filtered_df = df.query(logic).copy()

                if not filtered_df.empty:
                    # Wir fügen den Strategie-Namen hinzu, damit man weiß, warum das Signal kommt
                    filtered_df["strategy"] = name
                    final_hits.append(filtered_df)

            except Exception as e:
                logger.error(f"Fehler in Strategie-Logik '{name}': {e}")

        # 3. Senden
        if final_hits:
            result_df = pd.concat(final_hits)

            # --- START: FORMATIERUNG ---

            # 1. Datum bereinigen: Erst in Datetime wandeln, dann strikt als "YYYY-MM-DD" String formatieren
            result_df["date"] = pd.to_datetime(result_df["date"]).dt.strftime(
                "%Y-%m-%d"
            )

            # 2. Preis schön formatieren (immer 2 Nachkommastellen), falls float
            # Wir nutzen apply, um Fehler zu vermeiden, falls es kein reiner Float ist
            result_df["close"] = result_df["close"].apply(
                lambda x: f"{float(x):.2f}" if pd.notnull(x) else x
            )

            # --- ENDE: FORMATIERUNG ---

            output_columns = ["date", "symbol", "strategy", "close"]

            # Sortieren und Reset Index für saubere Ausgabe
            display_df = (
                result_df[output_columns]
                .sort_values(by=["date", "strategy"])
                .reset_index(drop=True)
            )

            self.telegram.send_dataframe(
                display_df,
                title=f"🎯 {title_prefix}: Strategie Treffer ({len(display_df)})",
            )
            logger.info(f"Strategie-Bericht gesendet: {len(display_df)} Treffer.")
        else:
            logger.info("Keine Strategie-Treffer nach Filterung.")
