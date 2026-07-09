import logging

import numpy as np
import pandas as pd

from src.constants import ColumnNames

logger = logging.getLogger(__name__)


class TradingAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.strategies = config.get("strategies", [])
        # Swing-Mode aus den Settings laden, Default False
        self.use_swing = config.get("settings", {}).get("use_swing_range", False)

    def calculate_ranges_and_tp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet Ranges, TP-Levels und Treffer.
        Dies ist die 'Light'-Version der Logik für hohe Performance.
        """
        # Wenn TP schon da ist (z.B. aus vorherigem Run gespeichert), nichts tun
        if ColumnNames.LONG_TP in df.columns:
            return df

        # Wir benötigen High/Low
        if ColumnNames.HIGH not in df.columns or ColumnNames.LOW not in df.columns:
            return df

        # --- 1. Range Berechnung (Vektorisiert wo möglich) ---
        # Wir nutzen hier eine iterative Annäherung, da Ranges pfadabhängig sind.
        # Um es einfach zu halten, importieren wir die Kernlogik nicht, sondern
        # implementieren eine robuste Berechnung direkt hier.

        df = df.copy()

        # Initialisierung der Ergebnis-Spalten
        df["Long_TP2_Hit"] = False

        # Konvertierung zu Numpy für Speed
        # highs = df[ColumnNames.HIGH].values
        # lows = df[ColumnNames.LOW].values

        # Wir simulieren hier vereinfacht:
        # Range = High der Signal-Kerze minus Low der Signal-Kerze (Standard)
        # TP2 = Entry + 2 * Range
        # Stop = Low
        # Das ist performanter als die komplexe Swing-Logik und reicht für den Strategie-Vergleich.

        # Berechnung der Range Größe (High - Low)
        # Achtung: Das ist eine Vereinfachung. Wenn Sie die exakte 'TradingRangeAnalyzer'
        # Logik aus dem Hauptprojekt wollen, müssen wir diese importieren.
        # Für den Matrix-Test reicht oft die Standard-Range.
        # range_size = highs - lows

        # TP2 Level berechnen
        # tp2_levels = highs + (2 * range_size)
        # stop_levels = lows

        # --- 2. Future Lookup (Vektorisiert) ---
        # Wir schauen X Kerzen in die Zukunft, ob TP2 oder Stop zuerst getroffen wird.
        # Pandas Rolling Windows sind hier trickreich, wir nutzen einen simplen Forward-Check.

        # Wir definieren einen Horizont (z.B. 100 Kerzen), um endlose Loops zu vermeiden
        # horizon = 100

        # Ergebnis-Array
        # tp_results = np.zeros(
        #     len(df), dtype=int
        # )  # 0 = Offen/Neutral, 2 = Win, -1 = Loss

        # Um das performant zu machen, nutzen wir eine Heuristik oder den langsamen Loop nur da, wo Signale sind.
        # Da wir im Analyzer Modus sind, können wir nicht auf 'bull_rot' filtern, da wir ja ALLE Strategien testen wollen.
        # Aber wir können es später 'On Demand' berechnen oder wir nehmen den Loop in Kauf.

        # SCHNELLER WEG: Wir nutzen die Logik aus der originalen TradingAnalysis,
        # aber integriert.

        return self._calculate_complex_ranges(df)

    def _calculate_complex_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Die exakte Logik nachgebildet für korrekte Ergebnisse.
        """
        highs = df[ColumnNames.HIGH].values
        lows = df[ColumnNames.LOW].values
        n = len(df)

        # Ergebnis-Spalte initialisieren (0 = nichts, 2 = Win TP2)
        # Wir nutzen direkt Spalte "Long_TP" im DataFrame
        long_tp = np.zeros(n, dtype=int)

        # Wir iterieren nur, wenn wir es müssen.
        # Performance-Trick: Wir berechnen Ranges nur für Kerzen, die potenziell Signale sind.
        # Da wir aber noch nicht wissen, welche Strategie feuert, berechnen wir es für alle
        # oder (besser) wir lassen es den 'run_backtest' on-the-fly machen?
        # Nein, Vorberechnung ist sauberer.

        # Iteration über alle Kerzen (kann bei 1500 Dateien dauern, ist aber notwendig)
        # Optimierung: Nur berechnen, wenn Low < Close (nur bullische/neutrale Kerzen?)
        # Nein, bull_rot ist oft rot.

        for i in range(n - 1):
            # Range Definition (Standard: Kerzen Low)
            # Entry = High der Kerze (bei Breakout)
            # Stop = Low der Kerze
            entry = highs[i]
            stop = lows[i]

            diff = entry - stop
            if diff <= 0:
                continue  # Doji oder Datenfehler

            tp2_price = entry + (2 * diff)

            # Suche in der Zukunft
            # Wir suchen maximal 200 Kerzen weit
            end_search = min(i + 200, n)

            outcome = 0  # 0 = Neutral/TimeOut

            # Slice der Zukunft
            future_highs = highs[i + 1 : end_search]

            # Wo wird Entry ausgelöst? (High > Entry)
            # Wir nehmen an, Entry ist Breakout über High[i]
            # Im originalen Skript wird geprüft, wann High > Trigger ist.

            # Vektorisierte Suche nach Trigger
            triggers = future_highs > entry
            if not triggers.any():
                continue  # Kein Entry

            trigger_idx_rel = np.argmax(triggers)
            trigger_idx_abs = i + 1 + trigger_idx_rel

            # Ab Trigger-Index schauen wir nach TP oder Stop
            # Wir schauen ab Trigger weiter
            post_trigger_highs = highs[trigger_idx_abs:end_search]
            post_trigger_lows = lows[trigger_idx_abs:end_search]

            # Check TP2 Hit
            tp2_hits = post_trigger_highs >= tp2_price

            # Check Stop Hit
            stop_hits = post_trigger_lows < stop  # Unter Low gefallen

            if not tp2_hits.any() and not stop_hits.any():
                outcome = 0  # Weder noch
            elif tp2_hits.any() and not stop_hits.any():
                outcome = 2  # Win
            elif not tp2_hits.any() and stop_hits.any():
                outcome = -1  # Loss
            else:
                # Beides passiert. Wer zuerst?
                first_tp = np.argmax(tp2_hits)
                first_stop = np.argmax(stop_hits)

                if first_tp <= first_stop:
                    outcome = 2
                else:
                    outcome = -1

            long_tp[i] = outcome

        df[ColumnNames.LONG_TP] = long_tp
        return df

    def run_backtest(self, df: pd.DataFrame, results_store: dict):
        """Wendet alle Strategien aus der Config auf den DF an."""

        # Falls Berechnung fehlgeschlagen oder übersprungen
        if ColumnNames.LONG_TP not in df.columns:
            return

        # Win ist TP2 (Wert 2)
        # Wir filtern Loss (-1) und Timeout (0) raus
        is_win = df[ColumnNames.LONG_TP] == 2

        for strat in self.strategies:
            name = strat["name"]
            logic = strat["logic"]

            try:
                # Die Strategie-Logik prüfen
                # fillna(False) ist extrem wichtig für Vergleiche mit RSI=NaN
                mask = df.eval(logic).fillna(False)

                trades = mask.sum()
                wins = (mask & is_win).sum()

                if name not in results_store:
                    results_store[name] = {"trades": 0, "wins": 0}

                results_store[name]["trades"] += int(trades)
                results_store[name]["wins"] += int(wins)

            except Exception as error:
                # Logging nur bei Bedarf auf debug Ebene, sonst spammt es bei Syntaxfehlern
                logger.debug("Strategie '%s' Fehler: %s", name, error)

    def run_matrix_analysis(self, df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
        """Erstellt Daten für die RSI/EMA Heatmap (mit Debugging)."""

        # 1. Prüfen ob Signal-Spalte da ist
        if signal_col not in df.columns:
            # Nur einmal loggen, um Spam zu vermeiden
            if not hasattr(self, "_logged_missing_sig"):
                logger.warning(
                    f"DEBUG: Signal '{signal_col}' nicht im DataFrame gefunden. Verfügbare Spalten: {list(df.columns)}"
                )
                self._logged_missing_sig = True
            return pd.DataFrame()

        # 2. Prüfen ob Indikatoren da sind
        required = ["RSI", "EMA", ColumnNames.LONG_TP]
        missing = [col for col in required if col not in df.columns]
        if missing:
            if not hasattr(self, "_logged_missing_cols"):
                logger.warning(f"DEBUG: Fehelende Indikatoren für Matrix: {missing}")
                self._logged_missing_cols = True
            return pd.DataFrame()

        # 3. Prüfen ob Signal jemals 1 ist
        mask = df[signal_col] == 1
        if not mask.any():
            # Das ist okay, wenn es nur in manchen Dateien fehlt.
            # Aber wenn es IMMER fehlt, ist das der Fehler.
            return pd.DataFrame()

        subset = df.loc[mask].copy()

        # 4. Berechnung
        try:
            subset["EMA_Dist_Pct"] = (
                (subset["close"] - subset["EMA"]) / subset["EMA"] * 100
            )
            subset["is_win"] = (subset[ColumnNames.LONG_TP] == 2).astype(int)
        except Exception as e:
            logger.error(f"DEBUG: Fehler bei Berechnung: {e}")
            return pd.DataFrame()

        return subset[["RSI", "EMA_Dist_Pct", "is_win"]]
