import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import ColumnNames

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.mappings = config.get("mappings", {})
        self.renaming = config.get("renaming", {})
        self.combinations = config.get("combinations", [])

    def load_and_process(self, file_path: Path) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Fehler beim Lesen von {file_path}: {e}")
            return None

        # Basis-Validierung (High/Low/Time müssen da sein)
        required = {ColumnNames.HIGH, ColumnNames.LOW, ColumnNames.TIME}
        if not required.issubset(df.columns):
            return None

        # 1. Standard Vorbereitung (Index, Dateiname)
        df = self._prepare_basics(df, file_path)

        # 2. Renaming (WICHTIG: Übersetzt CSV-Namen in System-Namen)
        df = self._apply_renaming(df)

        # 3. Mappings (Wandelt Text-Werte in Farben um)
        df = self._apply_mappings(df)

        # 4. Technische Indikatoren sicherstellen
        df = self._ensure_indicators(df)

        # 5. Kombinations-Signale erstellen
        df = self._apply_combinations(df)

        return df

    def _prepare_basics(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        df = df.copy()
        # Zeitstempel konvertieren
        if df[ColumnNames.TIME].dtype == "object":
            df[ColumnNames.TIME] = pd.to_datetime(df[ColumnNames.TIME])
        else:
            df[ColumnNames.TIME] = pd.to_datetime(df[ColumnNames.TIME], unit="s")

        df = df.set_index(ColumnNames.TIME)
        df[ColumnNames.FILE_NAME] = file_path.stem

        # Strings bereinigen (Leerzeichen, Lowercase) für robuste Vergleiche
        str_cols = ["wolke", "deluxe", "status", "trend", "welle", "kerze", "setter"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()

        return df

    def _apply_renaming(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet die 'renaming'-Regeln aus der Config an.
        Macht aus 'Red Devil Long' -> 'bull_rot'.
        """
        for old_name, new_name in self.renaming.items():
            if old_name in df.columns:
                # Signal aktiv (1) wenn nicht NaN
                df[new_name] = df[old_name].notna().astype(int)
        return df

    def _apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet Wert-Mappings an (z.B. deluxe 'Over Bought' -> 'grün').
        FIX: Initialisiert Spalten korrekt als 'object', um FutureWarnings zu vermeiden.
        """
        for target_col, rules in self.mappings.items():
            # Wenn Zielspalte noch nicht existiert, explizit als object (String) anlegen
            if target_col not in df.columns:
                df[target_col] = pd.Series(np.nan, index=df.index, dtype=object)

            for source_col, color_val in rules.items():
                if source_col in df.columns:
                    mask = df[source_col].notna()

                    # Sicherheits-Check: Falls Spalte doch float ist, umwandeln
                    if df[target_col].dtype != object:
                        df[target_col] = df[target_col].astype(object)

                    df.loc[mask, target_col] = color_val.lower()
        return df

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Falls EMA oder RSI fehlen, füllen wir sie mit NaN
        if "EMA" not in df.columns:
            df["EMA"] = np.nan
        if "RSI" not in df.columns:
            df["RSI"] = np.nan

        # Typkonvertierung für saubere Vergleiche
        cols_to_numeric = ["close", "EMA", "RSI"]
        for c in cols_to_numeric:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df

    def _apply_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Erzeugt neue Spalten basierend auf der Config-Logik."""
        for combo in self.combinations:
            name = combo["name"]
            logic = combo["logic"]
            try:
                # Pandas eval ausführen
                df[name] = df.eval(logic).fillna(0).astype(int)
            except Exception:
                df[name] = 0
        return df
