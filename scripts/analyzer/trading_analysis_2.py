"""
Trading Signal Analysis System - Optimized Version with Combination Support
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from constants import ColumnNames

logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================


class TradingConfig:
    """Trading analysis configuration container."""

    __slots__ = (
        "signals",
        "mappings",
        "renaming",
        "file_prefixes",
        "input_pattern",
        "output_dir",
        "min_trades",
        "use_swing_range",
        "shift_columns",
        "shift_depth",
        "combinations",  # <--- NEU
    )

    def __init__(
        self,
        signals: dict[str, str],
        mappings: dict[str, dict[str, str]],
        renaming: dict[str, str],
        file_prefixes: list[str],
        input_pattern: str,
        output_dir: Path,
        min_trades: int,
        use_swing_range: bool = False,
        shift_columns: list[str] = None,
        shift_depth: int = 2,
        combinations: list[dict[str, str]] = None,  # <--- NEU
    ) -> None:
        self.signals = signals
        self.mappings = mappings
        self.renaming = renaming
        self.file_prefixes = file_prefixes
        self.input_pattern = input_pattern
        self.output_dir = output_dir
        self.min_trades = min_trades
        self.use_swing_range = use_swing_range
        self.shift_columns = shift_columns or []
        self.shift_depth = shift_depth
        self.combinations = combinations or []  # <--- NEU


def load_config_with_env_override(
    config_path: str = "./scripts/signal.yaml",
) -> TradingConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        # Fallback Suche
        config_file = Path("signal.yaml")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    return TradingConfig(
        signals=raw_config.get("signals", {}),
        mappings=raw_config.get("mappings", {}),
        renaming=raw_config.get("renaming", {}),
        file_prefixes=raw_config.get("file_prefixes", []),
        input_pattern=os.getenv("TRADING_INPUT_PATTERN", "*.csv"),
        output_dir=Path(os.getenv("TRADING_OUTPUT_DIR", "output")),
        min_trades=int(os.getenv("TRADING_MIN_TRADES", "3")),
        use_swing_range=raw_config.get("use_swing_range", False),
        shift_columns=raw_config.get("shift_columns", []),
        shift_depth=raw_config.get("shift_depth", 2),
        combinations=raw_config.get("combinations", []),  # <--- NEU
    )


# ==================== DATA PROCESSING ====================


class DataProcessor:
    """Handles CSV loading and signal transformations."""

    __slots__ = ("_config",)

    def __init__(self, config: TradingConfig) -> None:  # Type hint updated
        self._config = config

    # ... (load_csv, extract_symbol_name, parse_file_info bleiben gleich) ...
    # ... (Kopiere die Methoden load_csv, _validate_columns, _prepare_dataframe von oben hier rein) ...

    # HIER NUR DIE GEÄNDERTEN METHODEN ZUR ÜBERSICHTLICHKEIT:

    def load_csv(self, file_path: Path) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(file_path)
        except (OSError, pd.errors.ParserError) as e:
            logger.error(f"Cannot read CSV {file_path.name}: {e}")
            return None

        # Minimal validation
        if not {ColumnNames.HIGH, ColumnNames.LOW, ColumnNames.TIME}.issubset(
            df.columns
        ):
            return None

        return self._prepare_dataframe(df, file_path)

    def _prepare_dataframe(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        df = df.copy()
        df[ColumnNames.TIME] = pd.to_datetime(df[ColumnNames.TIME], unit="s")
        df = df.set_index(ColumnNames.TIME)

        # Meta Infos
        file_name = file_path.stem
        for prefix in self._config.file_prefixes:
            file_name = file_name.removeprefix(prefix)
        df[ColumnNames.FILE_NAME] = file_name.split(",")[0].strip()

        # Numeric Cast
        cols = [
            ColumnNames.HIGH,
            ColumnNames.LOW,
            ColumnNames.OPEN,
            ColumnNames.CLOSE,
            ColumnNames.VOLUME,
        ]
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(np.float32)

        return df

    def merge_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply signal mappings, renaming, shifting AND combinations."""
        df = df.copy()
        df = self._apply_mappings(df)
        df = self._apply_renaming(df)
        df = self._apply_shifts(df)
        df = self._apply_combinations(df)  # <--- NEU: Kombis berechnen
        return df

    def _apply_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Erstellt Kombinations-Signale basierend auf der Config.
        Nutzt pandas.eval() für schnelle Berechnung von Logik-Strings.
        """
        if not self._config.combinations:
            return df

        for combo in self._config.combinations:
            name = combo.get("name")
            logic = combo.get("logic")

            if not name or not logic:
                continue

            try:
                # eval() führt den String als Code auf dem DataFrame aus
                # z.B. "bull_rot == 1 and bull_1 == 1"
                # Wir füllen NaNs mit 0, damit der Vergleich nicht crasht
                df[name] = df.eval(logic).fillna(False).astype(int)
            except Exception:
                # Logger auf Debug, da manche Signale in manchen Dateien fehlen können
                # logger.debug(f"Konnte Kombination '{name}' nicht erstellen: {e}")
                df[name] = 0  # Fallback

        return df

    # ... (Rest der Methoden _apply_shifts, _apply_mappings, _apply_renaming bleiben wie im Original) ...
    def _apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        for target_col, rules in self._config.mappings.items():
            for source_col, color in rules.items():
                if source_col in df.columns:
                    mask = df[source_col].notna()
                    df.loc[mask, target_col] = color
            cols_to_drop = [col for col in rules if col in df.columns]
            df = df.drop(columns=cols_to_drop)
        return df

    def _apply_renaming(self, df: pd.DataFrame) -> pd.DataFrame:
        for old_name, new_name in self._config.renaming.items():
            if old_name in df.columns:
                df[new_name] = df[old_name].notna()  # Convert to Bool existence

        # Cleanup old columns
        cols_to_drop = [col for col in self._config.renaming if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        return df

    def _apply_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._config.shift_columns:
            return df
        for col in self._config.shift_columns:
            if col not in df.columns:
                continue
            for i in range(1, self._config.shift_depth + 1):
                new_col = f"{col}_prev_{i}"
                df[new_col] = df[col].shift(i)
        return df


# ==================== RANGE ANALYZER (Kopieren Sie die Klasse aus Ihrem Original) ====================
class TradingRangeAnalyzer:
    def calculate_ranges(self, df):
        return df  # Placeholder, bitte Original-Code nutzen

    def create_tp_columns(self, df, prefix="Long", use_swing=False):
        return df  # Placeholder


# Hinweis: Da ich die Datei trading_analysis.py nicht komplett neu schreiben kann ohne Kontextverlust,
# ist der wichtige Teil oben die Klasse `DataProcessor` mit `_apply_combinations`.
# Bitte fügen Sie diese Logik in Ihre bestehende Datei ein.
