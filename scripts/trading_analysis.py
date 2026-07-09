"""
Trading Signal Analysis System - Optimized Version

Modern Python 3.12 implementation with:
- Full type annotations
- Small, testable functions
- Immutable operations
- Clear separation of concerns
- Hierarchical statistics with symbol aggregation
- TP2-only win calculation
- Configurable standard/swing range mode
- Multiprocessing for parallel file processing
- Memory-optimized categorical dtypes
- Progress bar visualization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from constants import ColumnNames

logger = logging.getLogger(__name__)


# ==================== ENUMS ====================


class TPLevel(StrEnum):
    """Take-Profit level identifiers."""

    TP1 = "1"
    TP2 = "2"
    TP3 = "3"


# ==================== DATA STRUCTURES ====================


@dataclass(frozen=True, slots=True)
class RangeResult:
    """Result of range calculation."""

    range_end: pd.Timestamp
    aktiv_date: pd.Timestamp | None
    tp_dates: dict[str, pd.Timestamp]


# ==================== PROTOCOLS ====================


class ConfigProtocol(Protocol):
    """Configuration protocol for type checking."""

    file_prefixes: list[str]
    mappings: dict[str, dict[str, str]]
    renaming: dict[str, str]
    signals: dict[str, str]
    input_pattern: str
    output_dir: Path
    min_trades: int
    use_swing_range: bool
    shift_columns: list[str]
    shift_depth: int


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
        "shift_columns",  # NEW
        "shift_depth",  # NEW
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
        shift_columns: list[str] = None,  # NEW
        shift_depth: int = 2,  # NEW
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


def load_config_with_env_override(
    config_path: str = "./scripts/signal.yaml",
) -> TradingConfig:
    """Load configuration from YAML file with environment variable overrides."""
    import os

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as f:
        raw_config: dict[str, Any] = yaml.safe_load(f)

    signals = raw_config.get("signals", {})
    mappings = raw_config.get("mappings", {})
    renaming = raw_config.get("renaming", {})
    file_prefixes = raw_config.get("file_prefixes", [])

    input_pattern = os.getenv("TRADING_INPUT_PATTERN", "*.csv")
    output_dir = Path(os.getenv("TRADING_OUTPUT_DIR", "output"))
    min_trades = int(os.getenv("TRADING_MIN_TRADES", "3"))

    use_swing_range = raw_config.get("use_swing_range", False)
    use_swing_range = os.getenv(
        "TRADING_USE_SWING_RANGE", str(use_swing_range)
    ).lower() in ("true", "1", "yes")

    shift_columns = raw_config.get("shift_columns", [])
    shift_depth = raw_config.get("shift_depth", 2)

    return TradingConfig(
        signals=signals,
        mappings=mappings,
        renaming=renaming,
        file_prefixes=file_prefixes,
        input_pattern=input_pattern,
        output_dir=output_dir,
        min_trades=min_trades,
        use_swing_range=use_swing_range,
        shift_columns=shift_columns,  # Pass new values
        shift_depth=shift_depth,
    )


# ==================== DATA PROCESSING ====================


class DataProcessor:
    """Handles CSV loading and signal transformations."""

    __slots__ = ("_config",)

    def __init__(self, config: ConfigProtocol) -> None:
        self._config = config

    def extract_symbol_name(self, file_path: Path) -> str:
        """Extract clean symbol name from file path."""
        file_name = file_path.stem
        for prefix in self._config.file_prefixes:
            file_name = file_name.removeprefix(prefix)
        return file_name.split(",")[0].strip()

    def parse_file_info(self, file_path: Path) -> tuple[str, str, str]:
        """Extract timeframe, exchange and symbol from filename."""
        stem = file_path.stem
        for prefix in self._config.file_prefixes:
            stem = stem.removeprefix(prefix)

        parts = stem.split("_", 2)
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return "unknown", "unknown", stem

    def load_csv(self, file_path: Path) -> pd.DataFrame | None:
        """Load and validate CSV file with proper error handling."""
        try:
            df = pd.read_csv(file_path)
        except (OSError, pd.errors.ParserError) as e:
            logger.error(f"Cannot read CSV {file_path.name}: {e}")
            return None

        if not self._validate_columns(df, file_path):
            return None

        df = self._prepare_dataframe(df, file_path)
        return df

    def _validate_columns(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Check if required columns exist."""
        required_cols = {ColumnNames.HIGH, ColumnNames.LOW, ColumnNames.TIME}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Missing columns in {file_path.name}: {missing}")
            return False
        return True

    def _prepare_dataframe(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Convert time, set index, add symbol, optimize dtypes."""
        df = df.copy()
        df[ColumnNames.TIME] = pd.to_datetime(df[ColumnNames.TIME], unit="s")
        df = df.set_index(ColumnNames.TIME)

        timeframe, exchange, symbol = self.parse_file_info(file_path)

        # Safe access to new constants
        col_timeframe = getattr(ColumnNames, "TIMEFRAME", "timeframe")
        col_exchange = getattr(ColumnNames, "EXCHANGE", "exchange")

        df[ColumnNames.FILE_NAME] = symbol
        df[col_timeframe] = timeframe
        df[col_exchange] = exchange

        categorical_cols = [
            col_timeframe,
            col_exchange,
            ColumnNames.FILE_NAME,
            ColumnNames.KERZE,
            ColumnNames.WOLKE,
            ColumnNames.TREND,
            ColumnNames.SETTER,
            ColumnNames.WELLE,
            ColumnNames.DELUXE,
            ColumnNames.STATUS,
            ColumnNames.MSI_DAY,
            ColumnNames.MSI_WEEK,
            ColumnNames.MSI_MONTH,
        ]

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        bool_cols = [col for col in df.columns if col.startswith(("bull_", "bear_"))]
        for col in bool_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype(bool)

        numeric_cols = [
            ColumnNames.HIGH,
            ColumnNames.LOW,
            ColumnNames.OPEN,
            ColumnNames.CLOSE,
            ColumnNames.VOLUME,
        ]
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        return df

    def merge_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply signal mappings and column renaming."""
        df = df.copy()
        df = self._apply_mappings(df)
        df = self._apply_renaming(df)
        df = self._apply_shifts(df)
        return df

    def _apply_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged columns based on configuration."""
        if not self._config.shift_columns:
            return df

        depth = self._config.shift_depth

        for col in self._config.shift_columns:
            if col not in df.columns:
                continue

            for i in range(1, depth + 1):
                new_col_name = f"{col}_prev_{i}"
                shifted_series = df[col].shift(i)

                if df[col].dtype == bool:
                    # FIX: Avoid "Downcasting object dtype arrays" warning.
                    # Instead of filling object-dtype directly, we convert:
                    # Object(True/False/NaN) -> Float(1.0/0.0/NaN) -> Fill 0 -> Bool
                    # This is explicit and safe.
                    df[new_col_name] = (
                        shifted_series.astype(float).fillna(0).astype(bool)
                    )

                elif isinstance(df[col].dtype, pd.CategoricalDtype):
                    # Handle categorical columns by ensuring NaN is a valid category
                    df[new_col_name] = shifted_series
                    if not df[new_col_name].cat.categories.isnull().any():
                        df[new_col_name] = df[new_col_name].cat.add_categories(np.nan)
                else:
                    # Standard handling for numeric/other columns
                    df[new_col_name] = shifted_series

        return df

    def _apply_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply color mappings from config."""
        for target_col, rules in self._config.mappings.items():
            for source_col, color in rules.items():
                if source_col in df.columns:
                    mask = df[source_col].notna()
                    df.loc[mask, target_col] = color

            cols_to_drop = [col for col in rules if col in df.columns]
            df = df.drop(columns=cols_to_drop)

        return df

    def _apply_renaming(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column renaming and convert to boolean."""
        for old_name, new_name in self._config.renaming.items():
            if old_name in df.columns:
                df[new_name] = df[old_name].notna()

        cols_to_drop = [col for col in self._config.renaming if col in df.columns]
        df = df.drop(columns=cols_to_drop)
        return df


# ==================== RANGE CALCULATION HELPERS ====================
# (Code unchanged from standard version)


def _find_range_end(
    lows: npt.NDArray[np.float64], start_idx: int, threshold: float
) -> int:
    future_lows = lows[start_idx:] < threshold
    if future_lows.any():
        return start_idx + np.argmax(future_lows)
    return len(lows) - 1


def _find_activation_index(
    highs: npt.NDArray[np.float64], start_idx: int, end_idx: int, trigger_level: float
) -> int | None:
    segment = highs[start_idx : end_idx + 1]
    activated = segment > trigger_level
    if activated.any():
        return start_idx + np.argmax(activated)
    return None


def _calculate_tp_levels(
    highs: npt.NDArray[np.float64],
    start_idx: int,
    end_idx: int,
    entry_price: float,
    range_size: float,
) -> dict[str, int]:
    tp_indices: dict[str, int] = {}
    segment = highs[start_idx : end_idx + 1]
    for tp_level in [TPLevel.TP1, TPLevel.TP2, TPLevel.TP3]:
        multiplier = int(tp_level)
        tp_value = entry_price + multiplier * range_size
        tp_hits = segment > tp_value
        if tp_hits.any():
            tp_indices[tp_level] = start_idx + np.argmax(tp_hits)
        else:
            break
    return tp_indices


def _calculate_swing_low(df: pd.DataFrame, idx: int, row_low: float) -> float:
    try:
        row = df.iloc[idx]
        values = [
            row_low,
            row[ColumnNames.WOLKE_LINIE_GRUEN],
            row[ColumnNames.WOLKE_LINIE_PINK],
        ]
        valid_values = [v for v in values if pd.notna(v)]
        return min(valid_values) if valid_values else row_low
    except (KeyError, TypeError, IndexError):
        return row_low


def _calculate_single_range(
    df: pd.DataFrame,
    idx: int,
    highs: npt.NDArray[np.float64],
    lows: npt.NDArray[np.float64],
    index_array: npt.NDArray[np.datetime64],
    low_threshold: float,
) -> RangeResult:
    row_high = highs[idx]
    range_end_idx = _find_range_end(lows, idx, low_threshold)
    range_end_date = pd.Timestamp(index_array[range_end_idx])
    aktiv_idx = _find_activation_index(highs, idx, range_end_idx, row_high)
    aktiv_date = pd.Timestamp(index_array[aktiv_idx]) if aktiv_idx else None
    range_size = row_high - low_threshold
    tp_indices = _calculate_tp_levels(highs, idx, range_end_idx, row_high, range_size)
    tp_dates = {
        tp_level: pd.Timestamp(index_array[tp_idx])
        for tp_level, tp_idx in tp_indices.items()
    }
    return RangeResult(
        range_end=range_end_date, aktiv_date=aktiv_date, tp_dates=tp_dates
    )


# ==================== RANGE ANALYZER ====================


class TradingRangeAnalyzer:
    """Calculates trading ranges and take-profit levels."""

    __slots__ = ()

    def calculate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._ensure_required_columns(df)
        df = self._initialize_range_columns(df)
        highs = df[ColumnNames.HIGH].to_numpy()
        lows = df[ColumnNames.LOW].to_numpy()
        index_array = df.index.to_numpy()
        mask = df[ColumnNames.LONG_RANGE].isna()
        indices_to_process = np.where(mask)[0]
        for idx in indices_to_process:
            row_low = lows[idx]
            row_high = highs[idx]
            df = self._process_standard_range(
                df, idx, highs, lows, index_array, row_low, row_high
            )
            df = self._process_swing_range(
                df, idx, highs, lows, index_array, row_low, row_high
            )
        return df

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [
            ColumnNames.HIGH,
            ColumnNames.LOW,
            ColumnNames.WOLKE_LINIE_GRUEN,
            ColumnNames.WOLKE_LINIE_PINK,
        ]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                df[col] = np.nan
        return df

    def _initialize_range_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        range_cols = [
            ColumnNames.LONG_RANGE,
            ColumnNames.LONG_AKTIV,
            ColumnNames.LONG_TP1,
            ColumnNames.LONG_TP2,
            ColumnNames.LONG_TP3,
            f"{ColumnNames.LONG_RANGE}_swing",
            f"{ColumnNames.LONG_AKTIV}_swing",
            f"{ColumnNames.LONG_TP1}_swing",
            f"{ColumnNames.LONG_TP2}_swing",
            f"{ColumnNames.LONG_TP3}_swing",
        ]
        for col in range_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object", index=df.index)
        return df

    def _process_standard_range(
        self,
        df: pd.DataFrame,
        idx: int,
        highs: npt.NDArray[np.float64],
        lows: npt.NDArray[np.float64],
        index_array: npt.NDArray[np.datetime64],
        row_low: float,
        row_high: float,
    ) -> pd.DataFrame:
        result = _calculate_single_range(df, idx, highs, lows, index_array, row_low)
        df.at[df.index[idx], ColumnNames.LONG_RANGE] = result.range_end
        df.at[df.index[idx], ColumnNames.LONG_AKTIV] = result.aktiv_date
        for tp_level, tp_date in result.tp_dates.items():
            col = f"{ColumnNames.LONG_TP}{tp_level}"
            df.at[df.index[idx], col] = tp_date
        return df

    def _process_swing_range(
        self,
        df: pd.DataFrame,
        idx: int,
        highs: npt.NDArray[np.float64],
        lows: npt.NDArray[np.float64],
        index_array: npt.NDArray[np.datetime64],
        row_low: float,
        row_high: float,
    ) -> pd.DataFrame:
        swing_low = _calculate_swing_low(df, idx, row_low)
        result = _calculate_single_range(df, idx, highs, lows, index_array, swing_low)
        df.at[df.index[idx], f"{ColumnNames.LONG_RANGE}_swing"] = result.range_end
        df.at[df.index[idx], f"{ColumnNames.LONG_AKTIV}_swing"] = result.aktiv_date
        for tp_level, tp_date in result.tp_dates.items():
            col = f"{ColumnNames.LONG_TP}{tp_level}_swing"
            df.at[df.index[idx], col] = tp_date
        return df

    def create_tp_columns(
        self, df: pd.DataFrame, prefix: str = "Long", use_swing: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        tp_col = f"{prefix}_TP"
        tp2_col = f"{prefix}_TP2_swing" if use_swing else f"{prefix}_TP2"
        aktiv_col = f"{prefix}_aktiv_swing" if use_swing else f"{prefix}_aktiv"
        range_col = f"{prefix}_Range_swing" if use_swing else f"{prefix}_Range"
        conditions = [
            df[tp2_col].notna(),
            (df[aktiv_col].notna() & df[range_col].notna() & df[tp2_col].isna()),
        ]
        choices = [2, -1]
        df[tp_col] = np.select(conditions, choices, default=0).astype(int)
        if tp2_col in df.columns:
            df = df.rename(columns={tp2_col: f"{tp2_col}_date"})
            df.loc[df[f"{tp2_col}_date"].notna(), tp2_col] = 1
        return df


# ==================== STATISTICS ====================


class StatisticsCalculator:
    """Calculate win/loss statistics."""

    __slots__ = ()

    def calculate_group_statistics(
        self, df_filtered: pd.DataFrame, group_cols: list[str]
    ) -> pd.DataFrame:
        if df_filtered.empty:
            return pd.DataFrame()
        result = (
            df_filtered.groupby(
                group_cols + [ColumnNames.LONG_TP], observed=True, dropna=False
            )[ColumnNames.FILE_NAME]
            .count()
            .reset_index(name="count")
        )
        result_pivot = result.pivot_table(
            index=group_cols, columns=ColumnNames.LONG_TP, values="count", fill_value=0
        ).reset_index()
        column_mapping = {-1: "loss", 0: "rejected", 2: "win"}
        result_pivot = result_pivot.rename(columns=column_mapping)
        for col in ["loss", "win", "rejected"]:
            if col not in result_pivot.columns:
                result_pivot[col] = 0
        return result_pivot

    def generate_hierarchical_stats(
        self, df_filtered: pd.DataFrame, min_trades: int = 3
    ) -> list[dict[str, Any]]:
        if df_filtered.empty:
            return []
        results: list[dict[str, Any]] = []
        self._add_overall_stats(df_filtered, min_trades, results)
        self._add_hierarchical_combinations(df_filtered, min_trades, results)
        return results

    def _add_overall_stats(
        self, df_filtered: pd.DataFrame, min_trades: int, results: list[dict[str, Any]]
    ) -> None:
        overall_stats = self.calculate_group_statistics(df_filtered, [])
        if not overall_stats.empty:
            row_dict = overall_stats.iloc[0][["loss", "win", "rejected"]].to_dict()
            total = sum(row_dict.values())
            if total >= min_trades:
                row_dict["total"] = total
                row_dict["level"] = "gesamt"
                results.append(row_dict)

    def _add_hierarchical_combinations(
        self, df_filtered: pd.DataFrame, min_trades: int, results: list[dict[str, Any]]
    ) -> None:
        hierarchies = [
            ([ColumnNames.WOLKE], "wolke"),
            ([ColumnNames.WOLKE, ColumnNames.WELLE], "wolke_welle"),
            (
                [ColumnNames.WOLKE, ColumnNames.WELLE, ColumnNames.TREND],
                "wolke_welle_trend",
            ),
            (
                [
                    ColumnNames.WOLKE,
                    ColumnNames.WELLE,
                    ColumnNames.TREND,
                    ColumnNames.SETTER,
                ],
                "wolke_welle_trend_setter",
            ),
        ]
        for group_cols, level_name in hierarchies:
            if all(col in df_filtered.columns for col in group_cols):
                stats = self.calculate_group_statistics(df_filtered, group_cols)
                self._append_valid_stats(stats, level_name, min_trades, results)

    def _append_valid_stats(
        self,
        stats_df: pd.DataFrame,
        level_name: str,
        min_trades: int,
        results: list[dict[str, Any]],
    ) -> None:
        for _, row in stats_df.iterrows():
            total = row["loss"] + row["win"] + row["rejected"]
            if total >= min_trades:
                result_dict = row.to_dict()
                result_dict["total"] = total
                result_dict["level"] = level_name
                results.append(result_dict)


# ==================== SIGNAL FILTER ====================


class SignalFilter:
    __slots__ = ()

    def filter_by_signal(
        self, df: pd.DataFrame, signal_name: str, query: str
    ) -> pd.DataFrame:
        try:
            df_filtered = df.query(query).copy()
            if df_filtered.empty:
                return pd.DataFrame()
            return self._select_relevant_columns(df_filtered)
        except (ValueError, KeyError, pd.errors.UndefinedVariableError) as e:
            logger.error(f"Error filtering signal '{signal_name}': {e}")
            return pd.DataFrame()

    def _select_relevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        base_cols = [ColumnNames.FILE_NAME, ColumnNames.LONG_TP]
        optional_cols = [
            ColumnNames.DELUXE,
            ColumnNames.STATUS,
            ColumnNames.KERZE,
            ColumnNames.WOLKE,
            ColumnNames.TREND,
            ColumnNames.SETTER,
            ColumnNames.WELLE,
            # ColumnNames.RSI,
        ]
        cols_to_keep = base_cols + [col for col in optional_cols if col in df.columns]
        return df[cols_to_keep]


# ==================== RESULT EXPORTER ====================


class ResultExporter:
    """Export statistics to CSV."""

    __slots__ = ()

    def save_statistics(
        self,
        statistics: dict[str, dict[str, list[dict[str, Any]]]],
        signal_definitions: dict[str, str],
        output_file: Path,
    ) -> None:
        rows = self._flatten_statistics(statistics)
        if not rows:
            logger.warning("No data to save")
            return
        df_output = pd.DataFrame(rows)
        df_output = self._add_aggregated_summaries(df_output)
        df_output = self._reorder_columns(df_output)
        df_output = self._calculate_rates(df_output)
        df_output = self._sort_results(df_output)
        self._write_csv(df_output, output_file)

    def _flatten_statistics(
        self, statistics: dict[str, dict[str, list[dict[str, Any]]]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for signal_name, symbol_stats in statistics.items():
            for symbol, stats_list in symbol_stats.items():
                for stats in stats_list:
                    row = {"signal": signal_name, "symbol": symbol}
                    row.update(stats)
                    rows.append(row)
        return rows

    def _add_aggregated_summaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary rows aggregating symbols but keeping Timeframe/Exchange separate."""
        if df.empty:
            return df

        summary_rows = []
        dimension_cols = [
            ColumnNames.WOLKE,
            ColumnNames.WELLE,
            ColumnNames.TREND,
            ColumnNames.SETTER,
        ]
        available_dimensions = [col for col in dimension_cols if col in df.columns]

        # New: Group also by Timeframe and Exchange to avoid mixing 15m and 1h data
        meta_cols = ["timeframe"]  # safe literals
        available_meta = [c for c in meta_cols if c in df.columns]

        group_cols = ["signal", "level"] + available_meta + available_dimensions

        # GroupBy requires careful handling of non-existing columns, so we only group by what's there
        for group_keys, group in df.groupby(group_cols, dropna=False):
            # The keys correspond to the order of group_cols
            # signal = keys[0], level = keys[1]
            # then meta cols, then dimensions

            summary = {
                "signal": group_keys[0],
                "symbol": "ALL_SYMBOLS",
                "level": group_keys[1],
                "total": group["total"].sum(),
                "win": group["win"].sum(),
                "loss": group["loss"].sum(),
                "rejected": group["rejected"].sum(),
            }

            # Extract metadata values (timeframe, exchange) from the group key
            offset = 2
            if available_meta:
                meta_values = group_keys[offset : offset + len(available_meta)]
                summary.update(dict(zip(available_meta, meta_values, strict=False)))
                offset += len(available_meta)

            # Extract dimension values
            if available_dimensions:
                dim_values = group_keys[offset:]
                summary.update(
                    dict(zip(available_dimensions, dim_values, strict=False))
                )

            summary_rows.append(summary)

        df_summary = pd.DataFrame(summary_rows)
        df_combined = pd.concat([df, df_summary], ignore_index=True)
        return df_combined

    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns: Signal, Symbol, Timeframe, Exchange, Level, ..."""
        # Define preferred order
        base_cols = [
            "signal",
            "symbol",
            "timeframe",
            "exchange",
            "level",
            "total",
            "win",
            "loss",
            "rejected",
        ]

        # Only use columns that actually exist
        present_base_cols = [c for c in base_cols if c in df.columns]

        dimension_cols = [
            ColumnNames.WOLKE,
            ColumnNames.WELLE,
            ColumnNames.TREND,
            ColumnNames.SETTER,
        ]
        available_dimensions = [col for col in dimension_cols if col in df.columns]

        other_cols = [
            col
            for col in df.columns
            if col not in present_base_cols + available_dimensions
            and not col.startswith("tp")
        ]
        return df[present_base_cols + available_dimensions + other_cols]

    def _calculate_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["win_rate"] = (
            df["win"] / (df["win"] + df["loss"]).replace(0, np.nan) * 100
        ).round(2)
        df["loss_rate"] = (
            df["loss"] / (df["win"] + df["loss"]).replace(0, np.nan) * 100
        ).round(2)
        df[["win_rate", "loss_rate"]] = df[["win_rate", "loss_rate"]].fillna(0)
        return df

    def _sort_results(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_sort_symbol"] = df["symbol"].apply(
            lambda x: "ZZZZ" if x == "ALL_SYMBOLS" else x
        )
        df_sorted = df.sort_values(
            ["signal", "_sort_symbol", "level", "total"],
            ascending=[True, True, True, False],
        )
        df_sorted = df_sorted.drop(columns=["_sort_symbol"])
        return df_sorted

    def _write_csv(self, df: pd.DataFrame, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8")
        try:
            df.to_csv("./data/croc_statistik.csv", index=False, encoding="utf-8")
        except Exception as e:
            logger.info(f"Write to ./data/croc_statistec.csv failed: {e}")

        logger.info(f"Statistics saved to: {output_file}")
        logger.info(
            f" Rows: {len(df):,} | Symbols: {df[df['symbol'] != 'ALL_SYMBOLS']['symbol'].nunique()} | Signals: {df['signal'].nunique()}"
        )


# ==================== MULTIPROCESSING WORKER ====================


def process_file_worker(
    file_path: Path, config_dict: dict[str, Any]
) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    try:
        config = TradingConfig(
            signals=config_dict["signals"],
            mappings=config_dict["mappings"],
            renaming=config_dict["renaming"],
            file_prefixes=config_dict["file_prefixes"],
            input_pattern=config_dict["input_pattern"],
            output_dir=Path(config_dict["output_dir"]),
            min_trades=config_dict["min_trades"],
            use_swing_range=config_dict["use_swing_range"],
            shift_columns=config_dict.get("shift_columns", []),
            shift_depth=config_dict.get("shift_depth", 2),
        )
        data_processor = DataProcessor(config)
        range_analyzer = TradingRangeAnalyzer()
        stats_calculator = StatisticsCalculator()
        signal_filter = SignalFilter()

        df = data_processor.load_csv(file_path)
        if df is None:
            return {}

        symbol = df[ColumnNames.FILE_NAME].iloc[0]
        # Robustly get new columns using getattr fallback
        col_tf = getattr(ColumnNames, "TIMEFRAME", "timeframe")
        col_ex = getattr(ColumnNames, "EXCHANGE", "exchange")
        timeframe = df[col_tf].iloc[0]
        exchange = df[col_ex].iloc[0]

        df = data_processor.merge_signals(df)
        df = range_analyzer.calculate_ranges(df)
        df = range_analyzer.create_tp_columns(
            df, prefix="Long", use_swing=config.use_swing_range
        )

        results: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        for signal_name, query in config.signals.items():
            df_signal = signal_filter.filter_by_signal(df, signal_name, query)
            if df_signal.empty:
                continue
            stats = stats_calculator.generate_hierarchical_stats(
                df_signal, min_trades=config.min_trades
            )
            if stats:
                # INJECT TIMEFRAME AND EXCHANGE INTO STATS HERE
                for s in stats:
                    s["timeframe"] = timeframe
                    s["exchange"] = exchange
                results[signal_name] = (symbol, stats)
        return results
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        return {}


# ==================== MAIN PIPELINE ====================


class TradingAnalysisPipeline:
    """Main pipeline orchestrating the analysis with multiprocessing and progress bar."""

    __slots__ = (
        "_config",
        "_data_processor",
        "_range_analyzer",
        "_stats_calculator",
        "_signal_filter",
        "_exporter",
    )

    def __init__(self, config: ConfigProtocol) -> None:
        self._config = config
        self._data_processor = DataProcessor(config)
        self._range_analyzer = TradingRangeAnalyzer()
        self._stats_calculator = StatisticsCalculator()
        self._signal_filter = SignalFilter()
        self._exporter = ResultExporter()

    def process_file(
        self, file_path: Path
    ) -> dict[str, tuple[str, list[dict[str, Any]]]]:
        # Single threaded version (for debugging), mirroring worker logic
        df = self._data_processor.load_csv(file_path)
        if df is None:
            return {}
        symbol = df[ColumnNames.FILE_NAME].iloc[0]
        col_tf = getattr(ColumnNames, "TIMEFRAME", "timeframe")
        col_ex = getattr(ColumnNames, "EXCHANGE", "exchange")
        timeframe = df[col_tf].iloc[0]
        exchange = df[col_ex].iloc[0]

        df = self._data_processor.merge_signals(df)
        df = self._range_analyzer.calculate_ranges(df)
        df = self._range_analyzer.create_tp_columns(
            df, prefix="Long", use_swing=self._config.use_swing_range
        )

        results = {}
        for signal_name, query in self._config.signals.items():
            df_signal = self._signal_filter.filter_by_signal(df, signal_name, query)
            if df_signal.empty:
                continue
            stats = self._stats_calculator.generate_hierarchical_stats(
                df_signal, min_trades=self._config.min_trades
            )
            if stats:
                for s in stats:
                    s["timeframe"] = timeframe
                    s["exchange"] = exchange
                results[signal_name] = (symbol, stats)
        return results

    def run(self) -> None:
        if not self._config.signals:
            logger.warning("No signals defined!")
            return
        files_list = list(Path("././in").glob(self._config.input_pattern))
        if not files_list:
            logger.warning(f"No files found: {self._config.input_pattern}")
            return
        self._log_pipeline_start(files_list)
        statistics = self._process_all_files_parallel(files_list)
        self._export_results(statistics)

    def _log_pipeline_start(self, files_list: list[Path]) -> None:
        logger.info("=" * 80)
        logger.info(
            f"Processing {len(files_list)} files with {len(self._config.signals)} signals"
        )
        logger.info(f"Min. trades: {self._config.min_trades}")
        logger.info(
            f"Range mode: {'SWING' if self._config.use_swing_range else 'STANDARD'}"
        )
        logger.info("=" * 80)
        for signal_name, query in self._config.signals.items():
            logger.info(f" - {signal_name:20s} | Query: {query}")

    def _process_all_files_parallel(
        self, files_list: list[Path]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        import os
        from functools import partial
        from multiprocessing import Pool, cpu_count

        try:
            from tqdm import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False
            logger.warning("tqdm not installed. Install with: pip install tqdm")

        statistics: dict[str, dict[str, list[dict[str, Any]]]] = {
            signal_name: {} for signal_name in self._config.signals
        }
        num_workers = int(os.getenv("TRADING_NUM_WORKERS", "0"))
        if num_workers <= 0:
            num_workers = cpu_count()
        logger.info(f"Using {num_workers} parallel workers (of {cpu_count()} CPUs)")

        config_dict = self._serialize_config()
        worker_func = partial(process_file_worker, config_dict=config_dict)

        if num_workers == 1:
            if has_tqdm:
                results_list = [
                    worker_func(file_path=fp)
                    for fp in tqdm(
                        files_list, desc="Processing", unit="file", ncols=100
                    )
                ]
            else:
                results_list = [worker_func(file_path=fp) for fp in files_list]
        else:
            with Pool(processes=num_workers) as pool:
                if has_tqdm:
                    results_list = list(
                        tqdm(
                            pool.imap_unordered(worker_func, files_list, chunksize=1),
                            total=len(files_list),
                            desc="Processing",
                            unit="file",
                            ncols=100,
                        )
                    )
                else:
                    worker_args = [(fp, config_dict) for fp in files_list]
                    results_list = pool.starmap(process_file_worker, worker_args)

        # AGGREGATION FIX: Avoid overwriting data when symbol is same but timeframe differs
        for results in results_list:
            if results:
                for signal_name, (symbol, stats_list) in results.items():
                    # We use a list to accumulate results for the same symbol
                    if symbol not in statistics[signal_name]:
                        statistics[signal_name][symbol] = []
                    statistics[signal_name][symbol].extend(stats_list)
        return statistics

    def _serialize_config(self) -> dict[str, Any]:
        return {
            "signals": self._config.signals,
            "mappings": self._config.mappings,
            "renaming": self._config.renaming,
            "file_prefixes": self._config.file_prefixes,
            "input_pattern": self._config.input_pattern,
            "output_dir": str(self._config.output_dir),
            "min_trades": self._config.min_trades,
            "use_swing_range": self._config.use_swing_range,
            "shift_columns": self._config.shift_columns,
            "shift_depth": self._config.shift_depth,
        }

    def _export_results(
        self, statistics: dict[str, dict[str, list[dict[str, Any]]]]
    ) -> None:
        total_stats = sum(len(symbol_dict) for symbol_dict in statistics.values())
        if total_stats > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self._config.output_dir / f"statistik_{timestamp}.csv"
            self._exporter.save_statistics(
                statistics, self._config.signals, output_file
            )
        else:
            logger.warning(
                f"No statistics with at least {self._config.min_trades} trades found"
            )


def main() -> None:
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config = load_config_with_env_override()
    pipeline = TradingAnalysisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
