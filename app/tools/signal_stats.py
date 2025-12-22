from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# Paths / configuration
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]  # script in ./tools -> project root
IN_DIR = BASE_DIR / "in"
YAML_PATH = BASE_DIR / "signal.yaml"


# --------------------------------------------------------------------------------------
# Load YAML configuration
# --------------------------------------------------------------------------------------


def load_signal_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


cfg = load_signal_config(YAML_PATH)  # [file:76]
SIGNAL_DEFS: Dict[str, str] = cfg["signals"]  # signal name -> expression [file:76]
COLOR_MAPPINGS: Dict[str, Dict[str, str]] = cfg[
    "mappings"
]  # logical color groups [file:76]
RENAMING: Dict[str, str] = cfg["renaming"]  # German -> English column names [file:76]
FILE_PREFIXES: List[str] = cfg.get(
    "file_prefixes", []
)  # e.g. BATS_, NASDAQ_DLY_ [file:76]


# --------------------------------------------------------------------------------------
# Filename parsing: derive symbol + timeframe
# --------------------------------------------------------------------------------------


def parse_filename(path: Path) -> Tuple[str, str]:
    """
    Extract (symbol, timeframe) from filenames like:
        'BATS_AAPL, 1D.csv'
    using known prefixes from signal.yaml. [file:76]
    """
    stem = path.stem  # e.g. "BATS_AAPL, 1D"
    parts = [p.strip() for p in stem.split(",")]
    base = parts[0]  # "BATS_AAPL"
    timeframe = parts[1] if len(parts) > 1 else ""

    for pref in FILE_PREFIXES:
        if base.startswith(pref):
            base = base[len(pref) :]
            break

    symbol = base
    return symbol, timeframe


# --------------------------------------------------------------------------------------
# Cleanup: renaming + color mappings (English internal names)
# --------------------------------------------------------------------------------------


def apply_column_renaming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use RENAMING from YAML:
      - create new boolean columns with English names (bull_rot, bear_1, ...)
      - drop original German source columns. [file:76]
    """
    for src, dst in RENAMING.items():
        if src in df.columns:
            df[dst] = (df[src].notna()) & (df[src] != 0)
    df.drop(
        columns=[c for c in RENAMING.keys() if c in df.columns],
        inplace=True,
        errors="ignore",
    )
    return df


def build_color_mappings_english() -> Dict[str, Dict[str, str]]:
    """
    Convert YAML color sections into English logical groups.
    YAML already uses English color words, normalize to lowercase. [file:76]
    """
    english: Dict[str, Dict[str, str]] = {}
    for group_name, rules in COLOR_MAPPINGS.items():
        group_map: Dict[str, str] = {}
        for src_col, color_name in rules.items():
            group_map[src_col] = color_name.lower()
        english[group_name] = group_map
    return english


COLOR_MAPPINGS_EN = build_color_mappings_english()


def apply_color_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each logical group in COLOR_MAPPINGS_EN:
      - create a new column with that group name (trend, welle, wolke, kerze, setter, status, deluxe, ...)
      - if a source column is non-null, assign the configured English color string
      - drop the original source columns. [file:76]
    """
    for target_col, rules in COLOR_MAPPINGS_EN.items():
        if target_col not in df.columns:
            df[target_col] = pd.Series(pd.NA, index=df.index, dtype="object")

        for source_col, color in rules.items():
            if source_col in df.columns:
                mask = df[source_col].notna()
                df.loc[mask, target_col] = color

        df.drop(
            columns=[c for c in rules.keys() if c in df.columns],
            inplace=True,
            errors="ignore",
        )
    return df


# --------------------------------------------------------------------------------------
# Trading range logic (trading_range_2)
# --------------------------------------------------------------------------------------


def trading_range_2(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze trading ranges and annotate potential long trade ranges,
    activation indices, and TP1/2/3 for both normal and swing variants.
    """
    required_cols = {"high", "low"}
    missing = required_cols - set(price_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [
        "Long_Range",
        "Long_aktiv",
        "Long_TP1",
        "Long_TP2",
        "Long_TP3",
        "Long_Range_swing",
        "Long_aktiv_swing",
        "Long_TP1_swing",
        "Long_TP2_swing",
        "Long_TP3_swing",
    ]:
        if col not in price_data.columns:
            price_data[col] = pd.Series(dtype="object")

    for i, row in price_data[price_data["Long_Range"].isna()].iterrows():
        offset = price_data.index.get_loc(i)

        # ----- standard range -----
        future_lows = np.where(price_data.loc[i:].low.values < row.low)[0]
        if len(future_lows) > 0:
            try:
                end_of_range_index = price_data.index[offset + future_lows[0]]
            except Exception as e:
                print(f"Error processing range for index {i}: {e}")
                end_of_range_index = price_data.index.max()
        else:
            end_of_range_index = price_data.index.max()

        trade: Dict[str, Any] = {"Long_Range": end_of_range_index}

        segment = price_data.loc[i:end_of_range_index]
        aktiv = np.where(segment.high.values > row.high)[0]

        if len(aktiv) > 0:
            trade["Long_aktiv"] = segment.index[aktiv][0]

            range_size = row.high - row.low
            for tp_level in [1, 2, 3]:
                tp_value = row.high + tp_level * range_size
                tps = np.where(segment.high.values > tp_value)[0]
                if len(tps) > 0:
                    trade[f"Long_TP{tp_level}"] = segment.index[tps][0]
                else:
                    break

        # ----- swing-based range -----
        swing_low = min(
            row.low,
            row["Wolke Linie Grün"],
            row["Wolke Linie Pink"],
            row["Swingpunkt Long"],
        )
        future_lows = np.where(price_data.loc[i:].low.values < swing_low)[0]
        if len(future_lows) > 0:
            try:
                end_of_range_index = price_data.index[offset + future_lows[0]]
            except Exception as e:
                print(f"Error processing swing range for index {i}: {e}")
                end_of_range_index = price_data.index.max()
        else:
            end_of_range_index = price_data.index.max()

        trade["Long_Range_swing"] = end_of_range_index

        segment = price_data.loc[i:end_of_range_index]
        aktiv = np.where(segment.high.values > row.high)[0]

        if len(aktiv) > 0:
            trade["Long_aktiv_swing"] = segment.index[aktiv][0]

            range_size = row.high - min(
                row.low,
                row["Wolke Linie Grün"],
                row["Wolke Linie Pink"],
                row["Swingpunkt Long"],
            )
            for tp_level in [1, 2, 3]:
                tp_value = row.high + tp_level * range_size
                tps = np.where(segment.high.values > tp_value)[0]
                if len(tps) > 0:
                    trade[f"Long_TP{tp_level}_swing"] = segment.index[tps][0]
                else:
                    break

        for key, value in trade.items():
            price_data.at[i, key] = value

    return price_data


# --------------------------------------------------------------------------------------
# Outcome classification (TP3 / SL / REJECTED)
# --------------------------------------------------------------------------------------


def classify_outcome(row: pd.Series) -> str:
    """
    Classify a single row:
      - 'TP3'      : Long_TP3 is set (3R reached)
      - 'SL'       : activated (Long_aktiv set) but TP3 not reached
      - 'REJECTED' : never activated (Long_aktiv missing)
    """
    if pd.isna(row.get("Long_aktiv")):
        return "REJECTED"
    if not pd.isna(row.get("Long_TP3")):
        return "TP3"
    return "SL"


# --------------------------------------------------------------------------------------
# Signal & color combination logic
# --------------------------------------------------------------------------------------


def signal_base_mask(df: pd.DataFrame, signal_expr: str) -> pd.Series:
    """
    Evaluate YAML signal expressions like:
        'bull_rot == 1 and bull_1 == 1'
    using boolean columns created by RENAMING. [file:76]
    """
    expr = signal_expr.replace("== 1", "== True").replace("!= 1", "== False")
    return df.eval(expr)


def lab_color_short(c: str) -> str:
    """Map English color to short German label, keep 'none' as 'none'."""
    if c == "none":
        return "none"
    mapping = {
        "red": "rot",
        "green": "grün",
        "black": "schwarz",
        "yellow": "gelb",
        "darkred": "dunkelrot",
        "darkgreen": "dunkelgrün",
    }
    return mapping.get(c, c)


def print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"\n{title}: keine Daten.")
        return

    print(f"\n{title}")
    headers = [
        "signal",
        "symbol",
        "timeframe",
        "wolke",
        "welle",
        "trend",
        "setter",
        "TP(3R)",
        "SL(-1R)",
        "Rejected(0R)",
        "level",
    ]
    col_widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(r.get(h, ""))))

    def fmt_row(r: Dict[str, Any]) -> str:
        return " | ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers)

    line_len = sum(col_widths.values()) + 3 * (len(headers) - 1)

    print("-" * line_len)
    print(fmt_row({h: h for h in headers}))
    print("-" * line_len)
    for r in rows:
        print(fmt_row(r))
    print("-" * line_len)


def compute_stats_for_signal(
    df: pd.DataFrame, name: str, expr: str
) -> List[Dict[str, Any]]:
    """
    For a given signal:
      - filter rows by signal expression,
      - classify each as TP3 / SL / REJECTED,
      - build + print 4 tables (wolke; wolke+welle; wolke+welle+trend; wolke+welle+trend+setter),
      - return all rows as list[dict] for DataFrame export.
    """
    base_mask = signal_base_mask(df, expr)
    if base_mask.sum() == 0:
        print(f"\nSignal '{name}': keine Treffer im Datensatz.")
        return []

    df_sig = df[base_mask].copy()
    df_sig["__outcome"] = df_sig.apply(classify_outcome, axis=1)

    for col in ["wolke", "welle", "trend", "setter"]:
        if col not in df_sig.columns:
            df_sig[col] = "none"
        else:
            df_sig[col] = df_sig[col].fillna("none").astype(str).str.lower()

    print(f"\n==================== {name} ====================")
    print(f"Anzahl Signale gesamt: {len(df_sig)}")

    def collect_rows(group_cols: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        # always include symbol + timeframe in the grouping
        gb_cols = ["symbol", "timeframe"] + group_cols
        grouped = df_sig.groupby(gb_cols)

        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = {col: val for col, val in zip(gb_cols, keys)}

            symbol = str(key_map.get("symbol", ""))
            tf = str(key_map.get("timeframe", ""))

            wolke_c = key_map.get("wolke", "none")
            welle_c = key_map.get("welle", "none")
            trend_c = key_map.get("trend", "none")
            setter_c = key_map.get("setter", "none")

            outcomes = group["__outcome"]
            tp3 = (outcomes == "TP3").sum()
            sl = (outcomes == "SL").sum()
            rej = (outcomes == "REJECTED").sum()

            rows.append(
                {
                    "signal": name,
                    "symbol": symbol,  # one symbol per row
                    "timeframe": tf,  # one timeframe per row
                    "wolke": lab_color_short(wolke_c),
                    "welle": lab_color_short(welle_c),
                    "trend": lab_color_short(trend_c),
                    "setter": lab_color_short(setter_c),
                    "TP(3R)": int(tp3),
                    "SL(-1R)": int(sl),
                    "Rejected(0R)": int(rej),
                    "level": "+".join(group_cols),
                }
            )
        return rows

    rows1 = collect_rows(["wolke"])
    print_table("Ebene 1 – pro Wolkenfarbe", rows1)

    rows2 = collect_rows(["wolke", "welle"])
    print_table("Ebene 2 – Wolkenfarbe + Welle", rows2)

    rows3 = collect_rows(["wolke", "welle", "trend"])
    print_table("Ebene 3 – Wolkenfarbe + Welle + Trend", rows3)

    rows4 = collect_rows(["wolke", "welle", "trend", "setter"])
    print_table("Ebene 4 – Wolkenfarbe + Welle + Trend + Setter", rows4)

    return rows1 + rows2 + rows3 + rows4


# --------------------------------------------------------------------------------------
# Load & prepare CSVs (including symbol + timeframe)
# --------------------------------------------------------------------------------------


def load_and_prepare_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    symbol, tf = parse_filename(path)
    df["symbol"] = symbol
    df["timeframe"] = tf
    df["source_file"] = path.name

    for col in ("high", "low"):
        if col not in df.columns:
            raise ValueError(f"{path.name}: missing required column '{col}'")

    df = apply_column_renaming(df)
    df = apply_color_mappings(df)
    df = trading_range_2(df)

    return df


def run_statistics_over_archive() -> pd.DataFrame:
    all_dfs: List[pd.DataFrame] = []

    if not IN_DIR.exists():
        print(f"Input directory '{IN_DIR}' does not exist.")
        return pd.DataFrame()

    for csv_path in sorted(IN_DIR.glob("*.csv")):
        try:
            df = load_and_prepare_csv(csv_path)
            all_dfs.append(df)
            print(f"Loaded and prepared {csv_path.name} with {len(df)} rows.")
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")

    if not all_dfs:
        print("No CSV files processed.")
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)

    stats_rows: List[Dict[str, Any]] = []
    for sig_name, expr in SIGNAL_DEFS.items():
        stats_rows.extend(compute_stats_for_signal(full_df, sig_name, expr))

    stats_df = pd.DataFrame(stats_rows)

    print("\nStatistics DataFrame preview:")
    print(stats_df.head())
    print(f"\nShape: {stats_df.shape}")

    return stats_df


if __name__ == "__main__":
    stats_df = run_statistics_over_archive()
    stats_df.to_csv(BASE_DIR / "import_statistics_output.csv", index=False)
