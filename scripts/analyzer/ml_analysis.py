"""
Final Strategy Backtester - Phase 3 (Harmonized Strategies)

Fokus: Finden von Signalkombinationen, die logisch zusammenpassen
(z.B. Reversal + Pullback Indikator) statt sich zu widersprechen.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Importiere Projekt-Module
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from constants import ColumnNames
    from trading_analysis import (
        DataProcessor,
        TradingRangeAnalyzer,
        load_config_with_env_override,
    )
except ImportError as e:
    sys.exit(
        f"Fehler beim Import: {e}. Bitte sicherstellen, dass constants.py und trading_analysis.py verfügbar sind."
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - BACKTEST - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    name: str
    total_trades: int = 0
    wins: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0


def apply_strategies(df: pd.DataFrame, results: dict[str, StrategyResult]):
    if "bull_rot" not in df.columns:
        return

    # === BASIS SIGNAL ===
    base_sig = df["bull_rot"] == 1
    if not base_sig.any():
        return

    # === DATEN VORBEREITEN ===
    # Strings normalisieren (Leerzeichen weg, kleinschreiben) für sauberen Vergleich
    wolke = (
        df["wolke"].astype(str).str.strip().str.lower()
        if "wolke" in df.columns
        else pd.Series("", index=df.index)
    )
    deluxe = (
        df["deluxe"].astype(str).str.strip().str.lower()
        if "deluxe" in df.columns
        else pd.Series("", index=df.index)
    )
    rsi = df["RSI"] if "RSI" in df.columns else pd.Series(0, index=df.index)

    # Pink Line Distance (Support Check)
    # Wir prüfen verschiedene Schreibweisen, da YAML manchmal variiert
    pink_col = None
    possible_names = ["Wolke Linie Pink", "wolke linie pink", "Wolke_Linie_Pink"]
    for p in possible_names:
        if p in df.columns:
            pink_col = p
            break

    if pink_col:
        # Ist der Kurs ÜBER der Pinken Linie? (Support gehalten)
        above_pink = df["close"] > df[pink_col]
    else:
        above_pink = pd.Series(False, index=df.index)

    # Ziel (Win)
    is_win = df[ColumnNames.LONG_TP] == 2

    # =========================================================
    # STRATEGIE 1: BASELINE
    # =========================================================
    _update_result(results["1. Baseline"], base_sig, is_win)

    # =========================================================
    # STRATEGIE 2: GREEN CLOUD (Der stabile Filter)
    # =========================================================
    mask_cloud = base_sig & (wolke == "grün")
    _update_result(results["2. Green Cloud"], mask_cloud, is_win)

    # =========================================================
    # STRATEGIE 3: PULLBACK KING (Harmonisch)
    # These: bull_rot passt zu deluxe 'dunkelrot' (Pullback).
    # Bedingung: Wolke muss Grün sein (Aufwärtstrend).
    # =========================================================
    mask_pullback = base_sig & (wolke == "grün") & (deluxe == "dunkelrot")
    _update_result(results["3. Pullback King (Dunkelrot)"], mask_pullback, is_win)

    # =========================================================
    # STRATEGIE 4: STABILIZER (Schwarz)
    # These: Deluxe 'schwarz' bedeutet Neutralisierung/Stabilisierung.
    # =========================================================
    mask_stabilizer = base_sig & (wolke == "grün") & (deluxe == "schwarz")
    _update_result(results["4. Stabilizer (Schwarz)"], mask_stabilizer, is_win)

    # =========================================================
    # STRATEGIE 5: PINK SUPPORT DEFENSE
    # Wir kaufen nur, wenn die Pinke Linie (Cloud Support) hält.
    # ML zeigte: Nähe zur Pinken Linie ist gut. Unter ihr ist schlecht.
    # =========================================================
    mask_pink = (
        base_sig
        & (wolke == "grün")
        & above_pink  # Kurs muss über Pinker Linie sein
        & (rsi > 40)  # Kein Crash
    )
    _update_result(results["5. Pink Support Hold"], mask_pink, is_win)

    # =========================================================
    # STRATEGIE 6: THE SNIPER (Kombination)
    # Pullback oder Stabilizer, aber strikt über Pinker Linie
    # =========================================================
    mask_sniper = (
        base_sig
        & (wolke == "grün")
        & ((deluxe == "dunkelrot") | (deluxe == "schwarz"))
        & above_pink
    )
    _update_result(results["6. Sniper Entry"], mask_sniper, is_win)


def _update_result(res: StrategyResult, mask: pd.Series, win_col: pd.Series):
    trades = mask.sum()
    if trades > 0:
        res.total_trades += int(trades)
        wins = (mask & win_col).sum()
        res.wins += int(wins)


def load_data_and_test(config):
    processor = DataProcessor(config)
    range_analyzer = TradingRangeAnalyzer()

    # Robuste Dateisuche
    output_dir_path = Path(config.output_dir)
    input_pattern_str = Path(config.input_pattern).name
    possible_input_dirs = [
        output_dir_path.parent / "in",
        Path("./in"),
        Path("../in"),
        Path("in"),
    ]

    files = []
    found_dir = None
    for d in possible_input_dirs:
        if d.exists():
            found = list(d.glob(input_pattern_str))
            found = [
                f
                for f in found
                if not f.name.startswith(("ml_", "statistik_", "feature_"))
            ]
            if found:
                files = found
                found_dir = d
                break

    if not files:
        logger.error("Keine Dateien gefunden.")
        return

    logger.info(f"Starte Phase 3 Backtest mit {len(files)} Dateien aus {found_dir}...")

    results = {
        "1. Baseline": StrategyResult("Nur bull_rot"),
        "2. Green Cloud": StrategyResult("Wolke Grün"),
        "3. Pullback King (Dunkelrot)": StrategyResult("Wolke Grün + Deluxe Dunkelrot"),
        "4. Stabilizer (Schwarz)": StrategyResult("Wolke Grün + Deluxe Schwarz"),
        "5. Pink Support Hold": StrategyResult("Wolke Grün + Über Pinker Linie"),
        "6. Sniper Entry": StrategyResult("Grün + (Dunkelrot/Schwarz) + >Pink"),
    }

    for i, fp in enumerate(files):
        df = processor.load_csv(fp)
        if df is None or df.empty:
            continue

        df = processor.merge_signals(df)
        df = range_analyzer.calculate_ranges(df)
        df = range_analyzer.create_tp_columns(
            df, prefix="Long", use_swing=config.use_swing_range
        )

        apply_strategies(df, results)

        if (i + 1) % 100 == 0:
            print(f"Fortschritt: {i + 1}/{len(files)}...", end="\r")

    print("\n" + "=" * 100)
    print(
        f"{'STRATEGIE NAME':<30} | {'BESCHREIBUNG':<35} | {'TRADES':<10} | {'WIN-RATE':<10}"
    )
    print("-" * 100)

    for key, res in results.items():
        desc = res.name
        print(f"{key:<30} | {desc:<35} | {res.total_trades:<10} | {res.win_rate:6.2f}%")
    print("=" * 100)


def main():
    try:
        config = load_config_with_env_override("./scripts/signal.yaml")
    except Exception:
        config = load_config_with_env_override("signal.yaml")
    load_data_and_test(config)


if __name__ == "__main__":
    main()
