# tools/analyzer/optimizer.py
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --- PFAD FIX ---
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

DEFAULT_CONFIG_PATH = current_dir / "config" / "strategies.yaml"

try:
    import yaml
    from src.analyzer import TradingAnalyzer
    from src.processor import DataProcessor
except ImportError as e:
    sys.exit(f"Import Fehler: {e}")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Optimizer")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_strategies():
    """
    Hier definieren wir den SUCHRAUM.
    Das Skript baut daraus alle Kombinationen.
    """
    strategies = []

    # 1. BASIS SIGNALE (Die wir testen wollen)
    base_signals = [
        ("OrangeLolly", "bull_orange == 1 and bull_1 == 1"),
        ("RedLolly", "bull_rot == 1 and bull_1 == 1"),
        # ("TripleLolly", "bull_rot == 1 and bull_orange == 1 and bull_1 == 1") # Optional
    ]

    # 2. RSI FILTER (Bereiche)
    rsi_filters = [
        ("NoRSI", ""),  # Kein Filter
        ("RSI_40_50", "and RSI >= 40 and RSI <= 50"),
        ("RSI_40_45", "and RSI >= 40 and RSI <= 45"),  # Der Sniper Bereich
        ("RSI_Low", "and RSI < 40"),
    ]

    # 3. EMA FILTER
    ema_filters = [
        ("NoEMA", ""),
        ("UnderEMA", "and close < EMA"),
        ("DeepDip", "and close < (EMA * 0.99)"),  # 1% unter EMA
    ]

    # 4. ZUSATZ INDIKATOREN (Trend, Welle, Setter, Kerze)
    # Wir testen jeden Indikator einzeln dazu
    indicators = [
        ("Raw", ""),  # Nichts dazu
        ("Trend_Grün", "and trend == 'grün'"),
        ("Trend_Schwarz", "and trend == 'schwarz'"),
        ("Trend_Rot", "and trend == 'rot'"),
        ("Setter_Grün", "and setter == 'grün'"),
        ("Setter_Schwarz", "and setter == 'schwarz'"),
        ("Setter_Rot", "and setter == 'rot'"),
        ("Welle_Grün", "and welle == 'grün'"),
        ("Kerze_Grün", "and kerze == 'grün'"),
        ("Kerze_Rot", "and kerze == 'rot'"),
        ("Kerze_Schwarz", "and kerze == 'schwarz'"),
        ("Status_Rot", "and status == 'rot'"),
        ("Status_Gelb", "and status == 'gelb'"),
        ("Status_Grün", "and status == 'grün'"),
    ]

    # --- KOMBINATORIK ---
    # Wir iterieren durch alle Listen und bauen Strategien
    for sig_name, sig_logic in base_signals:
        for rsi_name, rsi_logic in rsi_filters:
            for ema_name, ema_logic in ema_filters:
                for ind_name, ind_logic in indicators:
                    # Name bauen
                    full_name = f"{sig_name} | {rsi_name} | {ema_name} | {ind_name}"
                    # Logik bauen
                    full_logic = (
                        f"{sig_logic} {rsi_logic} {ema_logic} {ind_logic}".strip()
                    )

                    strategies.append({"name": full_name, "logic": full_logic})

    return strategies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--min-trades", type=int, default=500, help="Mindestanzahl Trades für Top-Liste"
    )
    args = parser.parse_args()

    # Config laden
    config = load_config(args.config)

    # Automatische Strategien generieren
    auto_strategies = generate_strategies()
    print("==================================================")
    print("AUTO-OPTIMIZER GESTARTET")
    print(f"Generierte Kombinationen: {len(auto_strategies)}")
    print("==================================================")

    # Config temporär überschreiben
    config["strategies"] = auto_strategies

    # Engine starten
    processor = DataProcessor(config)
    analyzer = TradingAnalyzer(config)

    # Dateien suchen
    project_root = current_dir.parent.parent
    in_dir = project_root / "in"
    if not in_dir.exists():
        in_dir = Path("in")
    files = list(in_dir.glob("*.csv"))
    files = [f for f in files if not f.name.startswith("ml_")]

    if not files:
        print("Keine Dateien gefunden.")
        return

    # Backtest Loop
    results = {}

    # Performance-Optimierung: Wir laden Files nur einmal und testen alle Strategien
    for fp in tqdm(files, desc="Optimiere", unit="file"):
        df = processor.load_and_process(fp)
        if df is None:
            continue

        df = analyzer.calculate_ranges_and_tp(df)
        analyzer.run_backtest(df, results)

    # --- AUSWERTUNG & RANKING ---
    final_results = []
    for name, data in results.items():
        trades = data["trades"]
        if trades < args.min_trades:
            continue  # Rauschen filtern

        win_rate = data["wins"] / trades * 100
        final_results.append({"Strategie": name, "Trades": trades, "WinRate": win_rate})

    # In DataFrame umwandeln und sortieren
    res_df = pd.DataFrame(final_results)

    if res_df.empty:
        print("Keine Strategien mit genügend Trades gefunden.")
        return

    # Top 20 nach Win-Rate
    top_20 = res_df.sort_values(by="WinRate", ascending=False).head(20)

    print("\n" + "=" * 100)
    print(f"TOP 20 KOMBINATIONEN (Min. {args.min_trades} Trades)")
    print("=" * 100)
    print(top_20.to_string(index=False))
    print("=" * 100)

    # Export
    csv_path = "optimizer_results.csv"
    res_df.sort_values(by="WinRate", ascending=False).to_csv(csv_path, index=False)
    print(f"\nAlle Ergebnisse gespeichert in: {csv_path}")


if __name__ == "__main__":
    main()
