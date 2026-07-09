import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path.cwd()))

from app.services.backtester.analytics import BacktestAnalytics


def convert_to_serializable(obj):
    """Recursively convert NumPy/Pandas types to standard Python types for JSON."""
    if isinstance(obj, np.int64 | np.int32 | np.int16 | np.int8):
        return int(obj)
    elif isinstance(obj, np.float64 | np.float32 | np.float16):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif hasattr(obj, "__dict__"):
        return convert_to_serializable(obj.__dict__)
    return obj


def compare_dicts(master, current, path=""):
    """Compares two dictionaries and prints differences."""
    deviations = []

    for k, v in master.items():
        curr_val = current.get(k)
        full_path = f"{path}.{k}" if path else k

        if isinstance(v, dict) and isinstance(curr_val, dict):
            deviations.extend(compare_dicts(v, curr_val, full_path))
        elif isinstance(v, int | float) and isinstance(curr_val, int | float):
            diff = abs(v - curr_val)
            # Threshold for reporting (ignore minor float precision noise)
            if diff > 1e-6:
                pct_diff = (diff / abs(v) * 100) if abs(v) > 1e-9 else 0
                deviations.append(
                    {
                        "metric": full_path,
                        "master": v,
                        "current": curr_val,
                        "diff": diff,
                        "pct_diff": pct_diff,
                    }
                )
        elif v != curr_val:
            deviations.append(
                {
                    "metric": full_path,
                    "master": v,
                    "current": curr_val,
                    "diff": "N/A",
                    "pct_diff": 0,
                }
            )
    return deviations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-db", required=True)
    parser.add_argument("--stocks-db", required=True)
    parser.add_argument("--output-dir", default="brain/snapshots")
    parser.add_argument(
        "--compare", action="store_true", help="Compare against existing snapshots"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analytics = BacktestAnalytics(args.backtest_db, args.stocks_db)

    print("Running Analytics...")
    current_metrics = convert_to_serializable(analytics.run_analysis())
    current_strategy_metrics = convert_to_serializable(
        analytics.run_strategy_analysis()
    )
    current_portfolio = convert_to_serializable(analytics.calculate_portfolio_kelly())

    if args.compare:
        print("\n=== DEVIATION ANALYSIS ===\n")

        # Load Master
        try:
            with open(f"{args.output_dir}/global_metrics.json") as f:
                master_metrics = json.load(f)

            deviations = compare_dicts(master_metrics, current_metrics)
            if not deviations:
                print("✅ Global Metrics: NO DEVIATIONS")
            else:
                print("❌ Global Metrics DEVIATIONS:")
                print(pd.DataFrame(deviations).to_string(index=False))

            # Strategy Metrics Comparison
            with open(f"{args.output_dir}/strategy_metrics.json") as f:
                master_strat = json.load(f)

            print("\nComparing Strategy Metrics...")
            for strat, master_m in master_strat.items():
                curr_m = current_strategy_metrics.get(strat, {})
                strat_devs = compare_dicts(master_m, curr_m, path=strat)
                if strat_devs:
                    print(f"❌ {strat} DEVIATIONS:")
                    print(pd.DataFrame(strat_devs).to_string(index=False))
                else:
                    print(f"✅ {strat}: NO DEVIATIONS")

        except Exception as e:
            print(f"Error loading master for comparison: {e}")
    else:
        # Capture Mode
        print("Capturing Snapshots...")
        with open(f"{args.output_dir}/global_metrics.json", "w") as f:
            json.dump(current_metrics, f, indent=4)

        with open(f"{args.output_dir}/strategy_metrics.json", "w") as f:
            json.dump(current_strategy_metrics, f, indent=4)

        with open(f"{args.output_dir}/portfolio_kelly.json", "w") as f:
            json.dump(current_portfolio, f, indent=4)

        print("Capturing Raw Closed Trades...")
        trades_df = analytics.get_all_closed_trades()
        if not trades_df.empty:
            trades_df.to_csv(f"{args.output_dir}/raw_trades.csv", index=False)

        print("Capturing Global Equity Curve...")
        try:
            equity_df = analytics.get_equity_curve()
            if not equity_df.empty:
                equity_df.to_csv(f"{args.output_dir}/equity_curve.csv", index=False)
        except Exception as e:
            print(f"Warning: Could not capture equity curve: {e}")

        print(f"Golden Master snapshots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
