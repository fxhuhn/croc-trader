import pandas as pd
import yaml

from app.backtest.backtest_core import (
    BacktestRepository,  # Adjust import path as needed
)
from app.config import settings

RESULTS_DIR = settings.backtest.report_path


class BacktestViewer:
    def __init__(self):
        # Initialize repo pointing to the backtest database
        self.repo = BacktestRepository(str(settings.database.backtest_path))

    def list_strategies(self):
        """Scans the results folder for metrics files."""
        strategies = []
        if not RESULTS_DIR.exists():
            return []

        for f in RESULTS_DIR.glob("*_metrics.yaml"):
            # Filename format: {Strategy_Name}_metrics.yaml
            # We assume the part before '_metrics.yaml' is the ID/Prefix
            file_prefix = f.name.replace("_metrics.yaml", "")

            with open(f, "r") as file:
                data = yaml.safe_load(file)

            strategies.append(
                {
                    "id": file_prefix,
                    "name": data.get("strategy_name", file_prefix),
                    "cagr": data["performance"].get("cagr_pct"),
                    "drawdown": data["performance"].get("max_drawdown_pct"),
                    "sharpe": data["performance"].get("sharpe_ratio"),
                    "trades": data["trades"].get("count"),
                }
            )
        return strategies

    def get_details(self, strategy_id):
        """Loads full details for a specific strategy."""
        metrics_path = RESULTS_DIR / f"{strategy_id}_metrics.yaml"
        monthly_path = RESULTS_DIR / f"{strategy_id}_monthly_returns.csv"

        if not metrics_path.exists():
            return None

        # 1. Metrics
        with open(metrics_path, "r") as f:
            metrics = yaml.safe_load(f)

        # 2. Monthly Returns
        monthly_html = None
        if monthly_path.exists():
            df = pd.read_csv(monthly_path)
            # Clean up for display
            df = df.fillna("-")
            # Convert to list of dicts or keep as HTML table
            monthly_html = df.to_html(
                classes="min-w-full text-sm text-left text-gray-400",
                index=False,
                border=0,
            )
            # Remove default pandas styles to let Tailwind take over
            monthly_html = monthly_html.replace('border="1"', "").replace(
                'style="text-align: right;"', ""
            )

        return {
            "metrics": metrics,
            "monthly_table": monthly_html,
            "chart_url": f"/strategies/image/{strategy_id}_chart.png",
        }

    def get_trades(self, strategy_name_in_db):
        """Fetches trade list from SQLite."""
        with self.repo._get_connection() as conn:
            # We filter by strategy_name column we added earlier
            # Note: The strategy name in DB must match what's passed here.
            # Ideally, store the exact 'strategy_name' string from the YAML in the DB.
            trades = pd.read_sql(
                "SELECT * FROM backtest_trades WHERE strategy_name = ? ORDER BY entry_date DESC",
                conn,
                params=(strategy_name_in_db,),
            )
        return trades.to_dict(orient="records")
