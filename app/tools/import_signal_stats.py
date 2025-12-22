import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.tools.signal_stats import run_statistics_over_archive  # your existing function

from app import create_app
from app.config import Config


def import_stats():
    app = create_app(Config())
    with app.app_context():
        stats_df = run_statistics_over_archive()
        rows = stats_df.to_dict(orient="records")
        app.container.repo.replace_signal_statistics(rows)
        print(f"Imported {len(rows)} statistic rows into signal_statistic.")


if __name__ == "__main__":
    import_stats()
