"""Database Schema Initializer for Croc-Trader.

Drops the 'trades' and 'trade_logs' tables in the active signals database (signals.db)
and recreates them from scratch using the trade repository schema parameters. Used for resetting
development databases to a clean slate.

Usage:
    python script/reset_trade_schema.py

Side Effects:
    DESTRUCTIVE. Drops active trade tables in data/signals.db, wiping all trade histories.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession


def main():
    print("WARNING: This will delete all existing trades in the signals database.")
    db_path = settings.get_path("signals")
    session = DatabaseSession(str(db_path))
    repo = TradeRepository(session)

    # Hier passiert die Magie: DROP TABLE -> CREATE TABLE
    repo.init_schema()
    print("✅ Database schema reset successfully.")


if __name__ == "__main__":
    main()
