"""SQLite View Creator for DipBuyer Strategy Diagnostics.

Creates or replaces the SQL view 'view_screener_debug' in the signals database
to extract and flatten JSON context values (e.g. entry, close, volume, score, atr, atr_pct)
associated with DipBuyer signals. This allows developers to query strategy metrics directly
via SQL and verify the correctness of the database state.

Usage:
    python script/create_debug_view.py

Side Effects:
    Modifies the signals database schema by adding or replacing 'view_screener_debug'.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Pfad Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402


def main():
    print("--- Erstelle Debug View ---")

    db_path = settings.get_path("signals")
    print(f"Datenbank: {db_path}")

    conn = sqlite3.connect(db_path)

    # SQL für den View (Mehrere Statements getrennt durch Semikolon)
    # WICHTIG: Wir holen jetzt auch 'close' und 'volume' aus dem JSON
    sql_script = """
    DROP VIEW IF EXISTS view_screener_debug;

    CREATE VIEW view_screener_debug AS
    SELECT
        id,
        symbol,
        round(entry_price, 2) as entry,

        -- Vergleichswerte aus dem Context
        round(json_extract(signal_context, '$.close'), 2) as close,
        json_extract(signal_context, '$.volume') as volume,
        round(json_extract(signal_context, '$.setup_score'), 2) as score,
        round(json_extract(signal_context, '$.atr5'), 2) as atr,
        round(json_extract(signal_context, '$.atr_r3'), 2) as atr_pct,

        -- Metadaten
        json_extract(signal_context, '$.date') as screener_date,
        json_extract(signal_context, '$.indices') as indices,
        status,
        created_at
    FROM trades
    WHERE strategy = 'DipBuyer';
    """

    try:
        # KORREKTUR: executescript statt execute für mehrere Statements
        conn.executescript(sql_script)
        print("✅ View 'view_screener_debug' erfolgreich erstellt.")

        # Test-Abfrage für den 22.01.2026
        print("\n--- Auszug Daten 22.01.2026 (Top 20 Symbole) ---")

        # Wir filtern hier exemplarisch, um zu sehen, ob deine Referenzwerte dabei sind
        query = """
        SELECT symbol, entry, close, score, atr, indices
        FROM view_screener_debug
        WHERE screener_date = '2026-01-22'
        ORDER BY symbol ASC
        LIMIT 20
        """

        df = pd.read_sql(query, conn)

        if not df.empty:
            # Pandas Option für breitere Anzeige, damit man die Indices sieht
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            print(df.to_string(index=False))
            print(f"\nGesamtanzahl Treffer im View: {len(df)}")

            # Zusatz-Check: Sind SHOP und AVGO dabei?
            print("\n--- Check auf Referenz-Symbole (SHOP, AVGO) ---")
            check_df = pd.read_sql(
                "SELECT * FROM view_screener_debug WHERE symbol IN ('SHOP', 'AVGO') AND screener_date = '2026-01-22'",
                conn,
            )
            if not check_df.empty:
                print(check_df.to_string(index=False))
            else:
                print("⚠️ SHOP und AVGO nicht gefunden! Prüfe Filter-Logik.")

        else:
            print("Keine Daten für 2026-01-22 gefunden.")

    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
