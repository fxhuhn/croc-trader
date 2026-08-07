from pathlib import Path

import yaml

from app.config import settings


def test_inspect_rules() -> None:
    path = settings.get_path("ranking_yaml")
    print("\n--- INSIDE TEST ---")
    print("Resolved path for ranking_yaml:", path)
    print("Path exists:", path.exists())
    if path.exists():
        with open(path, encoding="utf-8") as f:
            content = f.read()
            data = yaml.safe_load(content)
            print("Parsed yaml type:", type(data))
            if isinstance(data, list):
                print("Yaml list length:", len(data))
                for idx, rule in enumerate(data):
                    print(
                        f"Rule {idx}: {rule.get('Signal')} - Exit: {rule.get('Exit')}"
                    )
            elif isinstance(data, dict):
                print("Yaml dict keys:", list(data.keys()))
                rules = data.get("ranking_2026", [])
                print("rules length:", len(rules))
                for idx, rule in enumerate(rules):
                    print(
                        f"Rule {idx}: {rule.get('Signal')} - Exit: {rule.get('Exit')}"
                    )
            else:
                print("Yaml content length:", len(content))
    print("-------------------")


def test_no_scratch_or_db_files_in_root() -> None:
    """Governance Guard: Ensures no .db, .db-wal, or .db-shm files exist in repository root."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    db_files = list(root_dir.glob("*.db*"))
    invalid_root_files = [f.name for f in db_files if f.name != "trading.db"]
    assert not invalid_root_files, (
        f"Forbidden database/WAL files found in repository root: {invalid_root_files}. "
        "Databases must be stored in data/ or temporary test directories."
    )
