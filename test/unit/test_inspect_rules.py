import yaml
from app.config import settings


def test_inspect_rules():
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
