import json
from datetime import datetime
from pathlib import Path

from flask import Flask

from app.config import Config
from app.container import build_container
from app.routes import api as api_routes
from app.routes import ui as ui_routes


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates")
    cfg = config or Config()

    # NEW: load symbol → exchange map
    symbols_path = Path(__file__).parent.with_name("unique_symbols.json")
    if symbols_path.exists():
        with symbols_path.open("r", encoding="utf-8") as f:
            app.symbol_markets = json.load(f)
    else:
        app.symbol_markets = {}

    # attach container (simple DI)
    app.container = build_container(cfg.database.path)

    # register blueprints
    app.register_blueprint(ui_routes.bp)
    app.register_blueprint(api_routes.bp)

    @app.template_filter("datetimeformat")
    def datetimeformat_filter(value, fmt: str = "%Y-%m-%d %H:%M"):
        if not value:
            return ""
        try:
            if isinstance(value, str):
                value = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(value)
            else:
                dt = value
            return dt.strftime(fmt)
        except Exception:
            return value

    return app
