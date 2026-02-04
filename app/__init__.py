import logging.config
from flask import Flask

from .config import settings
from .extensions import cache
from .mapping import mapper

# WICHTIG: Wir importieren jetzt den Aggregator (main_bp) aus dem routes-Package
from .routes import main_bp 

# Services Setup
from .services.setup import register_services, configure_scheduler

def create_app(config_object=settings):
    app = Flask(__name__)

    # 1. Config & Cache
    app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
    app.config["SECRET_KEY"] = config_object.env.SECRET_KEY
    app.config["APP_CONFIG"] = config_object
    
    app.json.compact = False
    app.json.ensure_ascii = False
    
    cache.init_app(app)

    # 2. Logging Setup
    log_file_path = config_object.get_log_path()
    log_level = config_object.app.logging.level.upper()
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "datefmt": "%H:%M:%S"}},
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": log_level, "formatter": "standard"},
            "file": {"class": "logging.handlers.TimedRotatingFileHandler", "filename": log_file_path, "when": "midnight", "interval": 1, "backupCount": 5, "encoding": "utf-8", "level": log_level, "formatter": "standard"},
        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": log_level, "propagate": True},
            "apscheduler": {"level": "WARNING"},
            "werkzeug": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "yfinance": {"level": "ERROR"},
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)

    # 3. Mapper laden
    mapper.load()

    # 4. Services & Scheduler
    register_services(app, config_object)
    configure_scheduler(app, config_object)

    # 5. Blueprints registrieren
    # Hier wird nur noch der Haupt-Aggregator registriert
    app.register_blueprint(main_bp)

    logging.info("🚀 Croc-Trader App initialized (Aggregated Routes).")
    return app