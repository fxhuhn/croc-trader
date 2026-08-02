import logging.config

from flask import Flask, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import ConfigManager, settings
from .extensions import cache
from .mapping import mapper

# Import the aggregated routes blueprint from the routes package
from .routes import main_bp

# Services Setup
from .services.setup import configure_scheduler, register_services


def create_app(config_object: ConfigManager = settings) -> Flask:
    """Application factory for the Croc-Trader Flask application.

    Creates and configures the Flask app instance, initializes logging,
    registers blueprints, and starts background services.

    Args:
        config_object: The configuration manager providing all settings.

    Returns:
        A fully configured Flask application instance.
    """
    app = Flask(__name__, static_url_path="", static_folder="static")

    # Enable ProxyFix to trust standard headers (X-Forwarded-For, etc.)
    # Configured for 1 trusted upstream proxy (e.g. Synology Nginx proxy).
    app.wsgi_app = ProxyFix(
        app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=0, x_prefix=0
    )

    # 1. Config & Cache
    debug_mode = config_object.app.webserver.debug
    app.config["CACHE_TYPE"] = "NullCache" if debug_mode else "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
    app.config["SECRET_KEY"] = config_object.env.SECRET_KEY
    app.config["APP_CONFIG"] = config_object

    app.json.compact = False
    app.json.ensure_ascii = False

    cache.init_app(app)

    # Register local timezone conversion filter
    @app.template_filter("to_local_tz")
    def to_local_tz(utc_str: str | None) -> str:
        """Converts UTC date/time string from database to local Europe/Berlin timezone.

        Handles both naive (assumed UTC) and timezone-aware ISO strings
        such as '2026-06-04 11:30:01+00:00'.
        """
        if not utc_str:
            return "-"
        try:
            import datetime

            import pytz

            dt = datetime.datetime.fromisoformat(str(utc_str))
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            berlin_tz = pytz.timezone("Europe/Berlin")
            dt_local = dt.astimezone(berlin_tz)
            return dt_local.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(utc_str)

    # 2. Logging Setup
    log_file_path = config_object.get_log_path()
    log_level = config_object.app.logging.level.upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": log_file_path,
                "when": "midnight",
                "interval": 1,
                "backupCount": 5,
                "encoding": "utf-8",
                "level": log_level,
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": log_level,
                "propagate": True,
            },
            "apscheduler": {"level": "WARNING"},
            "werkzeug": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "yfinance": {"level": "ERROR"},
        },
    }
    logging.config.dictConfig(logging_config)

    # 3. Load exchange mapper
    mapper.load()

    # 4. Services
    register_services(app, config_object)

    # 5. Register blueprints
    # Only the main aggregator is registered here
    app.register_blueprint(main_bp)

    # 6. Static file root routes
    @app.route("/favicon.ico")
    def favicon():
        return send_from_directory(
            app.static_folder, "favicon.ico", mimetype="image/vnd.microsoft.icon"
        )

    @app.route("/apple-touch-icon-precomposed.png")
    def apple_touch_icon_precomposed():
        return send_from_directory(
            app.static_folder, "apple-touch-icon-precomposed.png", mimetype="image/png"
        )

    @app.route("/apple-touch-icon.png")
    @app.route("/apple-touch-icon-120x120.png")
    @app.route("/apple-touch-icon-120x120-precomposed.png")
    @app.route("/apple-touch-icon-152x152.png")
    @app.route("/apple-touch-icon-152x152-precomposed.png")
    @app.route("/apple-touch-icon-180x180.png")
    @app.route("/apple-touch-icon-180x180-precomposed.png")
    def apple_touch_icon():
        return send_from_directory(
            app.static_folder, "apple-touch-icon.png", mimetype="image/png"
        )

    @app.route("/robots.txt")
    def robots_txt():
        return send_from_directory(app.static_folder, "robots.txt")

    # 7. Scheduler & Startup Tasks (configured after all routes are registered)
    configure_scheduler(app, config_object)

    logging.info("🚀 Croc-Trader App initialized (Aggregated Routes).")
    return app
