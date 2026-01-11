import logging.config
from pathlib import Path

import pytz
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask

from .config import settings
from .extensions import cache

# NEU: Mapper importieren
from .mapping import mapper
from .routes import main_bp
from .services import BackgroundWorker, CsvImportWorker
from .services.database import SignalDatabase
from .services.market_data import MarketDataWorker
from .services.screener import ScreenerEngine
from .services.strategy_engine import StrategyEngine
from .services.telegram import TelegramBot
from .services.trade_manager import TradeManager


def create_app(config_object=settings):
    app = Flask(__name__)

    # ---------------------------------------------------------
    # 1. Konfiguration & Cache
    # ---------------------------------------------------------
    app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300
    app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
    app.config["SECRET_KEY"] = config_object.env.SECRET_KEY
    app.config["APP_CONFIG"] = config_object

    app.json.compact = False
    app.json.ensure_ascii = False

    cache.init_app(app)

    # ---------------------------------------------------------
    # 2. Logging Setup
    # ---------------------------------------------------------
    log_file_path = config_object.get_log_path()

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "filename": log_file_path,
                "when": "midnight",
                "interval": 1,
                "backupCount": 5,
                "encoding": "utf-8",
                "level": "INFO",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": "INFO", "propagate": True},
            "apscheduler": {"level": "WARNING"},
            "werkzeug": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "yfinance": {"level": "ERROR"},
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)

    class HealthCheckFilter(logging.Filter):
        def filter(self, record):
            return not (
                "/health" in record.getMessage() and " 200 " in record.getMessage()
            )

    logging.getLogger("werkzeug").addFilter(HealthCheckFilter())

    # ---------------------------------------------------------
    # 3. Pfade & DBs initialisieren
    # ---------------------------------------------------------
    db_stocks = config_object.get_db_path("stocks")
    db_signals = config_object.get_db_path("signals")
    db_strategies = config_object.get_db_path("strategies")

    # ---------------------------------------------------------
    # NEU: HIER den Mapper laden und alte Daten bereinigen
    # ---------------------------------------------------------
    mapper.load()

    # Datenbank Bereinigung (BATS Fix)
    try:
        logging.info("Führe Startup-Maintenance durch (BATS Fix)...")
        temp_db = SignalDatabase(Path(db_signals))
        # Nutzt den Mapper, um Exchanges in der DB zu korrigieren
        fixed_count = temp_db.clean_batz_exchanges()
        if fixed_count > 0:
            logging.info(
                f"✅ Exchange-Fix: {fixed_count} Datensätze von 'BATS' auf Real-Exchange korrigiert."
            )
        else:
            logging.info("Exchange-Datenbank ist sauber.")

        # Optional: Auch mal aufräumen (VACUUM)
        temp_db.optimize()

    except Exception as e:
        logging.warning(f"Startup-Maintenance fehlgeschlagen (nicht kritisch): {e}")

    # --- ZENTRALES LADEN DER STRATEGIEN (CROC UPDATE) ---
    yaml_config_path = config_object.get_strategy_path()
    loaded_config = {}  # Wir laden jetzt ein Config-Dict

    if yaml_config_path.exists():
        try:
            with open(yaml_config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Unterstützung für das neue CROC Format & Legacy Fallback
            if isinstance(data, dict):
                if "strategy_ranking" in data:
                    # Das ist das neue Format! Wir übergeben alles.
                    loaded_config = data
                    logging.info(
                        f"CROC Strategie geladen: {len(data.get('strategy_ranking', []))} Regeln."
                    )
                elif "signal" in data:
                    # Legacy Format Support
                    loaded_config = {"strategy_ranking": data["signal"]}
                elif "strategies" in data:
                    # Legacy Format Support
                    loaded_config = {"strategy_ranking": data["strategies"]}
            elif isinstance(data, list):
                # Legacy Format (nur Liste)
                loaded_config = {"strategy_ranking": data}

        except Exception as e:
            logging.error(f"Fehler beim Laden der Strategie-YAML: {e}")
            loaded_config = {}
    else:
        logging.warning(f"Keine Strategie-Datei gefunden unter: {yaml_config_path}")

    # ---------------------------------------------------------
    # 4. Services Initialisieren
    # ---------------------------------------------------------

    # A) Telegram
    tele_conf = config_object.app.telegram
    telegram_service = TelegramBot(
        token=tele_conf.token, chat_id=tele_conf.chat_id, enabled=tele_conf.enabled
    )
    app.extensions["telegram"] = telegram_service

    # B) Market Data Worker
    market_worker = MarketDataWorker(
        db_path=Path(db_stocks),
        run_on_start=False,
    )
    market_worker.start()
    app.extensions["market_worker"] = market_worker

    # C) Webhook Worker
    worker = BackgroundWorker(
        db_path=Path(db_signals),
        batch_size=config_object.app.worker.size,
        timeout=config_object.app.worker.timeout,
    )
    worker.start()
    app.extensions["worker"] = worker

    # D) CSV File Watcher
    csv_worker = CsvImportWorker(
        data_folder=config_object.db_root_path,
        db_path=Path(db_signals),
        check_interval=60,
    )
    csv_worker.start()
    app.extensions["csv_worker"] = csv_worker

    # E) Screener Engine (UPDATED: config statt strategies)
    screener = ScreenerEngine(
        stocks_db_path=Path(db_stocks),
        signals_db_path=Path(db_signals),
        config=loaded_config,
        telegram_bot=telegram_service,
    )
    app.extensions["screener_engine"] = screener

    # F) Trade Manager
    trade_manager = TradeManager(
        db_path=Path(db_signals), telegram_bot=telegram_service
    )
    app.extensions["trade_manager"] = trade_manager

    # G) Strategy Engine
    # Wir extrahieren die Liste für die StrategyEngine, da diese (noch) keine Full-Config erwartet
    strat_list = (
        loaded_config.get("strategy_ranking", [])
        if isinstance(loaded_config, dict)
        else []
    )

    strategy_engine = StrategyEngine(
        signals_db_path=Path(db_signals),
        strategy_db_path=Path(db_strategies),
        telegram_bot=telegram_service,
        strategies=strat_list,
    )
    app.extensions["strategy_engine"] = strategy_engine

    # ---------------------------------------------------------
    # 5. Scheduler Jobs
    # ---------------------------------------------------------
    app_scheduler = BackgroundScheduler()

    app_scheduler.add_job(
        func=trade_manager.check_active_positions,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=15,
            minute=50,
            timezone=pytz.timezone("America/New_York"),
        ),
        id="trade_manager_check",
        replace_existing=True,
    )

    def run_strategy_job():
        with app.app_context():
            logging.info("Starte täglichen Strategie-Check...")
            # Automatische Ausführung am Morgen
            screener.run_all(days=0)
            strategy_engine.run_daily_analysis(lookback_days=1)
            strategy_engine.send_telegram_report()

    app_scheduler.add_job(
        func=run_strategy_job,
        trigger=CronTrigger(hour=8, minute=0),
        id="strategy_morning_run",
        replace_existing=True,
    )

    app_scheduler.start()
    app.extensions["scheduler"] = app_scheduler

    # ---------------------------------------------------------
    # 6. Finalisierung
    # ---------------------------------------------------------
    app.register_blueprint(main_bp)

    if tele_conf.enabled:
        telegram_service.send("🚀 **Croc-Trader System gestartet!**")
        try:
            logging.info("Führe initialen Strategie-Check (30 Tage Rückblick) durch...")
            # Beim Start auch einmalig alles prüfen
            screener.run_all(days=30)
            strategy_engine.run_daily_analysis(lookback_days=30)
            strategy_engine.send_telegram_report()
        except Exception as e:
            logging.error(f"Fehler beim Startup-Check: {e}")

    return app
