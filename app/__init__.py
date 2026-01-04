import logging
from logging.handlers import TimedRotatingFileHandler  # <--- WICHTIG: Neuer Import
from pathlib import Path

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from flask import Flask

from .config import settings
from .extensions import cache
from .routes import main_bp
from .services import BackgroundWorker, CsvImportWorker
from .services.market_data import MarketDataWorker
from .services.screener import ScreenerEngine
from .services.telegram import TelegramBot
from .services.trade_manager import TradeManager


def create_app(config_object=settings):
    app = Flask(__name__)

    # Cache konfigurieren (SimpleCache nutzt den RAM, reicht für Docker)
    app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = 300  # Standard: 5 Minuten

    # 1. Konfiguration laden
    app.config["SECRET_KEY"] = config_object.env.SECRET_KEY
    app.config["APP_CONFIG"] = config_object

    app.json.compact = False
    app.json.ensure_ascii = False

    # Cache initialisieren
    cache.init_app(app)

    # 2. Logging Setup (MIT ROTATION)
    log_file_path = config_object.get_log_path()

    # Handler definieren: Täglich rotieren, 5 Backups behalten
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when="midnight",  # Rotiert um Mitternacht
        interval=1,  # Alle 1 Tag
        backupCount=5,  # Behält die letzten 5 Dateien (löscht ältere automatisch)
        encoding="utf-8",  # Umlaute-Sicherheit
    )

    # Formatierung für den File-Handler setzen
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Basis-Logging konfigurieren
    logging.basicConfig(
        level=getattr(logging, config_object.app.logging.level.upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(),  # Ausgabe in die Konsole (für Docker Logs wichtig)
            file_handler,  # Ausgabe in die Datei (mit Rotation)
        ],
        force=True,
    )

    # 3. Services Initialisieren
    db_path = config_object.get_db_path("signals")
    db_stocks = config_object.get_db_path("stocks")
    db_signals = config_object.get_db_path("signals")

    tele_conf = config_object.app.telegram
    telegram_service = TelegramBot(
        token=tele_conf.token, chat_id=tele_conf.chat_id, enabled=tele_conf.enabled
    )

    # A) Market Data Worker
    market_worker = MarketDataWorker(
        db_path=Path(db_stocks),
        run_on_start=False,
    )
    market_worker.start()
    app.extensions["market_worker"] = market_worker

    worker = BackgroundWorker(
        db_path=Path(db_path),
        batch_size=config_object.app.worker.size,
        timeout=config_object.app.worker.timeout,
    )
    worker.start()
    app.extensions["worker"] = worker

    # B) NEU: Der CSV File Watcher
    # Wir nutzen den base_folder aus der Config (normalerweise "data")
    data_folder = config_object.db_root_path

    csv_worker = CsvImportWorker(
        data_folder=data_folder,
        db_path=Path(db_path),
        check_interval=60,  # Prüft alle 60 Sekunden
    )
    csv_worker.start()
    app.extensions["csv_worker"] = csv_worker

    screener = ScreenerEngine(
        stocks_db_path=Path(db_stocks),
        signals_db_path=Path(db_signals),
        telegram_bot=telegram_service,
    )
    app.extensions["screener_engine"] = screener

    # Trade Manager Initialisieren
    trade_manager = TradeManager(
        db_path=Path(db_signals), telegram_bot=telegram_service
    )
    app.extensions["trade_manager"] = trade_manager

    # 2. Scheduler für Trade Manager (15:50 NY Zeit)
    # Wir hängen uns an den existierenden Scheduler vom MarketWorker oder starten einen neuen.
    # Da wir in __init__ sind, ist es am einfachsten, den Scheduler hier global für die App zu nutzen
    # oder einen dedizierten "SystemScheduler" zu haben.

    # Einfachste Lösung: Wir nutzen den Scheduler vom MarketDataWorker NICHT,
    # sondern erstellen einen zentralen Scheduler hier (oder nutzen den vom MarketWorker mit).
    # Da der MarketWorker in einem eigenen Thread läuft, ist Zugriff von außen schwer.

    # Wir erstellen einen kleinen App-Scheduler:
    app_scheduler = BackgroundScheduler()

    # Job: Trade Manager Check (Mo-Fr)
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

    app_scheduler.start()
    app.extensions["scheduler"] = app_scheduler

    # 4. Blueprints registrieren
    app.register_blueprint(main_bp)

    # Nachricht beim Start senden (Optional, gut zum Testen)
    if tele_conf.enabled:
        telegram_service.send("🚀 **Croc-Trader System gestartet!**")

    # In Extensions speichern für globalen Zugriff
    app.extensions["telegram"] = telegram_service

    return app
