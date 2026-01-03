import logging
from logging.handlers import TimedRotatingFileHandler  # <--- WICHTIG: Neuer Import
from pathlib import Path

from flask import Flask

from .config import settings
from .routes import main_bp
from .services import BackgroundWorker, CsvImportWorker
from .services.market_data import MarketDataWorker


def create_app(config_object=settings):
    app = Flask(__name__)

    # 1. Konfiguration laden
    app.config["SECRET_KEY"] = config_object.env.SECRET_KEY
    app.json.compact = False
    app.config["APP_CONFIG"] = config_object

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

    # 4. Blueprints registrieren
    app.register_blueprint(main_bp)

    return app
