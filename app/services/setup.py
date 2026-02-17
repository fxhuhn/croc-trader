import logging
import yaml
import pytz
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database.session import DatabaseSession
from ..database.repositories.trade import TradeRepository
from ..database.repositories.signal import SignalRepository

from ..database.repositories.market_data_provider import MarketDataProvider
from ..services.trade_manager import TradeManager
from ..services.screener import ScreenerEngine
from ..services.telegram import TelegramBot
from ..tools.market_holidays import MarketHolidayChecker

# Tasks importieren
from ..tasks import run_daily_strategy_check, run_market_data_update, run_db_maintenance, run_db_backup

logger = logging.getLogger(__name__)

def register_services(app, config):
    """Initialisiert alle Services und hängt sie an app.extensions."""
    
    db_stocks = Path(config.get_db_path("stocks"))
    db_signals = Path(config.get_db_path("signals"))

    # 1. Telegram Service
    tele_conf = config.app.telegram
    telegram = TelegramBot(token=tele_conf.token, chat_id=tele_conf.chat_id, enabled=tele_conf.enabled)
    app.extensions["telegram"] = telegram

    # 1.5 Holiday Checker
    holidays_path = config.get_path("holidays_yaml")
    holiday_checker = MarketHolidayChecker(holidays_path)
    app.extensions["holiday_checker"] = holiday_checker

    # 1.6 Symbol Filter (Background Init)
    from ..tools.symbol_filter import SymbolFilter
    # Initialize singleton to start background thread/cache loading
    symbol_filter = SymbolFilter() 
    app.extensions["symbol_filter"] = symbol_filter

    # 2. Market Data Infrastructure (Read-Side)
    stocks_session = DatabaseSession(str(db_stocks))
    # market_repo unused here
    
    md_provider = MarketDataProvider(stocks_session) 
    app.extensions["market_data_provider"] = md_provider

    # 3. Signal & Trade Infrastructure (Write-Side)
    signals_session = DatabaseSession(str(db_signals))
    
    # Repos erstellen
    trade_repository = TradeRepository(signals_session)
    signal_repository = SignalRepository(signals_session)
    
    # Init Schemas (Sicherstellen, dass Tabellen existieren)
    trade_repository.init_schema()
    signal_repository.init_schema()

    # 4. Screener Config laden
    yaml_path = config.get_strategy_path()
    loaded_config = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            loaded_config = data if isinstance(data, dict) else {"strategy_ranking": data}
            logging.info("Strategie-Config geladen.")
        except Exception as e:
            logging.error(f"Fehler beim Laden der Strategie-YAML: {e}")

    # 5. Screener Engine (DI: Repos direkt übergeben)
    screener = ScreenerEngine(
        trade_repository=trade_repository,     # <--- WICHTIG: Repo statt Pfad/DB-Wrapper
        signal_repository=signal_repository,   # <--- WICHTIG
        data_provider=md_provider,
        config=loaded_config,
        telegram_bot=telegram
    )
    app.extensions["screener_engine"] = screener

    # 6. Trade Manager
    tm = TradeManager(
        db_path=db_signals,
        stocks_db_path=db_stocks,
        telegram_bot=telegram
    )
    app.extensions["trade_manager"] = tm

def configure_scheduler(app, config):
    """Konfiguriert den Scheduler und die Jobs."""
    scheduler = BackgroundScheduler()
    db_stocks = Path(config.get_db_path("stocks"))
    
    # --- JOB 1: Marktdaten Update (Täglich 17:00 NY Time) ---
    scheduler.add_job(
        func=run_market_data_update,
        args=[db_stocks],
        trigger=CronTrigger(hour=17, minute=0, timezone=pytz.timezone("America/New_York")),
        id="market_data_update",
        replace_existing=True
    )

    # --- JOB 2: DB Maintenance (Sonntags 04:00) ---
    scheduler.add_job(
        func=run_db_maintenance,
        args=[db_stocks],
        trigger=CronTrigger(day_of_week="sun", hour=4, timezone=pytz.timezone("Europe/Berlin")),
        id="db_maintenance",
        replace_existing=True
    )

    # --- JOB 3: Trade Manager (Active Positions / Orders) ---
    tm = app.extensions.get("trade_manager")
    if tm:
        scheduler.add_job(
            func=tm.run_daily_process,
            trigger=CronTrigger(day_of_week="mon-sat", hour=7, minute=0, timezone=pytz.timezone("Europe/Berlin")),
            id="trade_manager_process",
            replace_existing=True
        )

    # --- JOB 4: Strategy Check (Screener) ---
    scheduler.add_job(
        func=run_daily_strategy_check,
        args=[app], 
        trigger=CronTrigger(day_of_week="mon-fri", hour=6, minute=30, timezone=pytz.timezone("Europe/Berlin")),
        id="strategy_check",
        replace_existing=True
    )

    # --- JOB 5: Daily Backup (Signals DB) ---
    db_signals = Path(config.get_db_path("signals"))
    scheduler.add_job(
        func=run_db_backup,
        args=[db_signals],
        trigger=CronTrigger(hour=1, minute=0, timezone=pytz.timezone("Europe/Berlin")),
        id="daily_backup_signals",
        replace_existing=True
    )

    scheduler.start()
    app.extensions["scheduler"] = scheduler
    logger.info("Scheduler gestartet und Jobs geplant.")