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

# Strategies
from .screener.strategies.croc_setup import CrocSetupStrategy
from .screener.strategies.dip_buyer import DipBuyerStrategy
from .screener.strategies.turnover_timing import TurnoverTimingStrategy
from .screener.strategies.two_percent_strategy import TwoPercentStrategy
from .screener.strategies.ndx_momentum import NDXMomentumScreener

# Tasks importieren
from ..tasks import (
    run_daily_strategy_check,
    run_market_data_update,
    run_db_maintenance,
    run_db_backup,
    run_cache_prewarm,
)

logger = logging.getLogger(__name__)


def register_services(app, config):
    """Initialisiert alle Services und hängt sie an app.extensions."""

    db_stocks = Path(config.get_db_path("stocks"))
    db_signals = Path(config.get_db_path("signals"))

    # 1. Telegram Service
    tele_conf = config.app.telegram
    telegram = TelegramBot(
        token=tele_conf.token, chat_id=tele_conf.chat_id, enabled=tele_conf.enabled
    )
    app.extensions["telegram"] = telegram

    # 1.5 Holiday Checker
    holidays_path = config.get_path("holidays_yaml")
    holiday_checker = MarketHolidayChecker(holidays_path)
    app.extensions["holiday_checker"] = holiday_checker

    # 1.6 Symbol Filter (Background Init)
    from ..tools.symbol_filter import SymbolFilter
    from ..tools.symbol_exchange import SymbolExchange

    # Initialize singletons to start background thread/cache loading
    symbol_filter = SymbolFilter()
    app.extensions["symbol_filter"] = symbol_filter

    symbol_exchange = SymbolExchange()
    app.extensions["symbol_exchange"] = symbol_exchange

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
            loaded_config = (
                data if isinstance(data, dict) else {"strategy_ranking": data}
            )
            logging.info("Strategie-Config geladen.")
        except Exception as e:
            logging.error(f"Fehler beim Laden der Strategie-YAML: {e}")

    # 4.5 Ranking System Verification
    ranking_yaml_path = config.get_path("ranking_yaml")
    if ranking_yaml_path.exists():
        try:
            with open(ranking_yaml_path, encoding="utf-8") as f:
                ranking_data = yaml.safe_load(f) or {}

            # Check DB availability for multiple attributes
            db_attributes = signal_repository.get_unique_signal_attributes()
            check_keys = [
                "Signal",
                "Status",
                "Kerze",
                "Wolke",
                "Trend",
                "Setter",
                "Welle",
            ]

            # Signals to ignore for now
            IGNORED_SIGNALS = {"bull_schwarz", "bull_1"}

            for key in check_keys:
                required_values = set()
                if isinstance(ranking_data, list):
                    for item in ranking_data:
                        if isinstance(item, dict) and key in item:
                            val = str(item[key]).strip()
                            if key == "Signal":
                                # Handle cases like 'bull_1 (NEU)' or 'Red Green Rocket (RGR)'
                                if " (" in val:
                                    val = val.split(" (")[0].strip()
                                if val in IGNORED_SIGNALS:
                                    continue
                            required_values.add(val)
                elif isinstance(ranking_data, dict):
                    if key == "Signal":
                        for signal_name, rules in ranking_data.items():
                            if isinstance(rules, dict) and (
                                "Score" in rules or "SQN" in rules
                            ):
                                val = str(signal_name).strip()
                                if " (" in val:
                                    val = val.split(" (")[0].strip()
                                if val in IGNORED_SIGNALS:
                                    continue
                                required_values.add(val)
                    else:
                        required_values = {
                            str(rules[key])
                            for rules in ranking_data.values()
                            if isinstance(rules, dict) and key in rules
                        }

                if not required_values:
                    continue

                db_values = db_attributes.get(key, set())
                missing_values = required_values - db_values
                available_values = required_values & db_values

                if missing_values:
                    logger.warning(
                        f"Ranking-Check WARNUNG: Folgende Werte für '{key}' aus {ranking_yaml_path.name} "
                        f"fehlen in der Datenbank: {', '.join(missing_values)}"
                    )
                if available_values:
                    logger.info(
                        f"Ranking-Check OK: {len(available_values)} '{key}'-Werte "
                        f"aus {ranking_yaml_path.name} in DB vorhanden."
                    )
        except Exception as e:
            logger.error(f"Fehler beim Ranking-Check für {ranking_yaml_path.name}: {e}")
    else:
        logger.warning(
            f"Ranking-Check: Ranking Datei {ranking_yaml_path} nicht gefunden."
        )

    # 5. Screener Engine (DI: Repos and Strategies)
    active_strategies = [
        DipBuyerStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            telegram_bot=telegram,
        ),
        TurnoverTimingStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            telegram_bot=telegram,
        ),
        CrocSetupStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            signal_repository=signal_repository,
            telegram_bot=telegram,
        ),
        TwoPercentStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            telegram_bot=telegram,
        ),
        NDXMomentumScreener(
            trade_repository=trade_repository,
            market_data_provider=md_provider,
            telegram_bot=telegram,
        ),
    ]

    screener = ScreenerEngine(
        trade_repository=trade_repository,
        signal_repository=signal_repository,
        data_provider=md_provider,
        strategies=active_strategies,
        configuration=loaded_config,
        telegram_bot=telegram,
    )
    app.extensions["screener_engine"] = screener

    # 6. Trade Manager
    tm = TradeManager(
        db_path=db_signals, stocks_db_path=db_stocks, telegram_bot=telegram
    )
    app.extensions["trade_manager"] = tm


def configure_scheduler(app, config):
    """Konfiguriert den Scheduler und die Jobs."""
    scheduler = BackgroundScheduler()
    db_stocks = Path(config.get_db_path("stocks"))

    # --- JOB 1: Marktdaten Update (Zweimal täglich) ---
    # a) 17:00 NY Time (EOD US)
    scheduler.add_job(
        func=run_market_data_update,
        args=[db_stocks],
        trigger=CronTrigger(
            hour=17, minute=0, timezone=pytz.timezone("America/New_York")
        ),
        id="market_data_update_ny",
        replace_existing=True,
    )

    # b) 03:00 Berlin Time (Sanitary Check / Early EU)
    scheduler.add_job(
        func=run_market_data_update,
        args=[db_stocks],
        trigger=CronTrigger(hour=3, minute=0, timezone=pytz.timezone("Europe/Berlin")),
        id="market_data_update_berlin",
        replace_existing=True,
    )

    # --- JOB 2: DB Maintenance (Sonntags 04:00) ---
    scheduler.add_job(
        func=run_db_maintenance,
        args=[db_stocks],
        trigger=CronTrigger(
            day_of_week="sun", hour=4, timezone=pytz.timezone("Europe/Berlin")
        ),
        id="db_maintenance",
        replace_existing=True,
    )

    # --- JOB 3: Trade Manager (Active Positions / Orders) ---
    tm = app.extensions.get("trade_manager")
    if tm:
        scheduler.add_job(
            func=tm.run_daily_process,
            trigger=CronTrigger(
                day_of_week="mon-sat",
                hour=7,
                minute=0,
                timezone=pytz.timezone("Europe/Berlin"),
            ),
            id="trade_manager_process",
            replace_existing=True,
        )

    # --- JOB 4: Strategy Check (Screener) ---
    scheduler.add_job(
        func=run_daily_strategy_check,
        args=[app],
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=6,
            minute=30,
            timezone=pytz.timezone("Europe/Berlin"),
        ),
        id="strategy_check",
        replace_existing=True,
    )

    # --- JOB 4.5: Cache Pre-warming (NACH Trade Manager) ---
    scheduler.add_job(
        func=run_cache_prewarm,
        args=[app],
        trigger=CronTrigger(
            day_of_week="mon-sat",
            hour=7,
            minute=15,
            timezone=pytz.timezone("Europe/Berlin"),
        ),
        id="cache_prewarming",
        replace_existing=True,
    )

    # --- JOB 5: Daily Backup (Signals DB) ---
    db_signals = Path(config.get_db_path("signals"))
    scheduler.add_job(
        func=run_db_backup,
        args=[db_signals],
        trigger=CronTrigger(hour=1, minute=0, timezone=pytz.timezone("Europe/Berlin")),
        id="daily_backup_signals",
        replace_existing=True,
    )

    scheduler.start()
    app.extensions["scheduler"] = scheduler
    logger.info("Scheduler gestartet und Jobs geplant.")
