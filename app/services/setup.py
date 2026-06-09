import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING
import yaml
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database.session import DatabaseSession
from ..database.repositories.trade import TradeRepository
from ..database.repositories.signal import SignalRepository

if TYPE_CHECKING:
    from flask import Flask
    from ..config import ConfigManager

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
    run_order_generation,
)

logger = logging.getLogger(__name__)


def register_services(app: "Flask", config: "ConfigManager") -> None:
    """Initializes all services and attaches them to app.extensions."""

    db_stocks = Path(config.get_db_path("stocks"))
    db_signals = Path(config.get_db_path("signals"))

    # 1. Telegram Service
    telegram_config = config.app.telegram
    telegram = TelegramBot(
        token=telegram_config.token,
        chat_id=telegram_config.chat_id,
        enabled=telegram_config.enabled,
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
    verify_ranking_system(
        ranking_yaml_path=config.get_path("ranking_yaml"),
        signal_repository=signal_repository,
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


def configure_scheduler(app: "Flask", config: "ConfigManager") -> None:
    """Configures the APScheduler background jobs."""
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

        # --- JOB 3.5: Order Generation (NACH Trade Manager) ---
        scheduler.add_job(
            func=run_order_generation,
            args=[app],
            trigger=CronTrigger(
                day_of_week="mon-sat",
                hour=7,
                minute=5,
                timezone=pytz.timezone("Europe/Berlin"),
            ),
            id="order_generation",
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


@dataclass(frozen=True)
class RankingVerificationResult:
    """Contains verification results for a specific attribute key."""

    attribute_key: str
    missing_values: list[str]
    available_values: list[str]


def verify_ranking_system(
    ranking_yaml_path: Path, signal_repository: SignalRepository
) -> None:
    """Loads ranking configuration and verifies attributes against the database."""
    if not ranking_yaml_path.exists():
        logger.warning(
            f"Ranking-Check: Ranking Datei {ranking_yaml_path} nicht gefunden."
        )
        return

    try:
        with open(ranking_yaml_path, encoding="utf-8") as f:
            ranking_data = yaml.safe_load(f) or {}

        database_attributes = signal_repository.get_unique_signal_attributes()
        results = check_ranking_attributes(ranking_data, database_attributes)

        for result in results:
            if result.missing_values:
                logger.warning(
                    f"Ranking-Check WARNUNG: Folgende Werte für '{result.attribute_key}' "
                    f"aus {ranking_yaml_path.name} fehlen in der Datenbank: "
                    f"{', '.join(result.missing_values)}"
                )
            if result.available_values:
                logger.info(
                    f"Ranking-Check OK: {len(result.available_values)} '{result.attribute_key}'-Werte "
                    f"aus {ranking_yaml_path.name} in DB vorhanden."
                )
    except Exception as e:
        logger.error(f"Fehler beim Ranking-Check für {ranking_yaml_path.name}: {e}")


def check_ranking_attributes(
    ranking_data: list[Any] | dict[str, Any],
    database_attributes: dict[str, set[str]],
) -> list[RankingVerificationResult]:
    """Checks if values defined in the ranking configuration exist in the database.

    This is a pure logic function (Functional Core). It normalizes keys
    case-insensitively and performs case-insensitive value matching to be
    robust.
    """
    check_keys = [
        "Signal",
        "Status",
        "Kerze",
        "Wolke",
        "Trend",
        "Setter",
        "Welle",
    ]

    results = []

    for key in check_keys:
        if isinstance(ranking_data, list):
            required_values = _collect_values_from_list(ranking_data, key)
        elif isinstance(ranking_data, dict):
            required_values = _collect_values_from_dict(ranking_data, key)
        else:
            required_values = set()

        if not required_values:
            continue

        db_values = database_attributes.get(key, set())
        db_values_lower = {v.lower() for v in db_values}

        available = []
        missing = []

        for req_val in required_values:
            if req_val.lower() in db_values_lower:
                # Find the original cased value from the database if possible
                original_cased = next(
                    (v for v in db_values if v.lower() == req_val.lower()),
                    req_val,
                )
                available.append(original_cased)
            else:
                missing.append(req_val)

        results.append(
            RankingVerificationResult(
                attribute_key=key,
                missing_values=sorted(missing),
                available_values=sorted(available),
            )
        )

    return results


def _collect_values_from_list(ranking_list: list[Any], key: str) -> set[str]:
    """Extracts values for a key from a list of configuration items."""
    values = set()
    for item in ranking_list:
        if not isinstance(item, dict):
            continue
        val = _get_case_insensitive_value(item, key)
        if val is None:
            continue
        val_str = str(val).strip()
        if not val_str:
            continue
        if key == "Signal" and " (" in val_str:
            val_str = val_str.split(" (")[0].strip()
        values.add(val_str)
    return values


def _collect_values_from_dict(ranking_dict: dict[str, Any], key: str) -> set[str]:
    """Extracts values for a key from a dictionary of configuration items."""
    values = set()
    if key == "Signal":
        for signal_name, rules in ranking_dict.items():
            if not isinstance(rules, dict):
                continue
            if "Score" in rules or "SQN" in rules:
                val_str = str(signal_name).strip()
                if " (" in val_str:
                    val_str = val_str.split(" (")[0].strip()
                values.add(val_str)
    else:
        for rules in ranking_dict.values():
            if not isinstance(rules, dict):
                continue
            val = _get_case_insensitive_value(rules, key)
            if val is not None:
                val_str = str(val).strip()
                if val_str:
                    values.add(val_str)
    return values


def _get_case_insensitive_value(item: dict[str, Any], key: str) -> Any | None:
    """Gets a value from a dictionary using case-insensitive key lookup."""
    target_key = key.lower()
    for item_key, item_value in item.items():
        if item_key.lower() == target_key:
            return item_value
    return None
