import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytz
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ..database.repositories.signal import SignalRepository
from ..database.repositories.trade import TradeRepository
from ..database.session import DatabaseSession

if TYPE_CHECKING:
    from flask import Flask

    from ..config import ConfigManager

from ..database.repositories.market_data_provider import MarketDataProvider
from ..services.screener import ScreenerEngine
from ..services.screener.engine import ScreenerConfiguration
from ..services.screener.protocols import StrategyProtocol
from ..services.telegram import TelegramBot
from ..services.trade_manager import TradeManager

# Tasks importieren
from ..tasks import (
    run_cache_prewarm,
    run_daily_strategy_check,
    run_db_backup,
    run_db_maintenance,
    run_market_data_update,
    run_order_generation,
)
from ..tools.market_holidays import MarketHolidayChecker
from ..tools.symbol_exchange import SymbolExchange
from ..tools.symbol_filter import SymbolFilter
from .ranking_verification import verify_ranking_system

# Strategies
from .screener.strategies.bounce_bandit import BounceBanditStrategy
from .screener.strategies.bridge_scout import BridgeScoutStrategy
from .screener.strategies.croc_setup import CrocSetupStrategy
from .screener.strategies.dip_buyer import DipBuyerStrategy
from .screener.strategies.ndx_momentum import NDXMomentumScreener
from .screener.strategies.tgim import TGIMStrategy
from .screener.strategies.turnover_timing import TurnoverTimingStrategy
from .screener.strategies.two_percent_strategy import TwoPercentStrategy

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

    # Create repositories
    trade_repository = TradeRepository(signals_session)
    signal_repository = SignalRepository(signals_session)

    # Initialize schemas (ensure tables exist)
    trade_repository.init_schema()
    signal_repository.init_schema()

    # 4. Load screener configuration
    yaml_path = config.get_strategy_path()
    loaded_config: ScreenerConfiguration = {}
    if yaml_path.exists():
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                loaded_config = {
                    "strategy_ranking": [
                        str(x) for x in data.get("strategy_ranking", [])
                    ]
                }
            elif isinstance(data, list):
                loaded_config = {"strategy_ranking": [str(x) for x in data]}
            logging.info("Strategy config loaded.")
        except Exception as e:
            logging.error("Failed to load strategy YAML: %s", e)

    # 4.5 Ranking System Verification
    verify_ranking_system(
        ranking_yaml_path=config.get_path("ranking_yaml"),
        signal_repository=signal_repository,
    )

    # 5. Screener Engine (DI: Repos and Strategies)
    active_strategies: list[StrategyProtocol] = [
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
        TGIMStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            telegram_bot=telegram,
        ),
        BridgeScoutStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
            telegram_bot=telegram,
            holiday_checker=holiday_checker,
        ),
        BounceBanditStrategy(
            trade_repository=trade_repository,
            data_provider=md_provider,
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

    # 7. MCP Domain Server
    if getattr(config.app, "mcp", None) and config.app.mcp.enabled:
        try:
            # Lazy import to avoid hard dependency on optional mcp package in production
            from ..mcp import create_mcp_server  # noqa: PLC0415

            app.extensions["mcp_server"] = create_mcp_server()
            logger.info("MCP Domain Server registered in app extensions.")
        except ImportError as import_err:
            logger.warning(
                "MCP package not available, skipping MCP server: %s", import_err
            )


def configure_scheduler(app: "Flask", config: "ConfigManager") -> None:
    """Configures the APScheduler background jobs."""
    scheduler = BackgroundScheduler()
    db_stocks = Path(config.get_db_path("stocks"))

    # --- JOB 1: Market Data Update (Twice daily) ---
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

    # --- JOB 2: DB Maintenance (Sundays 04:00) ---
    scheduler.add_job(
        func=run_db_maintenance,
        args=[db_stocks],
        trigger=CronTrigger(
            day_of_week="sun", hour=4, timezone=pytz.timezone("Europe/Berlin")
        ),
        id="db_maintenance",
        replace_existing=True,
    )

    # --- JOB 3: Trade Manager (Active Positions / Orders - 06:00) ---
    # Must run BEFORE screener to resolve yesterday's pending entries and free position slots
    tm = app.extensions.get("trade_manager")
    if tm:
        scheduler.add_job(
            func=tm.run_daily_process,
            trigger=CronTrigger(
                day_of_week="mon-sat",
                hour=6,
                minute=0,
                timezone=pytz.timezone("Europe/Berlin"),
            ),
            id="trade_manager_process",
            replace_existing=True,
        )

    # --- JOB 4: Strategy Check (Screener - 06:30) ---
    # Runs AFTER TradeManager on a guaranteed clean state
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

    # --- JOB 4.5: Order Generation (07:00) ---
    # Generates orders for active positions and new screener trades
    if tm:
        scheduler.add_job(
            func=run_order_generation,
            args=[app],
            trigger=CronTrigger(
                day_of_week="mon-sat",
                hour=7,
                minute=0,
                timezone=pytz.timezone("Europe/Berlin"),
            ),
            id="order_generation",
            replace_existing=True,
        )

    # --- JOB 4.6: Cache Pre-warming (07:15) ---
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
    logger.info("Scheduler started and jobs scheduled.")

    # --- Startup: Execute Cache Pre-warming ---
    run_cache_prewarm(app)
