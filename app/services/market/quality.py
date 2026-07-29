import logging
from datetime import datetime, timedelta

from app.services.market.updater import MarketDataUpdater
from app.services.telegram import TelegramBot
from app.tools.market_holidays import MarketHolidayChecker
from app.tools.trading_calendar import get_last_completed_trading_day
from app.types import TradeStatus

logger = logging.getLogger(__name__)


class MarketQualityService:
    """Responsible for checking data integrity and coverage.

    - Check Recency (Gap at the end)
    - Check History (Gap at the start)
    - Check Last Trading Day Completeness (Warning Alerts via Log & Telegram)
    """

    DEFAULT_CRITICAL_SYMBOLS: tuple[str, ...] = ("QQQ", "SPY")

    def __init__(
        self,
        updater: MarketDataUpdater,
        holiday_checker: MarketHolidayChecker | None = None,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        """Initializes MarketQualityService with required dependencies."""
        self.updater = updater
        self.repo = updater.repo
        self.holiday_checker = holiday_checker or MarketHolidayChecker()
        self.telegram_bot = telegram_bot

    def perform_gap_check(self) -> None:
        """Executes standard EOD gap checks and triggers repairs if needed."""
        logger.info("Performing gap check...")

        # 1. Check Recency (Data Gap at the end)
        # Dynamic calculation based on trading days and market holidays
        last_completed_trading_day = get_last_completed_trading_day(
            datetime.now().date(), self.holiday_checker
        )
        thresh_recency = last_completed_trading_day.strftime("%Y-%m-%d")

        # 2. Check History (Data Gap at the start / Shallow History)
        # Symbols starting AFTER this date (e.g. less than 300 days history)
        thresh_history = (datetime.now() - timedelta(days=300)).strftime("%Y-%m-%d")

        try:
            outdated = self.repo.get_outdated_symbols(thresh_recency, provider=None)
            shallow = self.repo.get_symbols_with_missing_history(thresh_history)

            # Combine unique symbols needing repair
            repair_candidates = set(outdated) | set(shallow)

            if repair_candidates:
                logger.warning(
                    "Repair Check: %d outdated, %d shallow history.",
                    len(outdated),
                    len(shallow),
                )
                logger.warning(
                    "Starting repair for %d symbols.", len(repair_candidates)
                )

                # Trigger Update for these specific symbols (Full Reload necessary to fix history)
                self.updater.run_update(
                    full_reload=True, specific_symbols=list(repair_candidates)
                )

                # Post-Repair Validation
                if shallow:
                    still_shallow = self.repo.get_symbols_with_missing_history(
                        thresh_history
                    )
                    persistent = set(shallow) & set(still_shallow)
                    if persistent:
                        sym_list = ", ".join(sorted(persistent))
                        logger.info(
                            "ℹ️ %d symbols remain incomplete (< %s) [IPO/Listing]: %s",
                            len(persistent),
                            thresh_history,
                            sym_list,
                        )
            else:
                logger.info(
                    "Gap Check: Everything up to date and history is sufficient."
                )

        except Exception as e:
            logger.error("Gap Check Error: %s", e, exc_info=True)

    def check_last_trading_day_completeness(
        self, max_allowed_missing_ratio: float = 0.05
    ) -> bool:
        """Validates market data completeness for the latest completed trading day.

        Checks fixed index symbols (QQQ, SPY) and symbols of active/created
        strategy trades. Triggers log and Telegram warnings if incomplete.

        Args:
            max_allowed_missing_ratio: Maximum allowed fraction of missing
              symbols across the universe before triggering a warning (default:
              5%).

        Returns:
            bool: True if data is complete, False if warning was triggered.
        """
        last_completed_trading_day = get_last_completed_trading_day(
            datetime.now().date(), self.holiday_checker
        )
        target_date_string = last_completed_trading_day.strftime("%Y-%m-%d")

        outdated_symbols = set(
            self.repo.get_outdated_symbols(target_date_string, provider=None)
        )
        all_symbols = set(self.repo.get_all_known_symbols())

        if not all_symbols:
            logger.info("No symbols registered in repository to check.")
            return True

        # Build critical symbols set (Fixed benchmarks + active strategy trades)
        critical_symbols = set(self.DEFAULT_CRITICAL_SYMBOLS)

        try:
            active_and_created = self.updater.trade_repository.get_by_status(
                [TradeStatus.CREATED, TradeStatus.ACTIVE]
            )
            for trade_record in active_and_created:
                symbol = trade_record.get("symbol")
                if symbol and isinstance(symbol, str):
                    critical_symbols.add(symbol)
        except Exception as repo_error:
            logger.warning(
                "Could not load active trade symbols for completeness check: %s",
                repo_error,
            )

        missing_critical = sorted(
            [symbol for symbol in critical_symbols if symbol in outdated_symbols]
        )
        missing_ratio = len(outdated_symbols) / len(all_symbols)

        if not missing_critical and missing_ratio <= max_allowed_missing_ratio:
            logger.info(
                "Market data for last trading day (%s) is complete (%d/%d symbols up to date).",
                target_date_string,
                len(all_symbols) - len(outdated_symbols),
                len(all_symbols),
            )
            return True

        # Construct diagnostic warning message
        warning_lines = [
            f"⚠️ **WARNUNG: Marktdaten für {target_date_string} unvollständig!**",
            f"• Fehlende Symbole gesamt: {len(outdated_symbols)} / {len(all_symbols)} ({missing_ratio:.1%})",
        ]

        if missing_critical:
            warning_lines.append(
                f"• Kritisches Strategie-Symbol fehlt: {', '.join(missing_critical)}"
            )

        warning_message = "\n".join(warning_lines)

        logger.warning(
            "Market Data Incomplete for %s: %d/%d missing (Critical Missing: %s)",
            target_date_string,
            len(outdated_symbols),
            len(all_symbols),
            missing_critical,
        )

        if self.telegram_bot:
            try:
                self.telegram_bot.send_message(warning_message)
            except Exception as telegram_error:
                logger.error("Failed to dispatch Telegram warning: %s", telegram_error)

        return False
