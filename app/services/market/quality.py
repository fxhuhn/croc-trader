import logging
from datetime import datetime, timedelta

from app.services.market.updater import MarketDataUpdater
from app.tools.market_holidays import MarketHolidayChecker
from app.tools.trading_calendar import get_last_completed_trading_day

logger = logging.getLogger(__name__)


class MarketQualityService:
    """
    Responsible for checking data integrity and coverage.
    - Check Recency (Gap at the end)
    - Check History (Gap at the start)
    """

    def __init__(
        self,
        updater: MarketDataUpdater,
        holiday_checker: MarketHolidayChecker | None = None,
    ):
        self.updater = updater
        self.repo = updater.repo
        self.holiday_checker = holiday_checker or MarketHolidayChecker()

    def perform_gap_check(self) -> None:
        """
        Executes standard EOD gap checks and triggers repairs if needed.
        """
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
            outdated = self.repo.get_outdated_symbols(thresh_recency)
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
