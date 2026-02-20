import logging
from datetime import datetime, timedelta


from app.services.market.updater import MarketDataUpdater

logger = logging.getLogger(__name__)


class MarketQualityService:
    """
    Responsible for checking data integrity and coverage.
    - Check Recency (Gap at the end)
    - Check History (Gap at the start)
    """

    def __init__(self, updater: MarketDataUpdater):
        self.updater = updater
        self.repo = updater.repo

    def perform_gap_check(self) -> None:
        """
        Executes standard EOD gap checks and triggers repairs if needed.
        """
        logger.info("Führe Gap-Check durch...")

        # 1. Check Recency (Data Gap at the end)
        # Symbols not updated in the last 3 days
        thresh_recency = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

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
                    f"Repair Check: {len(outdated)} outdated, {len(shallow)} shallow history."
                )
                logger.warning(f"Starte Repair für {len(repair_candidates)} Symbole.")

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
                            f"ℹ️ {len(persistent)} Symbole bleiben unvollständig (< {thresh_history}) [IPO/Listing]: {sym_list}"
                        )
            else:
                logger.info("Gap Check: Alles aktuell und Historie ausreichend.")

        except Exception as e:
            logger.error(f"Gap Check Error: {e}", exc_info=True)
