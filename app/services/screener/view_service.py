import json
import logging
from typing import Any

from ...const import Strategies, TradeStatus
from ...database.repositories.signal import SignalRepository

logger = logging.getLogger(__name__)


class ScreenerViewService:
    """Service to handle data preparation and business logic for screener views.

    Encapsulates logic for fetching candidates, parsing context, and aggregating results
    to keep view functions clean and focused on presentation.
    """

    def __init__(self, signal_repository: SignalRepository):
        self.signal_repository = signal_repository

    def _parse_context(
        self, raw_context: str | None | dict[str, Any]
    ) -> dict[str, Any]:
        """Safely parses signal context from JSON string or returns dictionary."""
        if isinstance(raw_context, dict):
            return raw_context

        if not raw_context:
            return {}

        if isinstance(raw_context, str):
            try:
                return json.loads(raw_context)
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning(
                    "Failed to parse signal context: %s. Error: %s", raw_context, error
                )
                return {}

        return {}

    def get_candidates(
        self, strategy: str | Strategies, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetches and prepares trade candidates for a given strategy.

        Args:
            strategy: The strategy identifier (Enum or string).
            limit: Maximum number of candidates to return.

        Returns:
            list[dict[str, Any]]: List of processed candidate dictionaries.
        """
        # Ensure we pass the string value of the Enum if it's an Enum
        strategy_value = str(strategy)

        # Handle "Croc" Aggregation
        if (
            strategy_value.lower().startswith("croc")
            or strategy_value == Strategies.CrocSetup
        ):
            strategies_to_fetch = [
                str(Strategies.HoldTarget),
                str(Strategies.SplitTarget),
                "Croc_",  # Legacy
            ]

            all_results = []
            seen_ids = set()

            for strat in strategies_to_fetch:
                rows = self.signal_repository.get_trade_candidates(
                    strat, limit=limit, statuses=[TradeStatus.CREATED]
                )
                for row in rows:
                    if row["id"] not in seen_ids:
                        all_results.append(row)
                        seen_ids.add(row["id"])

            # Sort by created_at descending
            results = sorted(
                all_results, key=lambda x: x.get("created_at") or "", reverse=True
            )[:limit]

        else:
            # Standard Single Strategy Fetch
            # This now handles NDX Momentum too (stored in trades as CREATED on month-end)
            results = self.signal_repository.get_trade_candidates(
                strategy_value, limit=limit, statuses=[TradeStatus.CREATED]
            )

        processed_results = []
        for row in results:
            candidate = dict(row)

            # Parse Context
            raw_context = candidate.get("signal_context")
            context = self._parse_context(raw_context)
            candidate["context"] = context

            # Standardize Date Display (Strict)
            date_val = context.get("date") or context.get("setup_date")
            if date_val:
                candidate["display_date"] = str(date_val).split("T")[0].split(" ")[0]
            else:
                logger.warning(
                    "No date found in signal context for %s", candidate.get("symbol")
                )
                candidate["display_date"] = "-"

            processed_results.append(candidate)

        # Sort by Momentum Score DESC for NDX Momentum
        if strategy_value == Strategies.NDXMomentum:
            processed_results.sort(
                key=lambda x: float(x.get("context", {}).get("momentum_score", 0.0)),
                reverse=True,
            )

        return processed_results

    @staticmethod
    def harmonize_indices(raw_indices: str) -> str:
        """Harmonizes index names to short codes (SPX, NDX, RUS, DOW)."""
        if not raw_indices:
            return "-"

        index_parts = [part.strip() for part in raw_indices.split(",")]
        mapped_index_parts = []

        mapping = {
            "NASDAQ_100": "NDX",
            "SP_500": "SPX",
            "RUSSELL_1000": "RUS",
            "RUSSELL_2000": "RUT",
            "DOW_JONES": "DOW",
        }

        for part in index_parts:
            # Check explicit mapping first
            if part in mapping:
                mapped_index_parts.append(mapping[part])
            else:
                # Fallback: Replace underscores, keep as is
                mapped_index_parts.append(part.replace("_", " "))

        return ", ".join(mapped_index_parts)

    def get_turnover_candidates(self, limit: int = 200) -> list[dict[str, Any]]:
        """Fetches and aggregates Turnover Timing candidates.

        Aggregates multiple signal variations (e.g., 0.5 vs 1.0) for the same symbol.
        Uses the 20-day Average Dollar Volume (Turnover SMA 20) for ranking.

        Args:
            limit: Maximum number of candidates to fetch.

        Returns:
            list[dict[str, Any]]: List of aggregated candidate dictionaries.
        """
        # Fetch raw candidates using the base strategy name to catch variants
        # Assuming the repository supports LIKE matching or we pass the base strategy
        results = self.signal_repository.get_trade_candidates(
            str(Strategies.TurnOverTiming), limit=limit, statuses=[TradeStatus.CREATED]
        )

        aggregated_results: dict[str, dict[str, Any]] = {}

        for row in results:
            symbol = row["symbol"]

            # Initialize aggregation bucket if needed
            if symbol not in aggregated_results:
                aggregated_results[symbol] = {
                    "symbol": symbol,
                    "display_date": "-",
                    "entry_0_5": None,
                    "entry_1_0": None,
                    "close": 0.0,
                    "atr": 0.0,
                    "dollar_volume": 0.0,
                    "index": "-",
                }

            try:
                raw_context = row.get("signal_context")
                context = self._parse_context(raw_context)

                # Update common metrics (last one wins, usually identical for same day)
                if context.get("setup_close"):
                    aggregated_results[symbol]["close"] = float(context["setup_close"])
                if context.get("setup_atr"):
                    aggregated_results[symbol]["atr"] = float(context["setup_atr"])
                if context.get("setup_turnover_sma"):
                    aggregated_results[symbol]["dollar_volume"] = float(
                        context["setup_turnover_sma"]
                    )

                # Strict Date Extraction (No created_at fallback)
                date_val = context.get("date") or context.get("setup_date")
                if date_val:
                    aggregated_results[symbol]["display_date"] = (
                        str(date_val).split("T")[0].split(" ")[0]
                    )

                # Extract and Harmonize Index
                raw_indices = context.get("indices") or context.get("bucket")
                if raw_indices:
                    aggregated_results[symbol]["index"] = self.harmonize_indices(
                        str(raw_indices)
                    )

                # Identify variant
                strategy_name = row["strategy"]
                entry_price = float(row.get("entry_price") or 0.0)

                # Strict Strategy Matching
                if strategy_name == Strategies.TurnOverTiming_05:
                    aggregated_results[symbol]["entry_0_5"] = entry_price
                elif strategy_name == Strategies.TurnOverTiming_10:
                    aggregated_results[symbol]["entry_1_0"] = entry_price

            except (ValueError, TypeError) as error:
                logger.warning(
                    "Error aggregating turnover candidate %s: %s", symbol, error
                )

        # Sort by Dollar Volume (Turnover SMA 20) descending
        sorted_candidates = sorted(
            aggregated_results.values(),
            key=lambda x: float(x.get("dollar_volume") or 0.0),
            reverse=True,
        )

        return sorted_candidates
