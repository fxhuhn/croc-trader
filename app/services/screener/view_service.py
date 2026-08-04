import json
import logging

from ...const import Strategies, TradeStatus
from ...database.repositories.signal import SignalRepository

logger = logging.getLogger(__name__)


class ScreenerViewService:
    """Service to handle data preparation and business logic for screener views.

    Encapsulates logic for fetching candidates, parsing context, and aggregating results
    to keep view functions clean and focused on presentation.
    """

    def __init__(self, signal_repository: SignalRepository) -> None:
        self.signal_repository = signal_repository

    def _parse_context(
        self, raw_signal_context: str | None | dict[str, object]
    ) -> dict[str, object]:
        """Safely parses signal context from JSON string or returns dictionary."""
        if isinstance(raw_signal_context, dict):
            return raw_signal_context

        if not raw_signal_context:
            return {}

        if isinstance(raw_signal_context, str):
            try:
                return json.loads(raw_signal_context)
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning(
                    "Failed to parse signal context: %s. Error: %s",
                    raw_signal_context,
                    error,
                )
                return {}

        return {}

    def get_candidates(
        self, strategy: str | Strategies, limit: int = 100
    ) -> list[dict[str, object]]:
        """Fetches and prepares trade candidates for a given strategy.

        Args:
            strategy: The strategy identifier (Enum or string).
            limit: Maximum number of candidates to return.

        Returns:
            list[dict[str, object]]: List of processed candidate dictionaries.
        """
        # Ensure we pass the string value of the Enum if it's an Enum
        strategy_value = str(strategy)

        if (
            strategy_value.lower().startswith("croc")
            or strategy_value == Strategies.CrocSetup
        ):
            results = self._fetch_croc_candidates(limit)
        else:
            results = self._fetch_standard_candidates(strategy, strategy_value, limit)

        processed_results = []
        for row in results:
            candidate = dict(row)

            # Parse Context
            raw_signal_context = candidate.get("signal_context")
            context = self._parse_context(raw_signal_context)
            candidate["context"] = context

            # Position Status (NEW vs HOLD)
            status_val = str(candidate.get("status", "")).upper()
            candidate["position_status"] = (
                "HOLD" if status_val == str(TradeStatus.ACTIVE).upper() else "NEW"
            )

            # Standardize Date Display (Strict)
            date_value = context.get("date") or context.get("setup_date")
            if date_value:
                candidate["display_date"] = str(date_value).split("T")[0].split(" ")[0]
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

        # Sort by Setup Score DESC for Dip Buyer
        if strategy_value == Strategies.DipBuyer:
            processed_results.sort(
                key=lambda x: float(
                    x.get("context", {}).get("setup_score", 0.0) or 0.0
                ),
                reverse=True,
            )

        return processed_results

    def _fetch_croc_candidates(self, limit: int) -> list[dict[str, object]]:
        """Fetches unique trade candidates for Croc strategies."""
        strategies_to_fetch = [
            str(Strategies.HoldTarget),
            str(Strategies.SplitTarget),
            "Croc_",  # Legacy
        ]
        all_results = []
        seen_ids = set()

        for strategy_name in strategies_to_fetch:
            rows = self.signal_repository.get_trade_candidates(
                strategy_name, limit=limit, statuses=[TradeStatus.CREATED]
            )
            self._filter_new_candidates(rows, seen_ids, all_results)

        # Sort by created_at descending and limit to strict 3
        return sorted(
            all_results, key=lambda x: x.get("created_at") or "", reverse=True
        )[:3]

    def _filter_new_candidates(
        self,
        candidates: list[dict[str, object]],
        seen_ids: set[str | int],
        destination: list[dict[str, object]],
    ) -> None:
        """Filters out duplicate candidates based on their unique ID.

        Args:
            candidates: List of candidate dictionary records.
            seen_ids: Set of IDs that have already been collected.
            destination: Target list to append unique candidates to.
        """
        for candidate in candidates:
            if candidate["id"] not in seen_ids:
                destination.append(candidate)
                seen_ids.add(candidate["id"])

    def _fetch_standard_candidates(
        self, strategy: str | Strategies, strategy_value: str, limit: int
    ) -> list[dict[str, object]]:
        """Fetches candidates for standard non-Croc strategies.

        Args:
            strategy: The strategy identifier or list of strategies.
            strategy_value: String representation of the strategy.
            limit: Maximum candidates to fetch.

        Returns:
            list[dict[str, object]]: Raw database candidate records.
        """
        if strategy_value == str(Strategies.NDXMomentum):
            return self.signal_repository.get_trade_candidates(
                strategy_value,
                limit=limit,
                statuses=[TradeStatus.CREATED, TradeStatus.ACTIVE],
            )
        if isinstance(strategy, list):
            # Using the newly updated repository method that supports lists
            return self.signal_repository.get_trade_candidates(
                strategy, limit=limit, statuses=[TradeStatus.CREATED]
            )
        return self.signal_repository.get_trade_candidates(
            strategy_value, limit=limit, statuses=[TradeStatus.CREATED]
        )

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

    def get_turnover_candidates(self, limit: int = 200) -> list[dict[str, object]]:
        """Fetches and aggregates Turnover Timing candidates.

        Aggregates multiple signal variations (e.g., 0.5 vs 1.0) for the same symbol.
        Uses the 20-day Average Dollar Volume (Turnover SMA 20) for ranking.

        Args:
            limit: Maximum number of candidates to fetch.

        Returns:
            list[dict[str, object]]: List of aggregated candidate dictionaries.
        """
        # Fetch raw candidates using the base strategy name to catch variants
        results = self.signal_repository.get_trade_candidates(
            str(Strategies.TurnOverTiming), limit=limit, statuses=[TradeStatus.CREATED]
        )

        aggregated_results: dict[str, dict[str, object]] = {}

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

            self._aggregate_turnover_row(aggregated_results[symbol], row)

        # Sort by Dollar Volume (Turnover SMA 20) descending
        sorted_candidates = sorted(
            aggregated_results.values(),
            key=lambda x: float(x.get("dollar_volume") or 0.0),
            reverse=True,
        )

        return sorted_candidates

    def _aggregate_turnover_row(
        self, candidate_dict: dict[str, object], row: dict[str, object]
    ) -> None:
        """Aggregates metrics and pricing from a single candidate row.

        Args:
            candidate_dict: The dictionary holding aggregated data for the symbol.
            row: The raw trade candidate database record.
        """
        try:
            raw_signal_context = row.get("signal_context")
            context = self._parse_context(raw_signal_context)

            # Update common metrics
            if context.get("setup_close"):
                candidate_dict["close"] = float(context["setup_close"])
            if context.get("setup_atr"):
                candidate_dict["atr"] = float(context["setup_atr"])
            if context.get("setup_turnover_sma"):
                candidate_dict["dollar_volume"] = float(context["setup_turnover_sma"])

            # Strict Date Extraction (No created_at fallback)
            date_value = context.get("date") or context.get("setup_date")
            if date_value:
                candidate_dict["display_date"] = (
                    str(date_value).split("T")[0].split(" ")[0]
                )

            # Extract and Harmonize Index
            raw_indices = context.get("indices") or context.get("bucket")
            if raw_indices:
                candidate_dict["index"] = self.harmonize_indices(str(raw_indices))

            # Identify variant
            strategy_name = row["strategy"]
            entry_price = float(row.get("entry_price") or 0.0)

            # Strict Strategy Matching
            if strategy_name == Strategies.TurnOverTiming_05:
                candidate_dict["entry_0_5"] = entry_price
            elif strategy_name == Strategies.TurnOverTiming_10:
                candidate_dict["entry_1_0"] = entry_price

        except (ValueError, TypeError) as error:
            logger.warning(
                "Error aggregating turnover candidate %s: %s",
                candidate_dict.get("symbol"),
                error,
            )
