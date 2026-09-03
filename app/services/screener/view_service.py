import json
import logging
from typing import Any

from ...const import STRATEGY_ALIASES, Strategies, TradeStatus
from ...database.repositories.signal import SignalRepository
from ...tools.indicators import extract_safe_float

logger = logging.getLogger(__name__)


MAX_NDX_MOMENTUM_LEADERS: int = 5

INDEX_CODE_MAPPING: dict[str, str] = {
    "NASDAQ_100": "NDX",
    "SP_500": "SPX",
    "RUSSELL_1000": "RUS",
    "RUSSELL_2000": "RUT",
    "DOW_JONES": "DOW",
}


def harmonize_index_string(raw_indices: str) -> str:
    """Harmonizes raw index string to short codes (SPX, NDX, RUS, DOW)."""
    if not raw_indices:
        return "-"
    return ", ".join(
        INDEX_CODE_MAPPING.get(part.strip(), part.strip().replace("_", " "))
        for part in raw_indices.split(",")
    )


def _resolve_canonical_strategy(
    strategy: str | Strategies | list[str],
) -> tuple[str | Strategies | list[str], str]:
    """Resolves strategy aliases and returns (resolved_strategy, strategy_value)."""
    normalized_key = str(strategy).replace(".", "_").replace("-", "_").lower()
    canonical_strategy = STRATEGY_ALIASES.get(normalized_key)
    resolved = canonical_strategy if canonical_strategy is not None else strategy
    return resolved, str(resolved)


def _is_croc_strategy(
    strategy: str | Strategies | list[str],
    resolved_strategy: str | Strategies | list[str],
    strategy_value: str,
) -> bool:
    """Checks whether the requested strategy is a Croc strategy."""
    croc_prefix = str(Strategies.CrocSetup).lower()
    return strategy_value.lower().startswith(croc_prefix) or Strategies.CrocSetup in (
        strategy,
        resolved_strategy,
        strategy_value,
    )


def _extract_display_date(context: dict[str, Any], symbol: object = "") -> str:
    """Extracts and formats display date from context or logs warning if missing."""
    date_value = context.get("date") or context.get("setup_date")
    if date_value:
        return str(date_value).split("T")[0].split(" ")[0]
    logger.warning("No date found in signal context for %s", symbol)
    return "-"


def _resolve_position_status(status_val: object) -> str:
    """Maps trade status to 'HOLD' (for ACTIVE) or 'NEW'."""
    return (
        "HOLD" if str(status_val).upper() == str(TradeStatus.ACTIVE).upper() else "NEW"
    )


def _enrich_candidate_row(
    row: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Enriches raw candidate dictionary with parsed context, status, and display date."""
    candidate = dict(row)
    candidate["context"] = context
    candidate["position_status"] = _resolve_position_status(candidate.get("status", ""))
    candidate["display_date"] = _extract_display_date(context, candidate.get("symbol"))
    return candidate


def _sort_candidates_by_score(
    candidates: list[dict[str, Any]], strategy_value: str
) -> list[dict[str, Any]]:
    """Sorts candidates by momentum_score (NDXMomentum) or setup_score (DipBuyer)."""
    if strategy_value == Strategies.NDXMomentum:
        candidates.sort(
            key=lambda x: extract_safe_float(
                x["context"].get("momentum_score")
                if isinstance(x.get("context"), dict)
                else None
            ),
            reverse=True,
        )
    elif strategy_value == Strategies.DipBuyer:
        candidates.sort(
            key=lambda x: extract_safe_float(
                x["context"].get("setup_score")
                if isinstance(x.get("context"), dict)
                else None
            ),
            reverse=True,
        )
    return candidates


def _update_turnover_metrics_from_context(
    candidate_dict: dict[str, Any], context: dict[str, Any]
) -> None:
    """Updates common numeric metrics, display date, and harmonized index from context."""
    if context.get("setup_close"):
        candidate_dict["close"] = float(str(context["setup_close"]))
    if context.get("setup_atr"):
        candidate_dict["atr"] = float(str(context["setup_atr"]))
    if context.get("setup_turnover_sma"):
        candidate_dict["dollar_volume"] = float(str(context["setup_turnover_sma"]))

    date_value = context.get("date") or context.get("setup_date")
    if date_value:
        candidate_dict["display_date"] = str(date_value).split("T")[0].split(" ")[0]

    raw_indices = context.get("indices") or context.get("bucket")
    if raw_indices:
        candidate_dict["index"] = harmonize_index_string(str(raw_indices))


def _apply_turnover_variant_entry(
    candidate_dict: dict[str, Any], strategy_name: object, entry_price: float
) -> None:
    """Updates candidate entry price based on 0.5 or 1.0 strategy variant."""
    if strategy_name == Strategies.TurnOverTiming_05:
        candidate_dict["entry_0_5"] = entry_price
    elif strategy_name == Strategies.TurnOverTiming_10:
        candidate_dict["entry_1_0"] = entry_price


class ScreenerViewService:
    """Service to handle data preparation and business logic for screener views.

    Encapsulates logic for fetching candidates, parsing context, and aggregating results
    to keep view functions clean and focused on presentation.
    """

    def __init__(self, signal_repository: SignalRepository) -> None:
        self.signal_repository = signal_repository

    def _parse_context(self, raw_signal_context: object) -> dict[str, Any]:
        """Safely parses signal context from JSON string or returns dictionary."""
        if isinstance(raw_signal_context, dict):
            return dict(raw_signal_context)

        if not raw_signal_context:
            return {}

        if isinstance(raw_signal_context, str):
            try:
                parsed = json.loads(raw_signal_context)
                if isinstance(parsed, dict):
                    return parsed
                return {}
            except (json.JSONDecodeError, TypeError) as error:
                logger.warning(
                    "Failed to parse signal context: %s. Error: %s",
                    raw_signal_context,
                    error,
                )
                return {}

        return {}

    def get_candidates(
        self, strategy: str | Strategies | list[str], limit: int = 100
    ) -> list[dict[str, Any]]:
        """Fetches and prepares trade candidates for a given strategy.

        Args:
            strategy: The strategy identifier (Enum or string or list).
            limit: Maximum number of candidates to return.

        Returns:
            list[dict[str, Any]]: List of processed candidate dictionaries.
        """
        resolved_strategy, strategy_value = _resolve_canonical_strategy(strategy)

        if _is_croc_strategy(strategy, resolved_strategy, strategy_value):
            results = self._fetch_croc_candidates(limit)
        else:
            results = self._fetch_standard_candidates(
                resolved_strategy, strategy_value, limit
            )

        processed_results = [
            _enrich_candidate_row(row, self._parse_context(row.get("signal_context")))
            for row in results
        ]

        return _sort_candidates_by_score(processed_results, strategy_value)

    def _fetch_croc_candidates(self, limit: int) -> list[dict[str, Any]]:
        """Fetches unique trade candidates for Croc strategies."""
        strategies_to_fetch = [
            str(Strategies.HoldTarget),
            str(Strategies.SplitTarget),
            "Croc_",  # Legacy
        ]
        all_results: list[dict[str, Any]] = []
        seen_ids: set[str | int] = set()

        for strategy_name in strategies_to_fetch:
            rows = self.signal_repository.get_trade_candidates(
                strategy_name, limit=limit, statuses=[TradeStatus.CREATED]
            )
            self._filter_new_candidates(rows, seen_ids, all_results)

        # Sort by created_at descending and limit to strict 3
        return sorted(
            all_results, key=lambda x: str(x.get("created_at") or ""), reverse=True
        )[:3]

    def _filter_new_candidates(
        self,
        candidates: list[dict[str, Any]],
        seen_ids: set[str | int],
        destination: list[dict[str, Any]],
    ) -> None:
        """Filters out duplicate candidates based on their unique ID.

        Args:
            candidates: List of candidate dictionary records.
            seen_ids: Set of IDs that have already been collected.
            destination: Target list to append unique candidates to.
        """
        for candidate in candidates:
            cand_id = candidate.get("id")
            if (
                cand_id is not None
                and isinstance(cand_id, str | int)
                and cand_id not in seen_ids
            ):
                destination.append(candidate)
                seen_ids.add(cand_id)

    def _fetch_standard_candidates(
        self, strategy: str | Strategies | list[str], strategy_value: str, limit: int
    ) -> list[dict[str, Any]]:
        """Fetches candidates for standard non-Croc strategies.

        Args:
            strategy: The strategy identifier or list of strategies.
            strategy_value: String representation of the strategy.
            limit: Maximum candidates to fetch.

        Returns:
            list[dict[str, Any]]: Raw database candidate records.
        """
        if strategy_value == str(Strategies.NDXMomentum):
            return self._fetch_ndx_momentum_candidates(limit)
        if isinstance(strategy, list):
            # Using the newly updated repository method that supports lists
            return self.signal_repository.get_trade_candidates(
                strategy, limit=limit, statuses=[TradeStatus.CREATED]
            )
        return self.signal_repository.get_trade_candidates(
            strategy_value, limit=limit, statuses=[TradeStatus.CREATED]
        )

    def _fetch_ndx_momentum_candidates(self, limit: int) -> list[dict[str, Any]]:
        """Fetches and deduplicates NDX Momentum candidates to unique symbols.

        If a complete set of CREATED signals (>= 5) exists for the monthly rebalance,
        these define the authoritative Top 5 for the new month. Symbols already held
        in the active portfolio are marked as HOLD, while newly entering symbols are
        marked as NEW.
        If fewer than 5 CREATED signals exist, remaining ACTIVE positions are included.
        """
        rows = self.signal_repository.get_trade_candidates(
            str(Strategies.NDXMomentum),
            limit=limit,
            statuses=[TradeStatus.CREATED, TradeStatus.ACTIVE],
        )

        (
            created_map,
            active_map,
            created_order,
            active_order,
        ) = self._group_momentum_rows(rows)

        return self._build_ndx_candidates(
            created_map, active_map, created_order, active_order, limit
        )

    def _group_momentum_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        list[str],
        list[str],
    ]:
        """Groups candidate rows into CREATED and ACTIVE symbol mappings."""
        created_map: dict[str, dict[str, Any]] = {}
        active_map: dict[str, dict[str, Any]] = {}
        created_order: list[str] = []
        active_order: list[str] = []

        for row in rows:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue

            status_str = str(row.get("status", "")).upper()
            if (
                status_str == str(TradeStatus.CREATED).upper()
                and symbol not in created_map
            ):
                created_map[symbol] = dict(row)
                created_order.append(symbol)
            elif (
                status_str == str(TradeStatus.ACTIVE).upper()
                and symbol not in active_map
            ):
                active_map[symbol] = dict(row)
                active_order.append(symbol)

        return created_map, active_map, created_order, active_order

    def _build_ndx_candidates(
        self,
        created_map: dict[str, dict[str, Any]],
        active_map: dict[str, dict[str, Any]],
        created_order: list[str],
        active_order: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Builds deduplicated Top 5 candidates, prioritizing CREATED rebalance signals."""
        results: list[dict[str, Any]] = []

        for symbol in created_order:
            candidate = dict(created_map[symbol])
            candidate["status"] = (
                str(TradeStatus.ACTIVE)
                if symbol in active_map
                else str(TradeStatus.CREATED)
            )
            results.append(candidate)

        if len(results) < MAX_NDX_MOMENTUM_LEADERS:
            for symbol in active_order:
                if symbol not in created_map:
                    results.append(dict(active_map[symbol]))
                    if len(results) >= MAX_NDX_MOMENTUM_LEADERS:
                        break

        return results[:limit]

    @staticmethod
    def harmonize_indices(raw_indices: str) -> str:
        """Harmonizes index names to short codes (SPX, NDX, RUS, DOW)."""
        return harmonize_index_string(raw_indices)

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
        results = self.signal_repository.get_trade_candidates(
            str(Strategies.TurnOverTiming), limit=limit, statuses=[TradeStatus.CREATED]
        )

        aggregated_results: dict[str, dict[str, Any]] = {}

        for row in results:
            symbol = str(row["symbol"])

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
        return sorted(
            aggregated_results.values(),
            key=lambda x: float(str(x.get("dollar_volume") or 0.0)),
            reverse=True,
        )

    def _aggregate_turnover_row(
        self, candidate_dict: dict[str, Any], row: dict[str, Any]
    ) -> None:
        """Aggregates metrics and pricing from a single candidate row.

        Args:
            candidate_dict: The dictionary holding aggregated data for the symbol.
            row: The raw trade candidate database record.
        """
        try:
            raw_signal_context = row.get("signal_context")
            context = self._parse_context(raw_signal_context)
            _update_turnover_metrics_from_context(candidate_dict, context)

            entry_price = float(str(row.get("entry_price") or 0.0))
            _apply_turnover_variant_entry(
                candidate_dict, row.get("strategy"), entry_price
            )

        except (ValueError, TypeError) as error:
            logger.warning(
                "Error aggregating turnover candidate %s: %s",
                candidate_dict.get("symbol"),
                error,
            )
