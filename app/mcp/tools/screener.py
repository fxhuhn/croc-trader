import datetime
import json
import logging
from dataclasses import dataclass
from typing import Any

from flask import current_app
from mcp.server import MCPServer

from ...const import Strategies
from ...routes.views.dependencies import (
    _get_screener_view_service,
    _get_signal_repository,
)
from ...tools.market_holidays import MarketHolidayChecker
from ...tools.trading_calendar import get_next_trading_day
from .actions import _resolve_strategy_enum

logger = logging.getLogger(__name__)


def get_screener_candidates(
    strategy: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Returns list of screener setup candidates.

    Args:
        strategy: Canonical strategy name (e.g. 'dip_buyer', 'tgim', 'ndx_momentum') or None for all.
        limit: Maximum candidates to return (default 50).
    """
    try:
        view_service = _get_screener_view_service()
        effective_limit = max(1, min(limit, 200))

        if strategy:
            candidates = view_service.get_candidates(strategy, limit=effective_limit)
        else:
            strategies_to_scan = [
                Strategies.DipBuyer,
                Strategies.TurnOverTiming,
                Strategies.TwoPercent,
                Strategies.NDXMomentum,
                Strategies.TGIM,
                Strategies.BridgeScout,
                Strategies.BounceBandit,
                Strategies.HoldTarget,
                Strategies.SplitTarget,
                Strategies.WoundedBullSprint,
            ]
            candidates = []
            for strat in strategies_to_scan:
                candidates.extend(view_service.get_candidates(strat, limit=20))

        return [dict(c) for c in candidates[:effective_limit]]
    except Exception as error:
        logger.error("Error fetching screener candidates: %s", error)
        return []


def get_turnover_candidates(limit: int = 50) -> list[dict[str, Any]]:
    """Returns list of turnover timing candidates.

    Args:
        limit: Maximum candidates to return (default 50).
    """
    try:
        effective_limit = max(1, min(limit, 200))
        view_service = _get_screener_view_service()
        candidates = view_service.get_turnover_candidates(limit=effective_limit)
        return [dict(c) for c in candidates[:effective_limit]]
    except Exception as error:
        logger.error("Error fetching turnover candidates: %s", error)
        return []


def get_webhook_signals(
    symbol: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Returns list of incoming webhook signals.

    Args:
        symbol: Optional ticker symbol filter.
        limit: Maximum signals to return (default 20, max 100).
    """
    try:
        effective_limit = max(1, min(limit, 100))
        repo = _get_signal_repository()
        if symbol:
            rows = repo.fetch_all(
                "SELECT * FROM croc WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol.upper().strip(), effective_limit),
            )
        else:
            rows = repo.fetch_all(
                "SELECT * FROM croc ORDER BY created_at DESC LIMIT ?",
                (effective_limit,),
            )
        results = []
        for row in rows:
            sig = dict(row)
            if isinstance(sig.get("data"), str):
                try:
                    sig["data"] = json.loads(str(sig["data"]))
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Signal payload is not valid JSON string.")
            results.append(sig)
        return results
    except Exception as error:
        logger.error("Error fetching webhook signals: %s", error)
        return []


@dataclass(frozen=True)
class _RegimeDateContext:
    as_of_market_date: datetime.date
    as_of_market_date_str: str
    upcoming_eod_date: datetime.date
    upcoming_eod_date_str: str


def _extract_strategy_regime_payload(
    strategy: Any,
    strategy_enum: Strategies,
    include_candidates: bool,
    limit: int,
) -> dict[str, Any]:
    """Extracts regime, calendar, and candidate data without modifying state."""
    dates = _resolve_regime_dates(strategy)
    candidates_data = _resolve_precheck_candidates(strategy, include_candidates, limit)

    if hasattr(strategy, "_is_calendar_window_active") and hasattr(
        strategy, "_is_macro_regime_defensive"
    ):
        return _build_wbs_regime_payload(
            strategy, strategy_enum, dates, candidates_data
        )

    if hasattr(strategy, "calculate_analysis"):
        analysis = strategy.calculate_analysis(
            analysis_date=dates.as_of_market_date_str, force_run=False
        )
        return {
            "status": "success",
            "strategy": strategy_enum.value,
            "as_of_market_date": dates.as_of_market_date_str,
            "upcoming_eod_date": dates.upcoming_eod_date_str,
            "date": analysis.get("date", dates.as_of_market_date_str),
            "regime": analysis.get("regime_indicators", {}),
            "is_rebalance_day": analysis.get("is_rebalance_day", False),
            "top_symbols": (
                analysis.get("top_symbols", [])[:limit] if include_candidates else []
            ),
            "pre_qualified_candidates": candidates_data,
        }

    return {
        "status": "success",
        "strategy": strategy_enum.value,
        "as_of_market_date": dates.as_of_market_date_str,
        "upcoming_eod_date": dates.upcoming_eod_date_str,
        "date": dates.as_of_market_date_str,
        "message": "Strategy does not define custom regime or calendar gates",
        "pre_qualified_candidates": candidates_data,
    }


def _resolve_regime_dates(strategy: Any) -> _RegimeDateContext:
    """Resolves as-of market date and projected next trading session date."""
    as_of_market_date_str = ""
    if hasattr(strategy, "data_provider") and hasattr(
        strategy.data_provider, "get_latest_date"
    ):
        latest = strategy.data_provider.get_latest_date()
        if isinstance(latest, str) and latest:
            as_of_market_date_str = latest

    if as_of_market_date_str:
        try:
            as_of_market_date = datetime.datetime.strptime(
                as_of_market_date_str, "%Y-%m-%d"
            ).date()
        except ValueError:
            as_of_market_date = datetime.date.today()
            as_of_market_date_str = as_of_market_date.strftime("%Y-%m-%d")
    else:
        as_of_market_date = datetime.date.today()
        as_of_market_date_str = as_of_market_date.strftime("%Y-%m-%d")

    holiday_checker = getattr(strategy, "holiday_checker", None)
    if isinstance(holiday_checker, MarketHolidayChecker):
        upcoming_eod_date = get_next_trading_day(as_of_market_date, holiday_checker)
    else:
        try:
            upcoming_eod_date = get_next_trading_day(as_of_market_date)
        except Exception:
            upcoming_eod_date = as_of_market_date

    return _RegimeDateContext(
        as_of_market_date=as_of_market_date,
        as_of_market_date_str=as_of_market_date_str,
        upcoming_eod_date=upcoming_eod_date,
        upcoming_eod_date_str=upcoming_eod_date.strftime("%Y-%m-%d"),
    )


def _resolve_precheck_candidates(
    strategy: Any,
    include_candidates: bool,
    limit: int,
) -> list[dict[str, Any]]:
    """Retrieves precheck candidate setups without mutating application state."""
    if not include_candidates:
        return []

    if hasattr(strategy, "get_precheck_candidates"):
        result = strategy.get_precheck_candidates(limit=limit)
        if isinstance(result, list) and (
            result or not hasattr(strategy, "latest_qualifying_setups")
        ):
            return result

    if hasattr(strategy, "latest_qualifying_setups"):
        raw_setups = strategy.latest_qualifying_setups
        if isinstance(raw_setups, set | list | frozenset) and raw_setups:
            cached_symbols = sorted(raw_setups)[:limit]
            return [
                strategy.analyze_single_symbol(symbol)
                for symbol in cached_symbols
                if hasattr(strategy, "analyze_single_symbol")
            ]

    return []


def _build_wbs_regime_payload(
    strategy: Any,
    strategy_enum: Strategies,
    dates: _RegimeDateContext,
    candidates_data: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assembles regime payload for strategies implementing calendar and macro gates."""
    is_calendar_active, tdom, td_left = strategy._is_calendar_window_active(
        dates.upcoming_eod_date
    )
    is_defensive, spy_mtd, tlt_mtd = strategy._is_macro_regime_defensive(
        dates.as_of_market_date, dates.as_of_market_date_str
    )
    available_slots = (
        strategy._get_available_portfolio_slots()
        if hasattr(strategy, "_get_available_portfolio_slots")
        else 0
    )
    max_positions = (
        getattr(strategy.configuration, "max_positions", 5)
        if hasattr(strategy, "configuration")
        else 5
    )

    overall_ready = bool(is_calendar_active and is_defensive and available_slots > 0)
    blocking_reason = (
        "Calendar window inactive"
        if not is_calendar_active
        else (
            "Macro regime not defensive"
            if not is_defensive
            else ("No available portfolio slots" if available_slots <= 0 else "Ready")
        )
    )

    next_active_day = (
        getattr(strategy.configuration, "daysp", 14) + 1
        if hasattr(strategy, "configuration")
        else 15
    )

    return {
        "status": "success",
        "strategy": strategy_enum.value,
        "as_of_market_date": dates.as_of_market_date_str,
        "upcoming_eod_date": dates.upcoming_eod_date_str,
        "date": dates.as_of_market_date_str,
        "calendar": {
            "trading_day_of_month": tdom,
            "bars_left": td_left,
            "is_window_active": is_calendar_active,
            "next_active_day": next_active_day,
            "gate_passed": is_calendar_active,
        },
        "macro_regime": {
            "benchmarks": {
                "SPY_MTD": round(spy_mtd * 100.0, 2),
                "TLT_MTD": round(tlt_mtd * 100.0, 2),
            },
            "condition": "TLT_MTD > SPY_MTD",
            "is_defensive": is_defensive,
            "gate_passed": is_defensive,
        },
        "portfolio": {
            "max_positions": max_positions,
            "available_slots": available_slots,
        },
        "summary": {
            "overall_signal_ready": overall_ready,
            "blocking_reason": blocking_reason,
        },
        "pre_qualified_candidates": candidates_data,
    }


def get_strategy_regime(
    strategy_name: str,
    include_candidates: bool = False,
    limit: int = 10,
) -> dict[str, Any]:
    """Returns macro regime, calendar status, and portfolio capacity for a strategy.

    Args:
        strategy_name: Canonical strategy name (e.g. 'wounded_bull_sprint', 'ndx_momentum').
        include_candidates: Whether to include pre-qualified candidates from latest run.
        limit: Maximum number of pre-qualified candidates to return (default 10).
    """
    try:
        strategy_enum = _resolve_strategy_enum(strategy_name)
        if not strategy_enum:
            return {
                "status": "error",
                "message": f"Strategy '{strategy_name}' not found",
            }

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return {"status": "error", "message": "ScreenerEngine not initialized"}

        strategy = screener_engine.get_strategy(strategy_enum)
        if not strategy:
            return {
                "status": "error",
                "message": f"Strategy '{strategy_enum.name}' not found in engine",
            }

        return _extract_strategy_regime_payload(
            strategy, strategy_enum, include_candidates, limit
        )
    except Exception as error:
        logger.exception(
            "Error fetching strategy regime for %s: %s", strategy_name, error
        )
        return {"status": "error", "message": str(error)}


def register(server: MCPServer) -> None:
    """Registers screener and signal inspection tools on the MCP server."""
    server.tool(
        name="get_screener_candidates",
        description=(
            "Fetches setup candidates generated by screening strategies "
            "(e.g. DipBuyer, NDXMomentum, TGIM, BridgeScout, BounceBandit, HoldTarget, SplitTarget)."
        ),
    )(get_screener_candidates)

    server.tool(
        name="get_strategy_regime",
        description=(
            "Fetches macro regime, calendar status, portfolio capacity, and optional "
            "pre-qualified candidates for a strategy without modifying state."
        ),
    )(get_strategy_regime)

    server.tool(
        name="get_turnover_candidates",
        description=(
            "Fetches aggregated Turnover Timing candidates ranked by 20-day Average Dollar Volume."
        ),
    )(get_turnover_candidates)

    server.tool(
        name="get_webhook_signals",
        description=(
            "Fetches recently ingested TradingView webhook signals from the signal registry."
        ),
    )(get_webhook_signals)
