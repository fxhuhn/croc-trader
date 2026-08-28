"""REST API routes and webhook ingestion handlers."""

import logging
import uuid
from threading import Lock, Thread
from typing import Any, TypedDict, cast

from flask import Blueprint, Response, current_app, jsonify, request

from ..const import STRATEGY_ALIASES, Strategies
from ..database.repositories.signal import SignalRepository
from ..database.session import DatabaseSession
from ..services.backfill_engine import run_strategy_backfill
from ..services.market.quality import MarketQualityService
from ..services.market.updater import MarketDataUpdater
from ..tasks import run_daily_eod_pipeline
from .security import require_ip_whitelist
from .views.dependencies import cache

logger = logging.getLogger(__name__)
api_blueprint = Blueprint("api", __name__)

_market_sync_lock: Lock = Lock()

type ApiResponse = Response | tuple[Response, int]


class WebhookPayload(TypedDict, total=False):
    """Schema for incoming signal webhooks."""

    symbol: str
    ticker: str
    timeframe: str
    signal: str
    strategy: str
    exchange: str
    timestamp: str
    date: str
    price: float


def _parse_boolean_parameter(
    value: str | bool | None, default_value: bool = True
) -> bool:
    """Parses incoming query parameters into booleans using guard clauses."""
    if value is None:
        return default_value
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("false", "0", "no")


def _extract_symbol_from_request() -> str | None:
    """Extracts target symbol from query parameters or JSON body."""
    symbol = request.args.get("symbol")
    if symbol:
        return str(symbol).strip().upper()

    data = request.get_json(force=True, silent=True)
    if data and isinstance(data, dict) and data.get("symbol"):
        return str(data["symbol"]).strip().upper()

    return None


def _extract_webhook_symbol(payload: WebhookPayload) -> str | None:
    """Extracts ticker symbol from a webhook payload dictionary."""
    symbol = payload.get("symbol") or payload.get("ticker")
    if symbol:
        return str(symbol).strip().upper()
    return None


# --- STANDARD ROUTES ---


@api_blueprint.route("/health", methods=["GET"])
def health_check() -> ApiResponse:
    """Simple health check endpoint.

    Returns:
        ApiResponse: JSON status OK.
    """
    return jsonify({"status": "ok"})


@api_blueprint.route("/", methods=["GET"])
@require_ip_whitelist
def root_check() -> ApiResponse:
    """Authenticated root check endpoint.

    Returns:
        ApiResponse: JSON status OK.
    """
    return jsonify({"status": "ok"})


# --- SIGNAL INGESTION ---


@api_blueprint.route("/webhook", methods=["POST"])
@require_ip_whitelist
def ingest_webhook() -> ApiResponse:
    """Ingests signal webhooks and persists them to the signal database.

    Returns:
        Response: JSON success with signal ID or error message.
    """
    try:
        payload: WebhookPayload | None = request.get_json(silent=True, force=True)

        if not payload:
            raw_data = request.get_data(as_text=True)
            logger.warning("⚠️ Malformed Webhook Data: %s", raw_data)
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400

        symbol = _extract_webhook_symbol(payload)
        if not symbol:
            logger.warning(
                "⚠️ Webhook rejected: Missing 'symbol' in payload %s", payload
            )
            return jsonify(
                {
                    "status": "error",
                    "message": "Missing mandatory field: symbol",
                }
            ), 400

        configuration = current_app.config.get("APP_CONFIG")
        database_path = (
            configuration.get_db_path("signals")
            if configuration
            else "instance/signals.db"
        )

        session = DatabaseSession(str(database_path))
        repository = SignalRepository(session)
        signal_id = repository.save_signal(dict(payload))

        logger.info("✅ Webhook saved: %s -> ID %s", symbol, signal_id)

        return jsonify({"status": "success", "id": signal_id}), 201

    except Exception as error:
        error_identifier = str(uuid.uuid4())[:8]
        logger.error(
            "Webhook processing error [%s]: %s",
            error_identifier,
            error,
            exc_info=True,
        )
        return jsonify(
            {
                "status": "error",
                "message": "Internal Server Error",
                "error_id": error_identifier,
            }
        ), 500


# --- SCREENER & TRADING ---


@api_blueprint.route("/pipeline/run", methods=["POST"])
@require_ip_whitelist
def trigger_eod_pipeline() -> ApiResponse:
    """Triggers a synchronous End-of-Day (EOD) pipeline run.

    Sequential Execution:
    TradeManager -> Screener Engine -> Order Generation -> Cache Pre-warming.

    Returns:
        ApiResponse: JSON status and execution summary.
    """
    try:
        logger.info("API Trigger: Manual EOD pipeline run initiated.")
        app_instance = cast(Any, current_app)._get_current_object()
        summary = run_daily_eod_pipeline(app_instance)
        return jsonify(summary), 200
    except Exception as error:
        logger.exception("Error during EOD pipeline run: %s", error)
        return jsonify({"status": "error", "error": str(error)}), 500


@api_blueprint.route("/screener/run", methods=["POST"])
@require_ip_whitelist
def trigger_screener() -> ApiResponse:
    """Triggers a manual run of all active screeners.

    Returns:
        ApiResponse: JSON status success with statistics.
    """
    try:
        days_lookback = request.args.get("days", default=0, type=int)
        target_strategy = request.args.get("strategy", type=str)

        screener_engine = current_app.extensions.get("screener_engine")
        if not screener_engine:
            return jsonify({"error": "Screener Engine not initialized"}), 503

        logger.info(
            "API Trigger: Screener run (days=%d, strategy=%s)",
            days_lookback,
            target_strategy,
        )
        statistics = screener_engine.run_all(
            days=days_lookback, strategy_filter=target_strategy
        )

        return jsonify({"status": "success", "stats": statistics}), 200
    except Exception as error:
        logger.exception("Error during screener run: %s", error)
        return jsonify({"error": str(error)}), 500


def _resolve_strategy_enum(strategy_name: str) -> Strategies | None:
    """Resolves a strategy name or alias into a canonical Strategies enum member."""
    raw_name = strategy_name.lower().strip()
    canonical_name = raw_name.replace("-", "_")
    resolved = STRATEGY_ALIASES.get(raw_name) or STRATEGY_ALIASES.get(canonical_name)
    if resolved:
        return resolved
    for item in Strategies:
        if (
            item.value.lower().replace("-", "_") == canonical_name
            or item.name.lower() == canonical_name
        ):
            return item
    return None


def _get_active_strategy(
    strategy_enum: Strategies, display_name: str | None = None
) -> tuple[Any | None, ApiResponse | None]:
    """Retrieves an active strategy from the screener engine extension."""
    screener_engine = current_app.extensions.get("screener_engine")
    if not screener_engine:
        return None, (
            jsonify({"status": "error", "message": "Engine not initialized"}),
            503,
        )
    strategy = screener_engine.get_strategy(strategy_enum)
    if not strategy:
        name_str = display_name or strategy_enum.name
        return None, (
            jsonify(
                {
                    "status": "error",
                    "message": f"Strategy '{name_str}' not initialized in engine"
                    if display_name
                    else f"{name_str} strategy not found",
                }
            ),
            404,
        )
    return strategy, None


@api_blueprint.route("/screener/run/<strategy_name>", methods=["POST"])
@require_ip_whitelist
def run_strategy_screener(strategy_name: str) -> ApiResponse:
    """Executes screening for a specific strategy by strategy_name parameter.

    Args:
        strategy_name: Name or alias of the strategy.

    Returns:
        ApiResponse: JSON with signal status.
    """
    try:
        analysis_date = request.args.get("date")
        days = request.args.get("days", default=0, type=int)
        strategy_enum = _resolve_strategy_enum(strategy_name)

        if not strategy_enum:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Strategy '{strategy_name}' not found",
                    }
                ),
                404,
            )

        strategy, err_resp = _get_active_strategy(
            strategy_enum, display_name=strategy_name
        )
        if err_resp or strategy is None:
            return err_resp or (jsonify({"status": "error"}), 500)

        candidates_count = strategy.run(days=days, analysis_date=analysis_date)
        return (
            jsonify(
                {
                    "status": "success",
                    "strategy": strategy_name.lower().replace("-", "_"),
                    "signals_found": candidates_count,
                }
            ),
            200,
        )
    except Exception as error:
        logger.exception("Error analyzing strategy '%s': %s", strategy_name, error)
        return jsonify({"status": "error", "message": str(error)}), 500


def _debug_single_symbol(strategy_enum: Strategies) -> ApiResponse:
    """Helper for single-symbol debug analysis on strategies implementing analyze_single_symbol."""
    try:
        symbol = _extract_symbol_from_request()
        if not symbol:
            message = (
                "Symbol required (query param or JSON)"
                if strategy_enum == Strategies.DipBuyer
                else "Symbol required"
            )
            return jsonify({"status": "error", "message": message}), 400

        strategy, err_resp = _get_active_strategy(strategy_enum)
        if err_resp or strategy is None:
            return err_resp or (jsonify({"status": "error"}), 500)

        analysis_result = strategy.analyze_single_symbol(symbol)
        return jsonify(analysis_result), 200

    except Exception as error:
        logger.exception("Error analyzing symbol: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/dip-buyer", methods=["POST"])
@require_ip_whitelist
def analyze_dip_buyer() -> ApiResponse:
    """Detailed debugging for DipBuyer strategy on a single symbol.

    Returns:
        ApiResponse: JSON analysis result or error.
    """
    return _debug_single_symbol(Strategies.DipBuyer)


@api_blueprint.route("/screener/turnover", methods=["POST"])
@require_ip_whitelist
def analyze_turnover() -> ApiResponse:
    """Detailed debugging for TurnoverTiming strategy on a single symbol.

    Returns:
        ApiResponse: JSON analysis result or error.
    """
    return _debug_single_symbol(Strategies.TurnOverTiming)


@api_blueprint.route("/screener/croc", methods=["POST"])
@require_ip_whitelist
def analyze_croc() -> ApiResponse:
    """Returns the full list of recommended signals for CrocSetup.

    Returns:
        ApiResponse: JSON analysis result or error.
    """
    try:
        days_lookback = request.args.get("days", default=0, type=int)
        analysis_date = request.args.get("date")

        strategy, err_resp = _get_active_strategy(Strategies.CrocSetup)
        if err_resp or strategy is None:
            return err_resp or (jsonify({"status": "error"}), 500)

        signals = strategy.get_all_recommendations(
            days=days_lookback, analysis_date=analysis_date
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "date": analysis_date,
                    "days_lookback": days_lookback,
                    "signals": signals,
                }
            ),
            200,
        )

    except Exception as error:
        logger.exception("Error analyzing Croc setup: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/screener/ndx-momentum", methods=["POST"])
@require_ip_whitelist
def analyze_ndx_momentum() -> ApiResponse:
    """Returns the current market regime and top momentum leaders for NDX.

    Returns:
        ApiResponse: JSON with regime status and top symbols.
    """
    try:
        analysis_date = request.args.get("date")

        strategy, err_resp = _get_active_strategy(Strategies.NDXMomentum)
        if err_resp or strategy is None:
            return err_resp or (jsonify({"status": "error"}), 500)

        analysis = strategy.calculate_analysis(
            analysis_date=analysis_date, force_run=True
        )

        return (
            jsonify(
                {
                    "status": "success",
                    "date": analysis.get("date"),
                    "requested_date": analysis.get("requested_date"),
                    "is_rebalance_day": analysis.get("is_rebalance_day", False),
                    "regime": analysis.get("regime_indicators"),
                    "top_leaders": analysis.get("top_symbols", []),
                    "error": analysis.get("error"),
                }
            ),
            200,
        )

    except Exception as error:
        logger.exception("Error analyzing NDX Momentum: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


def _dispatch_strategy_debug(
    strategy: Any,
    strategy_name: str,
    symbol: str | None,
    days_lookback: int,
    analysis_date: str | None,
) -> ApiResponse:
    """Invokes available debug inspection method on strategy instance."""
    if symbol and hasattr(strategy, "analyze_single_symbol"):
        return jsonify(strategy.analyze_single_symbol(symbol)), 200

    if hasattr(strategy, "get_all_recommendations"):
        signals = strategy.get_all_recommendations(
            days=days_lookback, analysis_date=analysis_date
        )
        return (
            jsonify(
                {
                    "status": "success",
                    "date": analysis_date,
                    "days_lookback": days_lookback,
                    "signals": signals,
                }
            ),
            200,
        )

    if hasattr(strategy, "calculate_analysis"):
        analysis = strategy.calculate_analysis(
            analysis_date=analysis_date, force_run=True
        )
        return jsonify({"status": "success", **analysis}), 200

    return (
        jsonify(
            {
                "status": "error",
                "message": f"No debug inspection method available for '{strategy_name}'",
            }
        ),
        400,
    )


@api_blueprint.route("/screener/<strategy_name>/debug", methods=["POST"])
@require_ip_whitelist
def debug_strategy(strategy_name: str) -> ApiResponse:
    """Unified debug and inspection endpoint for any screener strategy.

    Args:
        strategy_name: Name or alias of strategy.

    Returns:
        ApiResponse: JSON debug inspection data or error.
    """
    strategy_enum = _resolve_strategy_enum(strategy_name)
    if not strategy_enum:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Strategy '{strategy_name}' not found",
                }
            ),
            404,
        )

    strategy, err_resp = _get_active_strategy(strategy_enum, display_name=strategy_name)
    if err_resp or strategy is None:
        return err_resp or (jsonify({"status": "error"}), 500)

    try:
        symbol = _extract_symbol_from_request()
        days_lookback = request.args.get("days", default=0, type=int)
        analysis_date = request.args.get("date")
        return _dispatch_strategy_debug(
            strategy=strategy,
            strategy_name=strategy_name,
            symbol=symbol,
            days_lookback=days_lookback,
            analysis_date=analysis_date,
        )
    except Exception as error:
        logger.exception("Error debugging strategy '%s': %s", strategy_name, error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/orders/generate", methods=["POST"])
@require_ip_whitelist
def trigger_orders() -> ApiResponse:
    """Triggers the daily order generation process.

    Returns:
        Response: JSON status with generated file path.
    """
    trade_manager = current_app.extensions.get("trade_manager")
    if not trade_manager:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500
    try:
        order_file_path = trade_manager.generate_daily_orders()
        if order_file_path:
            return jsonify({"status": "success", "file": order_file_path}), 201

        return jsonify({"status": "success", "message": "No orders generated"}), 200
    except Exception as error:
        logger.error("Order generation failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


# --- TRADES & BACKFILL ---


@api_blueprint.route("/trades/backfill", methods=["POST"])
@require_ip_whitelist
def trigger_trades_backfill() -> ApiResponse:
    """Triggers a backfill or retry of trade processing.

    If strategy query parameter is specified, runs backfill for that strategy.

    Returns:
        ApiResponse: JSON confirmation.
    """
    strategy = (request.args.get("strategy") or "").lower()

    if strategy:
        return execute_strategy_backfill(strategy)

    trade_manager = current_app.extensions.get("trade_manager")
    if not trade_manager:
        return jsonify({"status": "error", "message": "TradeManager missing"}), 500

    try:
        trade_manager.run_daily_process()
        return jsonify(
            {
                "status": "success",
                "info": "Backfill executed via daily process",
            }
        )
    except Exception as error:
        logger.error("Trades backfill failed: %s", error)
        return jsonify({"status": "error", "message": str(error)}), 500


@api_blueprint.route("/trades/backfill/<strategy_name>", methods=["POST"])
@require_ip_whitelist
def execute_strategy_backfill(strategy_name: str) -> ApiResponse:
    """Executes historical backfill simulation for a given strategy_name.

    Args:
        strategy_name: Identifier of strategy (e.g. 'tgim', 'bridge_scout', 'bounce_bandit').

    Returns:
        ApiResponse: JSON containing backfill summary metrics and trades list.
    """
    canonical_strategy = strategy_name.lower().replace("-", "_")
    start_date = (
        request.args.get("start_date") or request.args.get("start") or "2025-01-01"
    )
    end_date = request.args.get("end_date") or request.args.get("end")
    budget = float(request.args.get("budget") or 10000.0)
    raw_clear = request.args.get("clear_existing") or request.args.get("clear")
    clear_existing = _parse_boolean_parameter(raw_clear, default_value=True)

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    stocks_session = DatabaseSession(str(configuration.get_path("stocks")))
    signals_session = DatabaseSession(str(configuration.get_path("signals")))

    try:
        result = run_strategy_backfill(
            stocks_session=stocks_session,
            signals_session=signals_session,
            strategy_name=canonical_strategy,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            clear_existing=clear_existing,
        )
        try:
            cache.clear()
        except Exception as cache_err:
            logger.debug("Cache clear skipped: %s", cache_err)
        return jsonify({"status": "success", "result": result})
    except ValueError as validation_error:
        logger.warning(
            "Strategy '%s' backfill validation error: %s",
            strategy_name,
            validation_error,
        )
        return jsonify({"status": "error", "message": str(validation_error)}), 400
    except Exception as error:
        logger.error("Strategy '%s' backfill failed: %s", strategy_name, error)
        return jsonify({"status": "error", "message": str(error)}), 500


# --- MARKET DATA ---


def _parse_market_update_params() -> tuple[str, bool]:
    json_data: dict[str, object] = request.get_json(silent=True) or {}

    provider_param = request.args.get("provider") or json_data.get("provider") or "auto"
    provider_mode = str(provider_param).lower()

    raw_ignore: object = request.args.get("ignore_today")
    if raw_ignore is None:
        raw_ignore = json_data.get("ignore_today")

    if isinstance(raw_ignore, bool):
        ignore_today = raw_ignore
    else:
        ignore_today = str(raw_ignore).lower() in ("true", "1", "yes")

    return provider_mode, ignore_today


@api_blueprint.route("/market/sync", methods=["POST"])
@require_ip_whitelist
def sync_market_data() -> ApiResponse:
    """Triggers background market data synchronization.

    Returns:
        ApiResponse: JSON status accepted or 409 if sync is already running.
    """
    if _market_sync_lock.locked():
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Market synchronization already in progress",
                }
            ),
            409,
        )

    should_full_sync = request.args.get("full", "false").lower() == "true"
    provider_mode, ignore_today = _parse_market_update_params()

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    database_path = configuration.get_db_path("stocks")
    signals_database_path = configuration.get_db_path("signals")
    telegram_bot = current_app.extensions.get("telegram")
    app = cast(Any, current_app)._get_current_object()

    def _execute_sync_task() -> None:
        """Background task for market synchronization."""
        if not _market_sync_lock.acquire(blocking=False):
            logger.warning("Market sync skipped: Synchronization already running")
            return
        try:
            with app.app_context():
                try:
                    market_session = DatabaseSession(str(database_path))
                    signals_session = DatabaseSession(str(signals_database_path))
                    updater = MarketDataUpdater(market_session, signals_session)
                    updater.run_update(
                        full_reload=should_full_sync,
                        provider_mode=provider_mode,
                        ignore_today=ignore_today,
                    )

                    quality_service = MarketQualityService(
                        updater, telegram_bot=telegram_bot
                    )
                    quality_service.perform_gap_check()
                    quality_service.check_last_trading_day_completeness()
                except (RuntimeError, ValueError) as sync_error:
                    logger.error("Market Sync Task Error: %s", sync_error)
        finally:
            _market_sync_lock.release()

    Thread(target=_execute_sync_task, daemon=True).start()
    return jsonify({"status": "accepted", "message": "Sync started"}), 202


@api_blueprint.route("/market/reload", methods=["POST"])
@require_ip_whitelist
def reload_market_data() -> ApiResponse:
    """Triggers a full manual reload of market data in the background.

    Returns:
        ApiResponse: JSON status queued or 409 if sync is already running.
    """
    if _market_sync_lock.locked():
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Market synchronization already in progress",
                }
            ),
            409,
        )

    provider_mode, ignore_today = _parse_market_update_params()

    configuration = current_app.config.get("APP_CONFIG")
    if not configuration:
        return jsonify({"status": "error", "message": "Configuration missing"}), 500

    database_path = configuration.get_db_path("stocks")
    signals_database_path = configuration.get_db_path("signals")
    telegram_bot = current_app.extensions.get("telegram")
    app = cast(Any, current_app)._get_current_object()

    def _execute_reload_task() -> None:
        """Background task for full market reload."""
        if not _market_sync_lock.acquire(blocking=False):
            logger.warning("Market reload skipped: Synchronization already running")
            return
        try:
            with app.app_context():
                try:
                    logger.info("Manual full reload via API started...")
                    market_session = DatabaseSession(str(database_path))
                    signals_session = DatabaseSession(str(signals_database_path))
                    updater = MarketDataUpdater(market_session, signals_session)
                    updater.run_update(
                        full_reload=True,
                        provider_mode=provider_mode,
                        ignore_today=ignore_today,
                    )

                    quality_service = MarketQualityService(
                        updater, telegram_bot=telegram_bot
                    )
                    quality_service.perform_gap_check()
                    quality_service.check_last_trading_day_completeness()
                except (RuntimeError, ValueError) as reload_error:
                    logger.error("Market Reload Task Error: %s", reload_error)
        finally:
            _market_sync_lock.release()

    Thread(target=_execute_reload_task, daemon=True).start()
    return jsonify({"status": "queued", "message": "Full reload triggered"}), 200
