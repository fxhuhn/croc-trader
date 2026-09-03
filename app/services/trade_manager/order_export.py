import csv
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ...config import settings
from ...const import Strategies
from ...models import Order, OrderLeg
from ...types import TradeData

logger = logging.getLogger(__name__)


def _get_override_for_symbol(
    symbol: str, overrides: dict[str, Any] | None = None
) -> dict[str, str]:
    """Retrieves order configuration overrides for a given symbol.

    Args:
        symbol: The original symbol to look up.
        overrides: Optional override mapping. If omitted, fetched from settings.app.

    Returns:
        dict[str, str]: The dictionary of overrides, or empty if none found.
    """
    if overrides is None:
        raw_overrides = getattr(settings.app, "order_overrides", {})
        overrides = raw_overrides if isinstance(raw_overrides, dict) else {}

    val = overrides.get(symbol, {})
    if isinstance(val, dict):
        return cast(dict[str, str], val)
    return {}


def _resolve_output_directory(output_directory: Path | None = None) -> Path:
    """Resolves target orders directory, prioritizing explicit parameter over settings."""
    if output_directory is not None:
        return output_directory

    database_config = getattr(settings.app, "database", None)
    if database_config is not None:
        folders = getattr(database_config, "folders", {})
        if isinstance(folders, dict) and "orders" in folders:
            orders_folder = str(folders["orders"])
            orders_path = Path(orders_folder)
            if orders_path.is_absolute():
                return orders_path
            base_folder = str(getattr(database_config, "base_folder", "data"))
            if orders_folder.startswith(base_folder):
                return orders_path
            return Path(base_folder) / orders_folder

    return Path("data/orders")


# Strategies whose orders are written to the daily CSV export.
# SplitTarget and HoldTarget are managed via the broker UI directly
# and are therefore intentionally excluded.
_CSV_SUPPORTED_STRATEGIES: frozenset[Strategies] = frozenset(
    {
        Strategies.NDXMomentum,
        Strategies.TurnOverTiming,
        Strategies.TurnOverTiming_10,
        Strategies.TurnOverTiming_05,
        Strategies.TwoPercent,
        Strategies.DipBuyer,
        Strategies.BounceBandit,
    }
)

_STRATEGY_DISPLAY_NAMES: dict[Strategies, str] = {
    Strategies.NDXMomentum: "NDXMomentum",
    Strategies.TurnOverTiming: "TurnoverTiming",
    Strategies.TurnOverTiming_10: "TurnoverTiming_1.0",
    Strategies.TurnOverTiming_05: "TurnoverTiming_0.5",
    Strategies.DipBuyer: "DipBuyer",
    Strategies.TwoPercent: "TwoPercent",
    Strategies.HoldTarget: "HoldTarget",
    Strategies.SplitTarget: "SplitTarget",
    Strategies.BounceBandit: "BounceBandit",
}

CSV_ORDER_HEADER: tuple[str, ...] = (
    "trade_group_id",
    "bracket_role",
    "symbol",
    "sec_type",
    "exchange",
    "account_id",
    "action",
    "quantity",
    "order_type",
    "target_price",
    "tif",
    "strategy_name",
    "currency",
)


def get_strategy_display_name(strategy_enum: Strategies) -> str:
    """Returns the standardized display name of a strategy for order reporting.

    Falls back to the raw enum value for any strategy not listed in the
    display-name table — new strategies are covered automatically.

    Args:
        strategy_enum: The resolved Strategies enum member.

    Returns:
        str: Human-readable display name used in CSV output.
    """
    return _STRATEGY_DISPLAY_NAMES.get(strategy_enum, str(strategy_enum.value))


def write_csv_orders_file(
    orders_data: Sequence[tuple[TradeData | dict[str, object], Order]],
    date_string: str,
    ibkr_account_id: str,
    resolve_strategy_fn: Callable[[str], Strategies | None],
    output_directory: Path | None = None,
) -> Path | None:
    """Transforms and saves generated orders to a CSV file in bracket layout."""
    filtered_orders_data: list[
        tuple[TradeData | dict[str, object], Order, Strategies]
    ] = []
    for trade, order in orders_data:
        resolved_strategy = resolve_strategy_fn(str(trade.get("strategy", "")))
        if resolved_strategy in _CSV_SUPPORTED_STRATEGIES:
            filtered_orders_data.append((trade, order, resolved_strategy))

    if not filtered_orders_data:
        logger.info("No orders found for CSV-supported strategies.")
        return None

    csv_rows = []

    for trade, order, resolved_strategy in filtered_orders_data:
        strategy_display_name = get_strategy_display_name(resolved_strategy)
        trade_database_id = trade.get("id")

        # Resolve override for symbol once
        override = _get_override_for_symbol(order.symbol)
        resolved_symbol = override.get("target_symbol", order.symbol)
        trade_group_id = (
            f"{trade_database_id}_{strategy_display_name}_{resolved_symbol}"
        )

        rows = map_order_to_csv_rows(
            trade,
            order,
            trade_group_id,
            strategy_display_name,
            ibkr_account_id,
        )
        csv_rows.extend(rows)

    if not csv_rows:
        return None

    target_directory = _resolve_output_directory(output_directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    csv_filename = f"orders_{date_string.replace('-', '_')}.csv"
    csv_file_path = target_directory / csv_filename

    with open(csv_file_path, "w", newline="") as csv_file_handle:
        writer = csv.DictWriter(csv_file_handle, fieldnames=list(CSV_ORDER_HEADER))
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("CSV Orders saved to: %s", csv_file_path)
    return csv_file_path


@dataclass(frozen=True)
class OrderExportContext:
    """Context parameters for formatting CSV order rows."""

    trade_group_id: str
    strategy_display_name: str
    ibkr_account_id: str
    symbol: str
    override: dict[str, str]


def _format_leg_row(
    context: OrderExportContext,
    leg: OrderLeg,
    bracket_role: str,
    fallback_quantity: int,
) -> dict[str, object]:
    """Formats an individual OrderLeg into a 13-column CSV row dictionary."""
    return {
        "trade_group_id": context.trade_group_id,
        "bracket_role": bracket_role,
        "symbol": context.symbol,
        "sec_type": context.override.get("sec_type", "STK"),
        "exchange": context.override.get("exchange", "SMART"),
        "account_id": context.ibkr_account_id,
        "action": leg.action,
        "quantity": leg.quantity if leg.quantity is not None else fallback_quantity,
        "order_type": leg.type,
        "target_price": f"{leg.price:.2f}",
        "tif": leg.time_in_force,
        "strategy_name": context.strategy_display_name,
        "currency": context.override.get("currency", ""),
    }


def map_order_to_csv_rows(
    trade: TradeData | dict[str, object],
    order: Order,
    trade_group_id: str,
    strategy_display_name: str,
    ibkr_account_id: str,
) -> list[dict[str, object]]:
    """Maps an order model and its legs to structured CSV row dictionaries."""
    override = _get_override_for_symbol(order.symbol)
    symbol = override.get("target_symbol", order.symbol)
    context = OrderExportContext(
        trade_group_id=trade_group_id,
        strategy_display_name=strategy_display_name,
        ibkr_account_id=ibkr_account_id,
        symbol=symbol,
        override=override,
    )

    rows: list[dict[str, object]] = []
    if order.entry:
        rows.append(
            _format_leg_row(
                context=context,
                leg=order.entry,
                bracket_role="ENTRY",
                fallback_quantity=order.quantity,
            )
        )

    for exit_leg in order.exits:
        bracket_role = (
            "EXIT"
            if order.entry is None
            else ("SL" if exit_leg.type == "STP" else "TP")
        )
        rows.append(
            _format_leg_row(
                context=context,
                leg=exit_leg,
                bracket_role=bracket_role,
                fallback_quantity=order.quantity,
            )
        )

    return rows
