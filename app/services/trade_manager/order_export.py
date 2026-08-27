import csv
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from ...config import settings
from ...const import Strategies
from ...models import Order
from ...types import TradeData

logger = logging.getLogger(__name__)


def _get_override_for_symbol(symbol: str) -> dict[str, str]:
    """Retrieves order configuration overrides for a given symbol.

    Args:
        symbol: The original symbol to look up.

    Returns:
        dict[str, str]: The dictionary of overrides, or empty if none found.
    """
    overrides = getattr(settings.app, "order_overrides", {})
    if isinstance(overrides, dict):
        val = overrides.get(symbol, {})
        if isinstance(val, dict):
            return cast(dict[str, str], val)
    return {}


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
}


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

        # Resolve override for symbol
        override = _get_override_for_symbol(order.symbol)
        resolved_symbol = override.get("target_symbol", order.symbol)
        trade_group_id = (
            f"{trade_database_id}_{strategy_display_name}_{resolved_symbol}"
        )

        rows = map_order_to_csv_rows(
            trade, order, trade_group_id, strategy_display_name, ibkr_account_id
        )
        csv_rows.extend(rows)

    if not csv_rows:
        return None

    output_directory = Path("data/orders")
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_filename = f"orders_{date_string.replace('-', '_')}.csv"
    csv_file_path = output_directory / csv_filename

    header = [
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
    ]

    with open(csv_file_path, "w", newline="") as csv_file_handle:
        writer = csv.DictWriter(csv_file_handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("CSV Orders saved to: %s", csv_file_path)
    return csv_file_path


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
    sec_type = override.get("sec_type", "STK")
    exchange = override.get("exchange", "SMART")
    currency = override.get("currency", "")

    rows = []
    if order.entry:
        entry_leg = order.entry
        rows.append(
            {
                "trade_group_id": trade_group_id,
                "bracket_role": "ENTRY",
                "symbol": symbol,
                "sec_type": sec_type,
                "exchange": exchange,
                "account_id": ibkr_account_id,
                "action": entry_leg.action,
                "quantity": (
                    entry_leg.quantity
                    if entry_leg.quantity is not None
                    else order.quantity
                ),
                "order_type": entry_leg.type,
                "target_price": f"{entry_leg.price:.2f}",
                "tif": entry_leg.time_in_force,
                "strategy_name": strategy_display_name,
                "currency": currency,
            }
        )

    for exit_leg in order.exits:
        if order.entry is None:
            bracket_role = "EXIT"
        else:
            bracket_role = "SL" if exit_leg.type == "STP" else "TP"
        rows.append(
            {
                "trade_group_id": trade_group_id,
                "bracket_role": bracket_role,
                "symbol": symbol,
                "sec_type": sec_type,
                "exchange": exchange,
                "account_id": ibkr_account_id,
                "action": exit_leg.action,
                "quantity": (
                    exit_leg.quantity
                    if exit_leg.quantity is not None
                    else order.quantity
                ),
                "order_type": exit_leg.type,
                "target_price": f"{exit_leg.price:.2f}",
                "tif": exit_leg.time_in_force,
                "strategy_name": strategy_display_name,
                "currency": currency,
            }
        )

    return rows
