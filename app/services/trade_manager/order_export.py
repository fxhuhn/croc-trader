import csv
import logging
from pathlib import Path
from ...const import Strategies
from ...models import Order

logger = logging.getLogger(__name__)

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
    orders_data: list[tuple[dict[str, object], Order]],
    date_string: str,
    ibkr_account_id: str,
    resolve_strategy_fn: callable,
) -> Path | None:
    """Transforms and saves generated orders to a CSV file in bracket layout."""
    filtered_orders_data: list[tuple[dict[str, object], Order, Strategies]] = []
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
        symbol = str(trade.get("symbol", ""))
        trade_group_id = f"{trade_database_id}_{strategy_display_name}_{symbol}"

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
    ]

    with open(csv_file_path, "w", newline="") as csv_file_handle:
        writer = csv.DictWriter(csv_file_handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info("CSV Orders saved to: %s", csv_file_path)
    return csv_file_path


def map_order_to_csv_rows(
    trade: dict[str, object],
    order: Order,
    trade_group_id: str,
    strategy_display_name: str,
    ibkr_account_id: str,
) -> list[dict[str, object]]:
    """Maps an order model and its legs to structured CSV row dictionaries."""
    rows = []
    if order.entry:
        entry_leg = order.entry
        rows.append(
            {
                "trade_group_id": trade_group_id,
                "bracket_role": "ENTRY",
                "symbol": order.symbol,
                "sec_type": "STK",
                "exchange": "SMART",
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
                "symbol": order.symbol,
                "sec_type": "STK",
                "exchange": "SMART",
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
            }
        )

    return rows
