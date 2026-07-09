import asyncio
import json
import logging
import secrets
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# NEU: AccountValue importiert
from ib_async import (
    IB,
    AccountValue,
    Contract,
    Order,
    PortfolioItem,
    Stock,
    Trade,
    util,
)

# Konfiguration
TWS_HOST: str = "127.0.0.1"
TWS_PORT: int = 7496
CLIENT_ID: int = 1
POSITIONS_FILE: Path = Path("positions.json")
ORDERS_FILE: Path = Path("orders.json")
HISTORY_FILE: Path = Path("orders_history.json")


def json_serializer(obj: Any) -> str | int | float:
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    return str(obj)


def save_data_to_json(data: list[Any], file_path: Path) -> None:
    """Overwrites a JSON file with new data."""
    tree_data = [util.tree(item) for item in data]
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(tree_data, json_file, indent=4, default=json_serializer)
    print(f"Successfully saved {len(data)} items to {file_path}")


def update_and_save_history(new_trades: list[Trade], file_path: Path) -> None:
    """Merges new completed trades with existing history in JSON file."""
    existing_data = []
    if file_path.exists():
        try:
            with file_path.open("r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []

    new_data_dicts = [util.tree(trade) for trade in new_trades]
    history_map = {}

    for record in existing_data:
        try:
            perm_id = record["Trade"]["order"]["Order"]["permId"]
            history_map[perm_id] = record
        except (KeyError, TypeError):
            continue

    for record in new_data_dicts:
        try:
            perm_id = record["Trade"]["order"]["Order"]["permId"]
            history_map[perm_id] = record
        except (KeyError, TypeError):
            continue

    merged_list = list(history_map.values())

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(merged_list, json_file, indent=4, default=json_serializer)

    print(
        f"Successfully updated history in {file_path}. Total records: {len(merged_list)}"
    )


def display_account_summary(account_values: list[AccountValue]) -> None:
    """Filters and prints key account metrics."""
    print(f"\n{'=' * 40}")
    print("ACCOUNT SUMMARY")
    print(f"{'=' * 40}")

    # Wir suchen nach spezifischen Keys.
    # Da accountValues oft Einträge für verschiedene Währungen enthält,
    # filtern wir bevorzugt nach 'USD' oder 'BASE' falls möglich,
    # oder geben einfach alle passenden Fundstücke aus.
    target_keys = {"NetLiquidation", "UnrealizedPnL", "RealizedPnL"}

    found_metrics = []

    for av in account_values:
        if av.tag in target_keys:
            # Wenn Währung vorhanden, mit anzeigen
            currency_suffix = f" {av.currency}" if av.currency else ""
            found_metrics.append((av.tag, av.value, currency_suffix))

    # Sortierung für schöne Ausgabe (optional)
    found_metrics.sort(key=lambda x: x[0])

    if found_metrics:
        for tag, val, curr in found_metrics:
            print(f"{tag:<20}: {val:>10}{curr}")
    else:
        print("No account metrics found yet (Wait for TWS sync).")
    print(f"{'=' * 40}\n")


async def fetch_and_process_data() -> None:
    """Main workflow."""
    ib_instance = IB()

    try:
        print(f"Connecting to TWS at {TWS_HOST}:{TWS_PORT}...")
        await ib_instance.connectAsync(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)
        print("Connection established.")

        # --- 1. Subscriptions ---
        print("Subscribing to account updates...")
        ib_instance.client.reqAccountUpdates(True, "")

        print("Requesting open orders...")
        ib_instance.client.reqAllOpenOrders()

        print("Requesting completed orders...")
        new_completed_trades: list[Trade] = await ib_instance.reqCompletedOrdersAsync(
            False
        )

        print("Waiting for data synchronization...")
        await asyncio.sleep(2)

        # --- 2. Retrieve Cache ---
        current_portfolio: list[PortfolioItem] = ib_instance.portfolio()
        all_open_trades: list[Trade] = ib_instance.openTrades()
        # NEU: Account Values abrufen
        account_vals: list[AccountValue] = ib_instance.accountValues()

        # --- 3. Save Data ---
        save_data_to_json(current_portfolio, POSITIONS_FILE)
        save_data_to_json(all_open_trades, ORDERS_FILE)
        update_and_save_history(new_completed_trades, HISTORY_FILE)

        # --- 4. Display ---
        display_portfolio_overview(current_portfolio, all_open_trades)

        # NEU: Account Summary am Ende
        display_account_summary(account_vals)

    except (ConnectionRefusedError, OSError) as error:
        logging.error(f"Failed to connect to TWS: {error}")
        print("Error: Could not connect to TWS. Is it running?")
    except Exception:
        logging.exception("An unexpected error occurred.")
    finally:
        if ib_instance.isConnected():
            ib_instance.disconnect()
            print("Disconnected from TWS.")


def display_portfolio_overview(
    portfolio_items: list[PortfolioItem], trades: list[Trade]
) -> None:
    """Displays portfolio items and unlinked open orders."""

    def format_price_info(order: Order) -> str:
        if order.lmtPrice and order.lmtPrice > 0:
            base = f"LMT {order.lmtPrice:.2f}"
            if order.auxPrice and order.auxPrice > 0:
                base += f" (Stp {order.auxPrice:.2f})"
            return base
        elif order.auxPrice and order.auxPrice > 0:
            return f"STP {order.auxPrice:.2f}"
        return "MKT"

    trades_by_con_id: dict[int, list[Trade]] = {}
    for trade in trades:
        if trade.contract and trade.contract.conId:
            trades_by_con_id.setdefault(trade.contract.conId, []).append(trade)

    linked_con_ids = set()

    if portfolio_items:
        print(f"\n--- Portfolio Overview ({len(portfolio_items)} Positions) ---")
        for item in portfolio_items:
            contract = item.contract
            con_id = contract.conId
            linked_con_ids.add(con_id)

            print(f"\nPosition: {contract.symbol} ({contract.secType})")
            print(f"  Quantity:       {item.position}")
            print(f"  Market Price:   {item.marketPrice:.2f}")
            print(f"  Avg Cost:       {item.averageCost:.2f}")
            print(f"  Market Value:   {item.marketValue:.2f}")
            print(f"  Unrealized PnL: {item.unrealizedPNL:.2f}")

            linked_trades = trades_by_con_id.get(con_id, [])
            if linked_trades:
                print(f"  -> Linked Open Orders ({len(linked_trades)}):")
                for trade in linked_trades:
                    order = trade.order
                    price_str = format_price_info(order)
                    print(
                        f"     * {order.action} {order.totalQuantity} @ {price_str} ({trade.orderStatus.status})"
                    )
            else:
                print("  -> No open orders for this position.")
    else:
        print("\n--- No Existing Positions ---")

    all_order_con_ids = set(trades_by_con_id.keys())
    unlinked_con_ids = all_order_con_ids - linked_con_ids

    if unlinked_con_ids:
        print("\n\n--- Pending Entry Orders (New Positions) ---")
        for con_id in unlinked_con_ids:
            trades_list = trades_by_con_id[con_id]
            if not trades_list:
                continue
            contract = trades_list[0].contract
            print(f"\nSymbol: {contract.symbol} ({contract.secType})")

            for trade in trades_list:
                order = trade.order
                price_str = format_price_info(order)
                print(
                    f"  * {order.action} {order.totalQuantity} @ {price_str} ({trade.orderStatus.status})"
                )


async def place_strategy_bracket_order(
    ib_instance: IB,
    symbol: str,
    qty: float,
    entry_stp_price: float,
    exit_lmt_price: float,
    exit_stp_price: float,
    strategy_name: str,
) -> Trade:
    """Platziert asynchron eine Stop-Buy Bracket Order (Entry: DAY, Exits: GTC).

    Args:
        ib_instance: Die aktive IB-Verbindung.
        symbol: Das Ticker-Symbol (z.B. "AAPL").
        qty: Die Menge der Aktien.
        entry_stp_price: Der Trigger-Preis für den Einstieg (Stop Buy).
        exit_stp_price: Der Preis für den Stop Loss.
        exit_lmt_price: Der Preis für den Take Profit.
        strategy_name: Wird im Feld 'orderRef' für das Tracking gespeichert.

    Returns:
        Trade: Das Trade-Objekt der Parent-Order (Entry).
    """

    # 1. Kontrakt definieren
    contract = Stock(symbol, "SMART", "USD")

    # KORREKTUR: Async qualifizieren, um den Event-Loop nicht zu blockieren
    await ib_instance.qualifyContractsAsync(contract)

    # 2. Parent Order (Entry Stop Buy)
    parent = Order()
    parent.action = "BUY"
    parent.orderType = "STP"
    parent.auxPrice = entry_stp_price
    parent.totalQuantity = qty
    parent.tif = "DAY"
    parent.orderRef = strategy_name
    parent.transmit = False  # Warten auf Kinder

    # PlaceOrder ist non-blocking und ok, aber wir müssen contract übergeben
    parent_trade = ib_instance.placeOrder(contract, parent)

    # Wichtig: Wir müssen kurz warten oder sicherstellen, dass die OrderId generiert wurde.
    # ib_async macht das meist sofort client-seitig, daher ist parent.orderId verfügbar.
    parent_id = parent.orderId

    # 3. Child Order: Stop Loss
    stop_loss = Order()
    stop_loss.action = "SELL"
    stop_loss.orderType = "STP"
    stop_loss.auxPrice = exit_stp_price
    stop_loss.totalQuantity = qty
    stop_loss.tif = "GTC"
    stop_loss.parentId = parent_id
    stop_loss.orderRef = strategy_name
    stop_loss.transmit = False

    # 4. Child Order: Take Profit
    take_profit = Order()
    take_profit.action = "SELL"
    take_profit.orderType = "LMT"
    take_profit.lmtPrice = exit_lmt_price
    take_profit.totalQuantity = qty
    take_profit.tif = "GTC"
    take_profit.parentId = parent_id
    take_profit.orderRef = strategy_name
    take_profit.transmit = True  # Jetzt alles senden

    # 5. Child Orders platzieren
    ib_instance.placeOrder(contract, stop_loss)
    ib_instance.placeOrder(contract, take_profit)

    return parent_trade


async def place_waterfall_exit_order(
    ib_instance: IB,
    symbol: str,
    qty: float,
    entry_lmt_price: float,
    strategy_name: str,
    exit_lmt_price: float = None,
    exit_lmt_on_close_price: float = None,
) -> Trade:
    """Erstellt eine Intraday-Order mit gestaffelten Verkaufs-Limits (Waterfall).

    Inklusive expliziter Zeitzonen-Angabe (US/Eastern) für IBKR Compliance.
    """

    # --- 1. Zeitberechnung (US Eastern Time) ---
    us_eastern = ZoneInfo("US/Eastern")
    now_in_ny = datetime.now(us_eastern)
    today_ny = now_in_ny.date()

    # Börsenschluss 16:00 Uhr ET
    market_close_dt = datetime.combine(today_ny, time(16, 0), tzinfo=us_eastern)

    # Switch-Point: 30 Minuten vor Schluss (15:30 Uhr ET)
    switch_point_dt = market_close_dt - timedelta(minutes=30)

    # Formatierung für IBKR (YYYYMMDD HH:mm:ss US/Eastern)
    # KORREKTUR: Wir hängen die Zeitzone explizit als Text an.
    # Das Format muss exakt so aussehen: "20260107 15:30:00 US/Eastern"
    base_time_str = switch_point_dt.strftime("%Y%m%d %H:%M:%S")
    ib_time_string = f"{base_time_str} US/Eastern"

    print(f"DEBUG: Switch-Zeit für IBKR: '{ib_time_string}'")

    # --- 2. Contract ---
    contract = Stock(symbol, "SMART", "USD")
    await ib_instance.qualifyContractsAsync(contract)

    # --- 3. Parent Order (Entry) ---
    parent = Order()
    parent.action = "BUY"
    parent.orderType = "LMT"
    parent.lmtPrice = entry_lmt_price
    parent.totalQuantity = qty
    parent.tif = "DAY"
    parent.orderRef = strategy_name
    parent.transmit = False

    parent_trade = ib_instance.placeOrder(contract, parent)
    parent_id = parent.orderId

    oca_group_name = f"WATERFALL_{parent_id}"

    # --- 4. Exit A: Das "Hohe Ziel" (Phase 1) ---
    if exit_lmt_price:
        tp1 = Order()
        tp1.action = "SELL"
        tp1.orderType = "LMT"
        tp1.lmtPrice = exit_lmt_price
        tp1.totalQuantity = qty
        tp1.parentId = parent_id
        tp1.orderRef = strategy_name

        tp1.tif = "GTD"
        tp1.goodTillDate = ib_time_string  # Jetzt inkl. " US/Eastern"

        tp1.ocaGroup = oca_group_name
        tp1.ocaType = 1
        tp1.transmit = False

    # --- 5. Exit B: Das "On Close Ziel" (Phase 2) ---
    if exit_lmt_on_close_price:
        tp2 = Order()
        tp2.action = "SELL"
        tp2.orderType = "LOC"
        tp2.lmtPrice = exit_lmt_on_close_price
        tp2.totalQuantity = qty
        tp2.parentId = parent_id
        tp2.orderRef = strategy_name

        tp2.tif = "DAY"
        tp2.goodAfterTime = ib_time_string  # Jetzt inkl. " US/Eastern"

        tp2.ocaGroup = oca_group_name
        tp2.ocaType = 3
        tp2.transmit = True

    # --- 6. Platzieren ---
    if exit_lmt_price:
        ib_instance.placeOrder(contract, tp1)
    if exit_lmt_on_close_price:
        ib_instance.placeOrder(contract, tp2)

    return parent_trade


async def attach_waterfall_exit_to_existing_position(
    ib_instance: IB,
    con_id: int,  # WICHTIG: Die ID aus deinem JSON (481863646)
    symbol: str,  # "APP"
    qty: float,  # 3.0
    exit_lmt_price: float,  # Das "hohe" Ziel (z.B. 630.00)
    exit_lmt_on_close_price: float,  # Das "sichere" Ziel ab 15:30 US-Zeit
    strategy_name: str,
) -> list[Trade]:
    """
    Erstellt nachträglich Waterfall-Exits für eine existierende Position.
    Nutzt OCA (One Cancels All), um Doppelverkäufe zu verhindern.
    """

    # --- 1. Zeitberechnung (US Eastern Time) ---
    us_eastern = ZoneInfo("US/Eastern")
    now_in_ny = datetime.now(us_eastern)
    today_ny = now_in_ny.date()

    # Börsenschluss 16:00 Uhr ET / Switch 15:30 Uhr ET
    market_close_dt = datetime.combine(today_ny, time(16, 0), tzinfo=us_eastern)
    switch_point_dt = market_close_dt - timedelta(minutes=30)

    # Format: "20260107 15:30:00 US/Eastern"
    base_time_str = switch_point_dt.strftime("%Y%m%d %H:%M:%S")
    ib_time_string = f"{base_time_str} US/Eastern"

    print(f"DEBUG: Switch-Point für '{symbol}': {ib_time_string}")

    # --- 2. Contract definieren ---
    # Wir nutzen die conId für maximale Präzision
    contract = Contract()
    contract.conId = con_id
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    await ib_instance.qualifyContractsAsync(contract)

    # --- 3. OCA Gruppe erstellen ---
    # Da wir keine Parent-ID haben, erzeugen wir einen Zufallsschlüssel,
    # der die beiden Orders logisch verbindet.
    oca_token = secrets.SystemRandom().randint(100000, 999999)
    oca_group_name = f"EXIT_MANUAL_{symbol}_{oca_token}"

    # --- 4. Order A: Hohes Limit (Bis 15:30 US-Zeit) ---
    order_a = Order()
    order_a.action = "SELL"
    order_a.orderType = "LMT"
    order_a.lmtPrice = exit_lmt_price
    order_a.totalQuantity = qty  # Volle 3.0 Stück
    order_a.orderRef = strategy_name

    # Zeitsteuerung: Stirbt um 15:30
    order_a.tif = "GTD"
    order_a.goodTillDate = ib_time_string

    # Verknüpfung
    order_a.ocaGroup = oca_group_name
    order_a.ocaType = 1  # Cancel all other orders in group
    order_a.transmit = False

    # --- 5. Order B: Close Limit (Ab 15:30 US-Zeit) ---
    order_b = Order()
    order_b.action = "SELL"
    order_b.orderType = "LOC"
    order_b.lmtPrice = exit_lmt_on_close_price
    order_b.totalQuantity = qty  # Volle 3.0 Stück
    order_b.orderRef = strategy_name

    # Zeitsteuerung: Wacht um 15:30 auf
    order_b.tif = "DAY"
    order_b.goodAfterTime = ib_time_string

    # Verknüpfung
    order_b.ocaGroup = oca_group_name
    order_b.ocaType = 3
    order_b.transmit = True  # Jetzt beide scharf schalten

    # --- 6. Platzieren ---
    trade_a = ib_instance.placeOrder(contract, order_a)
    trade_b = ib_instance.placeOrder(contract, order_b)

    return [trade_a, trade_b]


async def main():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=1)
    """
    try:
        # Daten aus deinem JSON:
        # conId: 481863646
        # symbol: APP
        # position: 3.0
        # Aktueller Preis (ca): 614.00 (aus deinem Log)

        trades = await attach_waterfall_exit_to_existing_position(
            ib_instance=ib,
            con_id=115441080,
            symbol="FANG",
            qty=14.0,
            # Szenario:
            exit_lmt_price=143.55,  # Versuch 1: Teuer verkaufen (+3%)
            exit_lmt_on_close_price=144.96,  # Versuch 2: Sicher verkaufen (nahe Einstand)
            strategy_name="dip_buyer_FANG",
        )

        print(f"Orders platziert für FANG. Status A: {trades[0].orderStatus.status}")

    except Exception as e:
        print(f"Fehler: {e}")

    await asyncio.sleep(1)
    ib.disconnect()
    """
    # BEISPIEL:
    # Wir wollen bis 22:00 Uhr handeln.
    # Ab 21:30 Uhr (30 Min vorher) soll der Limit-Versuch abgebrochen
    # und bestens (Market) verkauft werden.

    try:
        trade = await place_waterfall_exit_order(
            ib_instance=ib,
            symbol="FSLR",
            qty=8,
            entry_lmt_price=225.89,
            # exit_lmt_price=143.51,  # Optimistisches Ziel
            exit_lmt_on_close_price=255.93,  # Konservatives Ziel zum Schluss
            strategy_name="DiscountSniper_DVN",
        )

        print(f"Waterfall-Order platziert. Status: {trade.orderStatus.status}")

        print(
            f"Order platziert: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}"
        )
        print(f"Strategie Reference: {trade.order.orderRef}")
        print(f"Parent Status: {trade.orderStatus.status}")

        # Warten, um zu sehen, ob die Order von der TWS akzeptiert wird
        await asyncio.sleep(1)
        print(f"Status nach 1s: {trade.orderStatus.status}")

    except Exception as e:
        print(f"Fehler: {e}")

    finally:
        ib.disconnect()

    # BEISPIEL:
    # Wir wollen 10 Aktien von NVDA kaufen.
    # Wenn der Kurs die 500.00 Marke durchbricht (Stop Buy).
    # Stop Loss bei 490.00.
    # Take Profit bei 520.00.
    # Strategie: "BREAKOUT_V1"

    """

    try:
        trade = await place_strategy_bracket_order(
            ib_instance=ib,
            symbol="POST",
            qty=int(100 / (98.72 - 96.25)),
            entry_stp_price=98.72,
            exit_stp_price=96.25,
            exit_lmt_price=103.66,
            strategy_name="RedLolly",
        )

        print(
            f"Order platziert: {trade.order.action} {trade.order.totalQuantity} {trade.contract.symbol}"
        )
        print(f"Strategie Reference: {trade.order.orderRef}")
        print(f"Parent Status: {trade.orderStatus.status}")

        # Warten, um zu sehen, ob die Order von der TWS akzeptiert wird
        await asyncio.sleep(1)
        print(f"Status nach 1s: {trade.orderStatus.status}")

    except Exception as e:
        print(f"Fehler beim Platzieren der Order: {e}")

    finally:
        ib.disconnect()
    """


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # asyncio.run(main())
    asyncio.run(fetch_and_process_data())
