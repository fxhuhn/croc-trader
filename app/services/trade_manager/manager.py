import logging
from pathlib import Path
from datetime import datetime
import re

from ...const import Strategies, STRATEGY_ALIASES
from ...types import TradeStatus
from ...database.repositories.trade import TradeRepository
from ...database.repositories.market import MarketRepository
from ...database.session import DatabaseSession
from ...services.telegram import TelegramBot

# Strategien importieren
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.hold_target import HoldTargetStrategy
from .strategies.split_target import SplitTargetStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy
from .strategies.two_percent_strategy import TwoPercentStrategy
from .strategies.ndx_momentum import NDXMomentumTradeStrategy

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Manages the lifecycle of trades, including entry, exit, and order generation.
    Resolves legacy strategy names to strict Enum constants.
    """

    def __init__(
        self,
        db_path: Path,
        stocks_db_path: Path,
        telegram_bot: TelegramBot | None = None,
    ):
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot

        # Verbindung zur Stocks-DB (für Kursdaten)
        self.stocks_session = DatabaseSession(str(stocks_db_path))
        self.market_repo = MarketRepository(self.stocks_session)

        # Verbindung zur Signals-DB
        self.signals_session = DatabaseSession(str(db_path))
        self.trade_repository = TradeRepository(self.signals_session)

        # Strategie-Registry (Strict Keys via Enum)
        self.strategies = {
            Strategies.DipBuyer: DipBuyerStrategy(),
            Strategies.TurnOverTiming: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_10: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_05: TurnoverTimingStrategy(),
            Strategies.HoldTarget: HoldTargetStrategy(),
            Strategies.SplitTarget: SplitTargetStrategy(),
            Strategies.TwoPercent: TwoPercentStrategy(),
            Strategies.NDXMomentum: NDXMomentumTradeStrategy(),
        }

        logger.info(
            f"TradeManager init. Registered Strategies: {list(self.strategies.keys())}"
        )

    def _resolve_strategy_name(self, name: str) -> Strategies | None:
        """
        Resolves a string (legacy or new) to a strict `Strategies` Enum member.
        """
        if not name:
            return None

        # 1. Direct Match (if already correct string)
        try:
            return Strategies(name)
        except ValueError:
            pass

        # 2. Normalized Lookup from Aliases
        # Normalize: lowercase, remove non-alphanumeric (except dots/underscore if needed,
        # but our keys are mostly clean. Let's try direct alias lookup first)

        # Basic cleanup for lookup
        lower_name = name.lower()

        if lower_name in STRATEGY_ALIASES:
            return STRATEGY_ALIASES[lower_name]

        # 3. Fallback: Aggressive Normalization (remove underscores/spaces)
        # e.g. "Two Percent" -> "twopercent"
        clean_name = re.sub(r"[^a-z0-9]", "", lower_name)

        # Search aliases values / keys for match?
        # The aliases map above handles most cases. Let's add specific fallbacks if needed.

        # Manual Fallbacks for very messy inputs
        if "turnover" in clean_name:
            return Strategies.TurnOverTiming
        if "dip" in clean_name:
            return Strategies.DipBuyer
        if "hold" in clean_name or "tp3" in clean_name:
            return Strategies.HoldTarget
        if "split" in clean_name:
            return Strategies.SplitTarget
        if "percent" in clean_name:
            return Strategies.TwoPercent

        return None

    def _get_strategy(self, strategy_name: str):
        """
        Findet die passende Strategie-Instanz.
        """
        enum_key = self._resolve_strategy_name(strategy_name)

        if enum_key and enum_key in self.strategies:
            return self.strategies[enum_key]

        # 3. Nichts gefunden -> Warning loggen!
        logger.warning(
            f"⚠️ ACHTUNG: Unbekannte Strategie '{strategy_name}' (Resolved: {enum_key}). Keine Logik gefunden!"
        )
        return None

    def run_daily_process(self):
        logger.info("TradeManager: Starte Daily Process...")

        # 1. Active Trades verwalten (Exit Checks)
        try:
            active_trades = self.trade_repository.get_by_status(TradeStatus.ACTIVE)
            logger.info(f"Prüfe {len(active_trades)} aktive Trades auf Exits...")
            for trade in active_trades:
                self._process_active_trade(trade)
        except Exception as e:
            logger.error(f"Fehler beim Laden aktiver Trades: {e}")

        # 2. Pending Trades verwalten (Entry Checks)
        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
            logger.info(f"Prüfe {len(created_trades)} wartende Trades auf Entry...")
            for trade in created_trades:
                self._process_created_trade(trade)
        except Exception as e:
            logger.error(f"Fehler beim Laden wartender Trades: {e}")

        logger.info("TradeManager: Daily Process beendet.")

    def _process_active_trade(self, trade: dict):
        symbol = trade["symbol"]
        strat_name = trade["strategy"]
        strategy = self._get_strategy(strat_name)

        if not strategy:
            logger.warning(
                f"Keine Strategie-Klasse für '{strat_name}' ({symbol}) gefunden."
            )
            return

        try:
            # FIX: Data Loading jetzt IM try-Block, um DB-Locks abzufangen
            start_date = trade.get("entry_date")
            if not start_date:
                start_date = "2024-01-01"

            df_hist = self.market_repo.get_symbol_history_raw(
                symbol, start_date=str(start_date)
            )
            if df_hist.empty:
                return

            result = strategy.manage_active_trade(trade, df_hist, self.trade_repository)

            if result:
                logger.info(f"Trade Update {symbol}: {result}")
                # if self.telegram:
                #    self.telegram.send_message(f"🔄 **Trade Update**\n{result}")

            # Update current price für Dashboard
            if not df_hist.empty:
                current_close = df_hist.iloc[-1]["close"]
                self.trade_repository.update_trade(
                    trade["id"], {"current_price": current_close}
                )

        except Exception as e:
            logger.error(f"Fehler bei Exit Check {symbol}: {e}", exc_info=True)

    def _process_created_trade(self, trade: dict):
        symbol = trade["symbol"]
        strat_name = trade["strategy"]
        strategy = self._get_strategy(strat_name)

        if not strategy:
            return

        try:
            # FIX: Data Loading jetzt IM try-Block
            # trade.get('created_at') unused
            # Puffer: Laden ab 10 Tage vor Signal
            df_hist = self.market_repo.get_symbol_history_raw(
                symbol, start_date="2024-01-01"
            )
            if df_hist.empty:
                return

            # Letzte Kerze prüfen
            candle = df_hist.iloc[-1]

            result = strategy.check_entry(trade, candle, df_hist, self.trade_repository)

            if result:
                logger.info(f"Entry Check {symbol}: {result}")
                # if "FILLED" in result and self.telegram:
                #    self.telegram.send_message(f"✅ **Entry Executed**\n{result}")
        except Exception as e:
            logger.error(f"Fehler bei Entry Check {symbol}: {e}", exc_info=True)

    def generate_daily_orders(self) -> str | None:
        """
        Generiert die Daily Orders für alle CREATED Trades aller Strategien.
        Speichert das Ergebnis als YAML.
        """
        import yaml
        from dataclasses import asdict

        logger.info("Generiere Daily Orders...")

        # 1. Hole alle CREATED Trades
        # (Wir holen alle, Strategien filtern selbst oder generieren Orders wenn passend)
        created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)

        orders = []

        for trade in created_trades:
            symbol = trade["symbol"]
            strat_name = trade["strategy"]
            strategy = self._get_strategy(strat_name)

            if not strategy:
                logger.warning(f"Keine Strategie für Trade {symbol} ({strat_name})")
                continue

            try:
                # Load History (Efficiency: Only needed if strat uses it for generation logic, usually not for basic Order file)
                # But abstract signature demands it.
                # Assuming empty DataFrame is handled or needed.
                # Let's load minimal history just in case.
                df_hist = self.market_repo.get_symbol_history_raw(
                    symbol, start_date="2024-01-01"
                )

                # Default Budget from Trade Record (set by Portfolio Manager)
                budget = float(
                    trade.get("budget") or 2000.0
                )  # Fallback to 2000 if PM failed or missed

                order = strategy.generate_orders(
                    trade, df_hist, budget, self.trade_repository
                )

                if order:
                    # Clean None values (e.g. qty in Entry Leg)
                    order_dict = asdict(order)

                    def clean_nones(d):
                        if isinstance(d, dict):
                            return {
                                k: clean_nones(v) for k, v in d.items() if v is not None
                            }
                        if isinstance(d, list):
                            return [clean_nones(v) for v in d]
                        return d

                    cleaned_order = clean_nones(order_dict)
                    orders.append(cleaned_order)
                    logger.info(f"Order generiert: {symbol}")

            except Exception as e:
                logger.error(
                    f"Fehler bei Order Generierung {symbol}: {e}", exc_info=True
                )

        if not orders:
            logger.info("Keine Orders zu generieren.")
            return None

        # 2. Serialize to YAML
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"orders_{date_str}.yaml"
        output_dir = Path("data/orders")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        with open(file_path, "w") as f:
            yaml.dump(orders, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Orders gespeichert: {file_path}")
        return str(file_path)
