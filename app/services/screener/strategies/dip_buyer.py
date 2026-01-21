import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd

from ....services.database import SignalDatabase
from ....services.market_data_provider import MarketDataDict, MarketDataProvider
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from ..types import DipBuyerResult
from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DipBuyerConfig:
    """Konfiguration für die DipBuyer Strategie."""

    MIN_VOLUME: int = 1_000_000
    MIN_PRICE: float = 5.0
    MAX_ATR_R3: float = -1.0
    MIN_VOLA_RATIO: float = 0.03
    MAX_IBS: float = 0.2

    SMA_TREND_WINDOW: int = 200
    VOL_SMA_WINDOW: int = 20
    ATR_WINDOW: int = 5


class DipBuyerStrategy(BaseStrategy[DipBuyerResult]):
    name: str = "DipBuyer"
    table_name: str = "screener_dip_buyer"

    def __init__(
        self,
        signals_db: SignalDatabase,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
        config: DipBuyerConfig = DipBuyerConfig(),
    ) -> None:
        super().__init__(signals_db, data_provider, telegram_bot)
        self.cfg = config

        # Singleton Instanz abrufen
        exchange_data = ExchangeSymbol()
        self.dow_set = set(exchange_data.dow_30)
        self.sp500_set = set(exchange_data.sp_500)
        self.ndx_set = set(exchange_data.nasdaq_100)
        self.russell_set = set(exchange_data.russell_1000)

    @override
    def run(self, days: int = 0) -> int:
        mode = "Backfill" if days > 0 else "Daily"
        lookback_days = 400 + days

        logger.info(
            f"[{self.name}] Starte Analyse ({mode}, Lookback: {lookback_days}d)..."
        )

        # 1. Daten via Provider laden
        market_data = self.data_provider.get_all_daily_data(days=lookback_days)
        if not market_data:
            logger.warning(f"[{self.name}] Keine Marktdaten erhalten.")
            return 0

        # 2. Indikatoren berechnen
        indicators = self._calculate_indicators(market_data)

        # 3. Signale extrahieren
        total_len = len(indicators["close"])
        start_idx = (
            max(self.cfg.SMA_TREND_WINDOW, total_len - days)
            if days > 0
            else total_len - 1
        )

        all_results: list[DipBuyerResult] = []

        for i in range(start_idx, total_len):
            if daily_hits := self._scan_single_day(indicators, idx_pos=i):
                all_results.extend(daily_hits)

        if not all_results:
            logger.info(f"[{self.name}] Keine Treffer.")
            return 0

        # 4. Speichern (Generisch)
        results_as_dicts = [res.to_dict() for res in all_results]
        self.signals_db.save_screener_results(self.table_name, results_as_dicts)

        # Index-Analyse Loggen
        self._log_index_distribution(all_results)

        # Telegram & Trades nur bei Daily Run
        if days == 0:
            self._create_active_trades(all_results)
            latest_date = all_results[-1].date
            self._send_telegram_report("📉 Dip-Buyer", latest_date, results_as_dicts)

        logger.info(f"[{self.name}] Fertig: {len(all_results)} Treffer gespeichert.")
        return len(all_results)

    def _log_index_distribution(self, results: list[DipBuyerResult]) -> None:
        """Analysiert und loggt die Index-Zugehörigkeit pro Tag."""

        by_date = defaultdict(list)
        for res in results:
            by_date[res.date].append(res)

        sorted_dates = sorted(by_date.keys())

        logger.info("=======================================")
        logger.info("📊 TÄGLICHE INDEX-VERTEILUNG (DipBuyer)")
        logger.info("=======================================")

        for date_str in sorted_dates:
            day_results = by_date[date_str]

            stats = {"DOW": 0, "SPX": 0, "NDX": 0, "RUS": 0, "RUS_EXCLUSIVE": []}

            for res in day_results:
                sym = res.symbol
                in_major = False

                if sym in self.dow_set:
                    stats["DOW"] += 1
                    in_major = True
                if sym in self.sp500_set:
                    stats["SPX"] += 1
                    in_major = True
                if sym in self.ndx_set:
                    stats["NDX"] += 1
                    in_major = True

                if sym in self.russell_set:
                    stats["RUS"] += 1
                    if not in_major:
                        stats["RUS_EXCLUSIVE"].append(sym)

            logger.info(f"📅 {date_str} (Treffer: {len(day_results)})")
            logger.info(
                f"   DOW: {stats['DOW']} | SPX: {stats['SPX']} | NDX: {stats['NDX']} | RUS: {stats['RUS']}"
            )

            unique_gems = sorted(list(set(stats["RUS_EXCLUSIVE"])))
            if unique_gems:
                gems_str = ", ".join(unique_gems)
                # Kürzen bei zu vielen Treffern
                if len(gems_str) > 120:
                    gems_str = gems_str[:120] + "..."
                logger.info(f"   💎 Hidden Gems: {gems_str}")
            else:
                logger.info("   💎 Hidden Gems: -")

            logger.info("   -----------------------------------")

    def _create_active_trades(self, results: list[DipBuyerResult]) -> None:
        for res in results:
            self.signals_db.add_trade(
                symbol=res.symbol,
                signal_date=res.date,
                entry_price=res.entry_limit,
                atr_at_entry=res.atr5,
                quantity=1,
                strategy=self.name,
            )

    def _load_market_data(self, days: int) -> MarketDataDict | None:
        # Fallback Methode, falls Provider nicht genutzt wird (Legacy Support)
        # Aber im neuen Code nutzen wir self.data_provider.
        return self.data_provider.get_all_daily_data(days)

    def _calculate_indicators(self, data: MarketDataDict) -> dict[str, pd.DataFrame]:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        sma200 = close.rolling(window=self.cfg.SMA_TREND_WINDOW, min_periods=150).mean()
        vol_sma20 = volume.rolling(window=self.cfg.VOL_SMA_WINDOW).mean()

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        rma_span = (2 * self.cfg.ATR_WINDOW) - 1
        atr5 = (
            pd.DataFrame(tr, index=close.index, columns=close.columns)
            .ewm(span=rma_span, adjust=False)
            .mean()
        )

        atr_r3 = (close - close.shift(3)) / atr5.replace(0, np.nan)
        range_hl = (high - low).replace(0, 0.01)
        ibs = (close - low) / range_hl
        entry_limits = close - atr5

        return {
            "sma200": sma200,
            "vol_sma20": vol_sma20,
            "atr5": atr5,
            "atr_r3": atr_r3,
            "ibs": ibs,
            "entry_limits": entry_limits,
            "close": close,
            "open": data["open"],
            "high": high,
            "prev_close": prev_close,
            "prev_open": data["open"].shift(1),
        }

    def _scan_single_day(
        self, ind: dict[str, pd.DataFrame], idx_pos: int
    ) -> list[DipBuyerResult]:
        def _get(key: str) -> pd.Series:
            return ind[key].iloc[idx_pos]

        current_close = _get("close")

        mask = (
            (_get("vol_sma20") > self.cfg.MIN_VOLUME)
            & (current_close > self.cfg.MIN_PRICE)
            & (current_close > _get("sma200"))
            & (_get("atr_r3") < self.cfg.MAX_ATR_R3)
            & ((_get("atr5") / current_close) > self.cfg.MIN_VOLA_RATIO)
            & (current_close < _get("open"))
            & (_get("prev_close") < _get("prev_open"))
            & (_get("ibs") < self.cfg.MAX_IBS)
        )

        hits = mask[mask].index.tolist()
        if not hits:
            return []

        date_str = ind["close"].index[idx_pos].strftime("%Y-%m-%d")
        results = []

        atr_r3_s = _get("atr_r3")
        entry_s = _get("entry_limits")
        high_s = _get("high")
        atr5_s = _get("atr5")

        for symbol in hits:
            if pd.isna(current_close[symbol]):
                continue

            res = DipBuyerResult(
                date=date_str,
                symbol=symbol,
                exchange=self._get_exchange(symbol),
                timeframe="1D",
                close=round(float(current_close[symbol]), 2),
                high=round(float(high_s[symbol]), 2),
                atr_r3=round(float(atr_r3_s[symbol]), 2),
                setup_score=round(float(atr_r3_s[symbol]) * -1, 2),
                entry_limit=round(float(entry_s[symbol]), 2),
                atr5=round(float(atr5_s[symbol]), 2),
            )
            results.append(res)

        results.sort(key=lambda x: x.setup_score, reverse=True)
        return results
