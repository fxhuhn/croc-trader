import json
import logging
from pathlib import Path
from typing import Any

import yaml

from ....config import settings
from ....tools.symbol_lists import ExchangeSymbol
from ....database.repositories.trade import TradeRepository
from ....database.repositories.signal import SignalRepository
from ....database.repositories.market_data_provider import MarketDataProvider
from ....services.telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)

class CrocSetupStrategy(BaseStrategy):
    """
    CrocSetup Screener.
    
    Features:
    - Processes webhook signals from the 'croc' table (via SignalRepository).
    - Dynamically computes percentage SMA distances from raw data (Close, SMA20, SMA200).
    - Applies YAML ranking rules (with flexible mapping and match/case logic).
    """
    name: str = "CrocSetup"

    def __init__(
        self,
        trade_repo: TradeRepository,
        data_provider: MarketDataProvider,
        signal_repo: SignalRepository,
        telegram_bot: TelegramBot | None = None
    ) -> None:
        super().__init__(data_provider, telegram_bot)
        
        self.trade_repo = trade_repo
        self.signal_repo = signal_repo
        
        self.config_path: Path = settings.get_path("ranking_yaml")
        self.ranking_rules = self._load_config()
        self.exchange_symbols = ExchangeSymbol()
        
        logger.info(f"🐊 {self.name} initialized. Rules: {len(self.ranking_rules)} entries.")

    def _load_config(self) -> list[dict[str, Any]]:
        if not self.config_path.exists():
            logger.error(f"Config missing: {self.config_path}")
            return []
        try:
            with open(self.config_path, encoding="utf-8") as file_handle:
                data = yaml.safe_load(file_handle)
            if isinstance(data, list):
                return data
            return data.get("ranking_2026", [])
        except Exception as error:
            logger.error(f"YAML Error at {self.config_path}: {error}")
            return []

    def _get_indices_string(self, symbol: str) -> str:
        indices = []
        if symbol in self.exchange_symbols.sp_500:
            indices.append("SPX")
        if symbol in self.exchange_symbols.nasdaq_100:
            indices.append("NDX")
        if symbol in self.exchange_symbols.dow_30:
            indices.append("DOW")
        if symbol in self.exchange_symbols.russell_1000: 
            if "SPX" not in indices:
                indices.append("RUS_EXCL")
            else:
                indices.append("RUS")
        return ",".join(indices) if indices else "-"

    def run(self, days: int = 0, analysis_date: str | None = None, specific_symbols: list[str] | None = None) -> int:
        try:
            signals = self.signal_repo.get_signals_by_date(
                analysis_date=analysis_date, 
                days_lookback=days
            )
        except Exception as error:
            logger.error(f"Error loading signals from repository: {error}")
            return 0

        if not signals:
            return 0

        hits = 0
        for row in signals:
            # 1. Unpack & Merge Data
            try:
                signal_data = json.loads(row['data'])
            except (ValueError, TypeError):
                signal_data = {}
            
            flat_row = dict(row) # Copy
            flat_row.update(signal_data)
            
            # 2. Enrich Data (Compute SMA Distances)
            self._compute_sma_distances(flat_row)
            
            # 3. Perform Matching
            best_match = self._find_best_match(flat_row)
            
            if best_match:
                self._create_trade_from_signal(flat_row, best_match)
                hits += 1
                
        logger.info(f"🐊 [{self.name}] {hits} Trades created from signals.")
        return hits

    def _compute_sma_distances(self, row: dict[str, Any]) -> None:
        """
        Calculates SMA distances in percent from raw data.
        Formula: ((Close - SMA) / SMA) * 100
        Modifies the dictionary in-place.
        """
        try:
            close = float(row.get('close') or 0.0)
            sma_20 = float(row.get('sma_20') or 0.0)
            sma_200 = float(row.get('sma_200') or 0.0)

            if sma_20 > 0:
                row['dist_sma_20'] = ((close - sma_20) / sma_20) * 100
            else:
                row['dist_sma_20'] = 0.0

            if sma_200 > 0:
                row['dist_sma_200'] = ((close - sma_200) / sma_200) * 100
            else:
                row['dist_sma_200'] = 0.0
        except (ValueError, TypeError):
            # Fail gracefully on bad data, just don't add the fields
            pass

    def _find_best_match(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """
        Checks the signal against all YAML rules.
        Ignores fields that are not present in the data.
        """
        signal_name = row.get("signal")
        candidates = [r for r in self.ranking_rules if r.get("Signal") == signal_name]
        
        if not candidates: return None

        best_match = None
        best_score = -999.0

        for rule in candidates:
            passed = True
            
            for yaml_key, condition_value in rule.items():
                
                # 1. Ignore Metadata
                if yaml_key in {"Signal", "Exit", "Score", "Status", "R_Hist", "Risk_95", "R_2026", "R_4W_Trend", "Avg_Days", "TimeExit"}:
                    continue

                # 2. Key Mapping (YAML -> DB Field)
                db_key = yaml_key.lower()
                
                # EMA / SMA Mapping (Check 200 before 20 to avoid confusion)
                if "ema" in db_key or "sma" in db_key:
                    if "200" in db_key:
                        db_key = "dist_sma_200"
                    elif "20" in db_key:
                        db_key = "dist_sma_20"
                
                # RSI Mapping
                if "rsi" in db_key:
                    db_key = "rsi"

                # 3. Existence Check
                if db_key not in row:
                    continue 

                market_value = row[db_key]
                
                # 4. Condition Check (Match/Case)
                if not self._check_condition(market_value, condition_value, yaml_key):
                    passed = False
                    break
            
            if passed:
                score = float(rule.get("Score", 0.0))
                if best_match is None or score > best_score:
                    best_match = rule
                    best_score = score
        
        return best_match

    def _check_condition(self, market_val: Any, condition_str: Any, context_key: str = "Unknown") -> bool:
        """
        Checks condition using 'match / case'.
        Logs error for unknown condition strings.
        """
        if market_val is None: return False
        
        try:
            val = float(market_val)
        except (ValueError, TypeError):
            # Fallback for non-numbers
            return str(market_val).lower() == str(condition_str).lower()

        condition_string = str(condition_str).strip()
        key_lower = context_key.lower()

        # --- A) EMA / SMA Logic ---
        if "ema" in key_lower or "sma" in key_lower:
            match condition_string:
                case "< -10%":
                    return val < -10.0
                case "-10 to -3%":
                    return -10.0 <= val <= -3.0
                case "-3 to 0%":
                    return -3.0 <= val <= 0.0
                case "0 to 3%":
                    return 0.0 <= val <= 3.0
                case "3 to 10%":
                    return 3.0 <= val <= 10.0
                case "> 10%":
                    return val > 10.0
                case _:
                    logger.error(f"❌ Unknown EMA Condition: '{condition_string}' in '{context_key}'")
                    return False

        # --- B) RSI Logic ---
        if "rsi" in key_lower:
            match condition_string:
                # Exact Matches (with brackets)
                case "Oversold (<30)":
                    return val < 30.0
                case "Weak (30-45)":
                    return 30.0 <= val < 45.0
                case "Neutral (45-55)":
                    return 45.0 <= val <= 55.0
                case "Strong (55-70)":
                    return 55.0 < val <= 70.0
                case "Overbought (>70)":
                    return val > 70.0
                
                # Fallback for pure text labels (if YAML varies)
                case "Oversold":
                    return val < 30.0
                case "Weak":
                    return 30.0 <= val < 45.0
                case "Neutral":
                    return 45.0 <= val <= 55.0
                case "Strong":
                    return 55.0 < val <= 70.0
                case "Overbought":
                    return val > 70.0

                case _:
                    logger.error(f"❌ Unknown RSI Condition: '{condition_string}' in '{context_key}'")
                    return False

        return False

    def _create_trade_from_signal(self, row: dict[str, Any], match: dict[str, Any]) -> None:
        symbol = row.get('symbol', 'UNKNOWN')
        indices = self._get_indices_string(symbol)
        
        try:
            high = float(row.get('high') or 0)
            low = float(row.get('low') or 0)
            close = float(row.get('close') or 0)
        except (ValueError, TypeError): 
            logger.error(f"Missing price data for {symbol}")
            return

        entry_price = high
        risk_range = high - low
        sl_price = entry_price - risk_range

        if risk_range <= 0:
            logger.error(f"❌ Invalid Risk Range: {risk_range} for {symbol}")
            return
        
        # --- TARGET CALCULATION ---
        yaml_exit_name = match.get("Exit", "unknown").lower()

        # For Hold/TP3 Strategies: 3R
        if ("tp3" in yaml_exit_name) and ("hold" in yaml_exit_name):
            r_multiple = 3.0
        else:
            # logger.info(f"❌ Unknown Exit Name: {yaml_exit_name}")  
            r_multiple = 1.0

        target_price = entry_price + (r_multiple * risk_range)
        
        context = {
            "source": "webhook",
            "date": row.get('date_str', row.get('timestamp')),
            "setup_score": float(match.get("Score", 0)),
            "close": close,
            "volume": row.get('volume'),
            "indices": indices,
            "original_signal": row.get('signal'),
            "match_rule": match,
            
            # Strategy Data from YAML
            "R_Hist": match.get("R_Hist"),
            "R_4W_Trend": match.get("R_4W_Trend"),
            "Avg_Days": match.get("Avg_Days"),
            
            # Debug Data
            "calc_dist_sma_20": row.get("dist_sma_20"),
            "calc_dist_sma_200": row.get("dist_sma_200")
        }

        self.trade_repo.create_trade(
            symbol=symbol,
            strategy=f"Croc_{yaml_exit_name}",
            size=0,
            entry=entry_price,
            sl=sl_price,
            target=target_price,
            context=context
        )