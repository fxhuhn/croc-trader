"""Bounce Bandit Trade Manager Execution Strategy.

Execution Rules:
1. Entry (CREATED -> ACTIVE):
   - Market On Open (MOO) on the trading session following setup generation (Bar t+1).
   - Entry price = Open of Bar t+1.
   - Position size calculated from portfolio budget allocation.
2. Exits (ACTIVE -> CLOSED):
   - Market On Close (MOC) on any active holding session (evaluated at Close).
   - Exit condition: Close > SMA(8) OR RSI(2) > 75.
   - Evaluated starting on the entry day close (t+1).
"""

import logging
from decimal import Decimal
from typing import final, override

import pandas as pd

from ....const import Strategies
from ....models import Order, TradeParams
from ....tools.indicators import calculate_rsi, calculate_sma
from ....types import TradeData
from ..types import TradeTransition
from .abstract import BaseTradeStrategy, OrderOptions

logger = logging.getLogger(__name__)


def calculate_required_sma_exit(close_series: pd.Series) -> float:
    """Pure calculation: computes minimum close price required to exceed 8-period SMA on current candle."""
    last_7_closes = close_series.iloc[-7:]
    return float(last_7_closes.mean()) + 0.01


def calculate_required_rsi_exit(close_series: pd.Series) -> float:
    """Pure calculation: computes minimum close price required to exceed RSI(2) > 75 on current candle."""
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain_series = gain.ewm(alpha=0.5, adjust=False).mean()
    avg_loss_series = loss.ewm(alpha=0.5, adjust=False).mean()

    last_avg_gain = float(avg_gain_series.iloc[-1])
    last_avg_loss = float(avg_loss_series.iloc[-1])
    last_close = float(close_series.iloc[-1])

    required_delta_rsi = max(0.0, (3.0 * last_avg_loss) - last_avg_gain)
    return last_close + required_delta_rsi + 0.01


def calculate_bounce_bandit_targets(
    close_series: pd.Series, exit_sma_length: int = 8
) -> dict[str, float]:
    """Pure calculation: computes dynamic daily indicators and exit target prices for Bounce Bandit."""
    sma_8_series = calculate_sma(close_series, exit_sma_length)
    rsi_2_series = calculate_rsi(close_series, 2)

    if sma_8_series.empty or rsi_2_series.empty:
        return {}

    current_sma_8 = float(sma_8_series.iloc[-1])
    current_rsi_2 = float(rsi_2_series.iloc[-1])

    required_sma_exit = calculate_required_sma_exit(close_series)
    required_rsi_exit = calculate_required_rsi_exit(close_series)
    target_price = min(required_sma_exit, required_rsi_exit)

    return {
        "sma_8": round(current_sma_8, 2),
        "rsi_2": round(current_rsi_2, 2),
        "target": round(target_price, 2),
        "target_price": round(target_price, 2),
        "required_sma_exit": round(required_sma_exit, 2),
        "required_rsi_exit": round(required_rsi_exit, 2),
    }


@final
class BounceBanditTradeStrategy(BaseTradeStrategy):
    """Manages execution and exit lifecycle for the 'Bounce Bandit' strategy.

    Rules:
    1. Entry: MOO entry executed on the next trading session open following setup.
    2. Exits: Evaluated at market close. Exit if Close > SMA(8) OR RSI(2) > 75.
    """

    STRATEGY_IDENTIFIER = Strategies.BounceBandit
    name = Strategies.BounceBandit

    EXIT_SMA_LEN = 8
    RSI_EXIT_THRESHOLD = 75.0

    @override
    def get_current_parameters(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame | None = None,
    ) -> TradeParams | None:
        """Calculates current strategy parameters for display."""
        entry_price = float(trade.get("entry_price") or 0.0)

        return TradeParams(
            stop_loss=0.0,
            take_profit_1=0.0,
            extras={
                "entry_price": entry_price,
                "current_size": float(trade.get("current_size") or 0.0),
                "exit_sma_len": self.EXIT_SMA_LEN,
                "rsi_exit_threshold": self.RSI_EXIT_THRESHOLD,
            },
        )

    @override
    def get_daily_updates(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
    ) -> dict[str, object]:
        """Calculates daily indicator values (SMA_8, RSI_2) and required exit target prices to persist in signal_context."""
        if dataframe_history.empty or len(dataframe_history) < self.EXIT_SMA_LEN:
            return {}

        close_series = dataframe_history["close"].astype(float)
        targets = calculate_bounce_bandit_targets(close_series, self.EXIT_SMA_LEN)
        return dict(targets)

    @override
    def _generate_entry_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates MOO entry order for CREATED trades with OPG time-in-force."""
        return self._generate_budget_entry_order(
            trade=trade,
            budget=budget,
            options=OrderOptions(order_type="MKT", time_in_force="OPG"),
        )

    @override
    def _generate_exit_order(
        self,
        trade: TradeData,
        dataframe_history: pd.DataFrame,
        budget: float,
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates dynamic LOC exit orders for ACTIVE trades.

        Uses calculate_bounce_bandit_targets to determine the required threshold
        for Close > SMA_8 or RSI_2 > 75 and places a Limit On Close (LOC) order.
        """
        quantity = int(trade.get("current_size") or 0)
        if (
            quantity <= 0
            or dataframe_history.empty
            or len(dataframe_history) < self.EXIT_SMA_LEN
        ):
            return None

        close_series = dataframe_history["close"].astype(float)
        targets = calculate_bounce_bandit_targets(close_series, self.EXIT_SMA_LEN)
        target_price = targets.get("target_price")

        if not target_price or target_price <= 0:
            return None

        return self._create_exit_order(
            symbol=trade["symbol"],
            quantity=quantity,
            price=Decimal(str(target_price)),
            options=OrderOptions(order_type="LOC", time_in_force="DAY"),
        )

    @override
    def check_entry(
        self,
        trade: TradeData,
        candle: pd.Series,
        dataframe_history: pd.DataFrame,
        active_symbols: set[str] | None = None,
    ) -> TradeTransition | None:
        """Activates entry on the open of the next trading session (MOO Entry)."""
        candle_date = (
            pd.Timestamp(candle["date"]).date()
            if isinstance(candle["date"], str)
            else candle["date"].date()
        )

        setup_date = self._get_setup_date(trade)
        if setup_date and candle_date <= setup_date:
            # Entry setup candle is bar t; MOO entry executes on bar t+1
            return None

        open_price = float(candle.get("open") or trade.get("entry_price") or 0.0)
        if open_price <= 0:
            return None

        date_string = candle_date.strftime("%Y-%m-%d")

        return self._execute_activation(
            trade,
            open_price,
            "Bounce Bandit MOO Entry",
            date_string,
        )

    @override
    def _do_manage_active_trade(
        self,
        trade: TradeData,
        current_candle: pd.Series,
        date_string: str,
        dataframe_history: pd.DataFrame,
        latest_leaders: set[str] | None = None,
    ) -> TradeTransition | None:
        """Manages Exits for Bounce Bandit:

        Evaluated at market close MOC.
        Exit if Close > SMA(8) OR RSI(2) > 75.
        """
        if dataframe_history.empty or len(dataframe_history) < self.EXIT_SMA_LEN:
            raise ValueError(
                f"Insufficient price history for Bounce Bandit trade ID {trade.get('id')} ({trade.get('symbol')}): "
                f"expected at least {self.EXIT_SMA_LEN} candles, got {len(dataframe_history)}"
            )

        close_series = dataframe_history["close"].astype(float)
        sma_8_series = calculate_sma(close_series, self.EXIT_SMA_LEN)
        rsi_2_series = calculate_rsi(close_series, 2)

        current_close = float(current_candle["close"])
        current_sma_8 = float(sma_8_series.iloc[-1])
        current_rsi_2 = float(rsi_2_series.iloc[-1])

        sma_exit = current_close > current_sma_8
        rsi_exit = current_rsi_2 > self.RSI_EXIT_THRESHOLD

        if sma_exit or rsi_exit:
            if rsi_exit and sma_exit:
                exit_reason = "RSI / SMA"
            elif rsi_exit:
                exit_reason = "RSI"
            else:
                exit_reason = "SMA"

            reason_text = f"Bounce Bandit MOC Exit: {exit_reason} (SMA_8={current_sma_8:.2f}, RSI_2={current_rsi_2:.2f})"
            logger.info(
                "Closing Bounce Bandit trade ID %s on %s: %s",
                trade.get("id"),
                date_string,
                reason_text,
            )
            return self._close_trade(
                trade,
                current_close,
                exit_reason,
                date_string,
            )

        return None
