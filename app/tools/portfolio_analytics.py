"""Pure domain and mathematical calculations for portfolio analytics and allocation models.

Follows the Functional Core pattern: deterministic calculations without side effects or I/O.
"""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # Standard un-typed library import

from ..const import STRATEGY_ALIASES
from ..tools import metrics
from ..tools.portfolio_optimization import (
    build_covariance_matrix,
    calculate_risk_contributions,
    compute_downside_deviation,
    optimize_max_sharpe_weights,
    optimize_risk_parity_weights,
)

MIN_SERIES_LEN: int = 2
LOW_DATA_THRESHOLD: int = 5
CALC_EPSILON: float = 1e-6
GERMAN_MONTH_NAMES: dict[int, str] = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember",
}


def calculate_unweighted_monthly_pct(month_df: pd.DataFrame) -> float:
    """Calculates the unweighted average percentage return of trades in a month.

    Args:
        month_df: DataFrame of closed trades for the month.

    Returns:
        float: Unweighted average percentage return across trades.
    """
    if month_df.empty:
        return 0.0

    pnl = pd.to_numeric(month_df["realized_pnl"], errors="coerce").fillna(0.0)
    entry_prices = pd.to_numeric(month_df["entry_price"], errors="coerce").fillna(0.0)
    initial_sizes = pd.to_numeric(month_df["initial_size"], errors="coerce").fillna(0.0)

    invested = entry_prices * initial_sizes
    valid_mask = invested > 0.0

    if not valid_mask.any():
        return 0.0

    trade_pcts = (pnl[valid_mask] / invested[valid_mask]) * 100.0
    return float(trade_pcts.mean())


def calculate_active_months(strat_df: pd.DataFrame) -> float:
    """Computes the active month span of a strategy based on entry and exit dates.

    Args:
        strat_df: DataFrame containing trade records with entry_date and exit_date_dt.

    Returns:
        float: Active month count (minimum 1.0).
    """
    if strat_df.empty or "entry_date" not in strat_df.columns:
        return 1.0

    entry_dates = pd.to_datetime(strat_df["entry_date"], errors="coerce").dropna()
    exit_dates = (
        strat_df["exit_date_dt"].dropna()
        if "exit_date_dt" in strat_df.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    if entry_dates.empty or exit_dates.empty:
        return 1.0

    first_date = entry_dates.min()
    last_date = exit_dates.max()
    days_span = max(1.0, float((last_date - first_date).days))
    return max(1.0, days_span / 30.44)


def extract_roi_series(strat_df: pd.DataFrame) -> pd.Series:
    """Extracts the realized ROI series from a trade DataFrame.

    Args:
        strat_df: DataFrame containing trade records with realized_pnl, entry_price, initial_size.

    Returns:
        pd.Series: ROI values per trade.
    """
    if strat_df.empty:
        return pd.Series(dtype=float)

    entry_prices = pd.to_numeric(strat_df["entry_price"], errors="coerce").fillna(0.0)
    initial_sizes = pd.to_numeric(strat_df["initial_size"], errors="coerce").fillna(0.0)
    invested_capital = entry_prices * initial_sizes
    valid_roi_mask = invested_capital > 0.0

    if not valid_roi_mask.any():
        return pd.Series(dtype=float)

    return (
        strat_df.loc[valid_roi_mask, "realized_pnl"] / invested_capital[valid_roi_mask]
    )


def calculate_evm_allocations(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
) -> dict[str, float]:
    """Calculates EV/M strategy weights from a cumulative closed trades slice.

    Args:
        slice_df: DataFrame of closed trades up to cutoff date.
        strategy_groups: Strategy group name to filter list mapping.

    Returns:
        dict[str, float]: Mapping from strategy group name to percentage weight (0.0 to 1.0).
    """
    num_strats = len(strategy_groups)
    default_weight = 1.0 / num_strats if num_strats > 0 else 0.0

    if slice_df.empty or "strategy" not in slice_df.columns:
        return dict.fromkeys(strategy_groups, default_weight)

    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    strategy_evs: dict[str, float] = {}
    for name, filters in strategy_groups.items():
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]
        roi_series = extract_roi_series(strat_slice_df)
        average_roi = float(roi_series.mean()) if not roi_series.empty else 0.0
        active_months = calculate_active_months(strat_slice_df)
        trades_per_month = len(strat_slice_df) / active_months
        strategy_evs[name] = trades_per_month * average_roi

    total_ev = sum(max(0.0, val) for val in strategy_evs.values())
    if total_ev > 0.0:
        return {
            name: (max(0.0, strategy_evs[name]) / total_ev) for name in strategy_groups
        }

    return dict.fromkeys(strategy_groups, default_weight)


def _extract_strategy_return_vectors(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
) -> tuple[list[str], list[pd.Series], list[float]]:
    """Extracts aligned ROI series and mean return vectors for strategy groups."""
    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )
    strat_names = list(strategy_groups.keys())
    strat_returns: list[pd.Series] = []
    mus: list[float] = []

    for name in strat_names:
        filters = strategy_groups[name]
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]
        roi_series = extract_roi_series(strat_slice_df)
        strat_returns.append(roi_series)
        mus.append(float(roi_series.mean()) if not roi_series.empty else 0.0)

    return strat_names, strat_returns, mus


def calculate_mean_variance_allocations(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
    model_type: str = "max_sharpe",
) -> dict[str, float]:
    """Calculates Mean-Variance or Risk Parity strategy weights from a cumulative closed trades slice."""
    num_strats = len(strategy_groups)
    default_weight = 1.0 / num_strats if num_strats > 0 else 0.0

    if slice_df.empty or "strategy" not in slice_df.columns:
        return dict.fromkeys(strategy_groups, default_weight)

    strat_names, strat_returns, mus = _extract_strategy_return_vectors(
        slice_df, strategy_groups
    )

    max_len = max((len(s) for s in strat_returns), default=0)
    if max_len < MIN_SERIES_LEN:
        return dict.fromkeys(strategy_groups, default_weight)

    padded_dict = {
        name: ser.reset_index(drop=True)
        for name, ser in zip(strat_names, strat_returns, strict=True)
    }
    returns_df = pd.DataFrame(padded_dict)

    cov_matrix = build_covariance_matrix(returns_df)
    mu_vector = np.array(mus, dtype=float)

    if model_type == "risk_parity":
        opt_weights = optimize_risk_parity_weights(cov_matrix)
    else:
        opt_weights = optimize_max_sharpe_weights(mu_vector, cov_matrix)

    return {name: float(w) for name, w in zip(strat_names, opt_weights, strict=True)}


def calculate_mean_variance_dashboard_data(
    slice_df: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
) -> dict[str, Any]:
    """Computes comprehensive Section 6 metrics (Max Sharpe, Risk Parity, MCR, TRC, PRC)."""
    strat_names = list(strategy_groups.keys())
    if slice_df.empty or "strategy" not in slice_df.columns:
        return _build_empty_mv_dashboard_data(strat_names)

    resolved_strategies = slice_df["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    stat_data = _compute_mv_strategy_statistics(
        slice_df, resolved_strategies, strat_names, strategy_groups
    )
    padded_dict = {
        name: ser.reset_index(drop=True)
        for name, ser in zip(strat_names, stat_data["returns"], strict=True)
    }
    cov_matrix = build_covariance_matrix(pd.DataFrame(padded_dict))
    mu_vector = np.array(stat_data["mus"], dtype=float)

    weights_ms = optimize_max_sharpe_weights(mu_vector, cov_matrix)
    weights_rp = optimize_risk_parity_weights(cov_matrix)
    mcr_ms, trc_ms, prc_ms = calculate_risk_contributions(weights_ms, cov_matrix)
    mcr_rp, trc_rp, prc_rp = calculate_risk_contributions(weights_rp, cov_matrix)

    strategies_res = []
    for idx, name in enumerate(strat_names):
        strategies_res.append(
            {
                "name": name,
                "trades_per_month": stat_data["t_per_m"][idx],
                "mu": stat_data["mus"][idx] * 100.0,
                "sigma": stat_data["sigmas"][idx] * 100.0,
                "sigma_d": stat_data["sigma_ds"][idx] * 100.0,
                "sharpe": stat_data["sharpes"][idx],
                "sortino": stat_data["sortinos"][idx],
                "weight_max_sharpe": weights_ms[idx] * 100.0,
                "weight_risk_parity": weights_rp[idx] * 100.0,
                "mcr_max_sharpe": mcr_ms[idx] * 100.0,
                "trc_max_sharpe": trc_ms[idx] * 100.0,
                "prc_max_sharpe": prc_ms[idx],
                "mcr_risk_parity": mcr_rp[idx] * 100.0,
                "trc_risk_parity": trc_rp[idx] * 100.0,
                "prc_risk_parity": prc_rp[idx],
                "trades_count": stat_data["counts"][idx],
            }
        )

    return {"strategies": strategies_res, "has_low_data": stat_data["has_low_data"]}


def _build_empty_mv_dashboard_data(strat_names: list[str]) -> dict[str, Any]:
    """Generates default zero-metric structure for empty state."""
    default_weight = (1.0 / len(strat_names)) * 100.0 if strat_names else 0.0
    return {
        "strategies": [
            {
                "name": name,
                "trades_per_month": 0.0,
                "mu": 0.0,
                "sigma": 0.0,
                "sigma_d": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "weight_max_sharpe": default_weight,
                "weight_risk_parity": default_weight,
                "mcr_max_sharpe": 0.0,
                "trc_max_sharpe": 0.0,
                "prc_max_sharpe": default_weight,
                "mcr_risk_parity": 0.0,
                "trc_risk_parity": 0.0,
                "prc_risk_parity": default_weight,
                "trades_count": 0,
            }
            for name in strat_names
        ],
        "has_low_data": True,
    }


def _compute_mv_strategy_statistics(
    slice_df: pd.DataFrame,
    resolved_strategies: pd.Series,
    strat_names: list[str],
    strategy_groups: dict[str, list[Any]],
) -> dict[str, Any]:
    """Computes return and risk statistics for mean-variance dashboard rows."""
    strat_returns, trades_counts, t_per_months = [], [], []
    mus, sigmas, sigma_ds, sharpes, sortinos = [], [], [], [], []
    has_low_data = False

    for name in strat_names:
        filters = strategy_groups[name]
        strat_slice_df = slice_df[resolved_strategies.isin(filters)]
        t_count = len(strat_slice_df)
        trades_counts.append(t_count)
        if t_count < LOW_DATA_THRESHOLD:
            has_low_data = True

        roi_series = extract_roi_series(strat_slice_df)
        avg_roi = float(roi_series.mean()) if not roi_series.empty else 0.0
        active_months = calculate_active_months(strat_slice_df)

        t_per_m = t_count / active_months
        t_per_months.append(t_per_m)
        mu_val = t_per_m * avg_roi
        mus.append(mu_val)
        strat_returns.append(roi_series)

        sig = float(roi_series.std(ddof=1)) if len(roi_series) > 1 else 0.0
        sigmas.append(sig)
        sig_d = compute_downside_deviation(roi_series) if not roi_series.empty else 0.0
        sigma_ds.append(sig_d)
        sharpes.append((mu_val / sig) if sig > CALC_EPSILON else 0.0)
        sortinos.append((mu_val / sig_d) if sig_d > CALC_EPSILON else 0.0)

    return {
        "returns": strat_returns,
        "counts": trades_counts,
        "t_per_m": t_per_months,
        "mus": mus,
        "sigmas": sigmas,
        "sigma_ds": sigma_ds,
        "sharpes": sharpes,
        "sortinos": sortinos,
        "has_low_data": has_low_data,
    }


def calculate_concurrent_exposure(
    strat_df: pd.DataFrame,
    strat_active: list[dict[str, Any]],
) -> tuple[int, int]:
    """Calculates max concurrent open positions and the 95th percentile utilization."""
    events = _collect_trade_events(strat_df, strat_active)
    if not events:
        return 0, 0

    events.sort(key=lambda x: (x[0], x[1]))
    max_concurrent, current_open = 0, 0
    for _, change in events:
        current_open += change
        max_concurrent = max(max_concurrent, current_open)

    percentile_95 = _calculate_95th_percentile(events)
    return max_concurrent, percentile_95


def _collect_trade_events(
    strat_df: pd.DataFrame, strat_active: list[dict[str, Any]]
) -> list[tuple[pd.Timestamp, int]]:
    """Gathers datetime entry (+1) and exit (-1) events."""
    events: list[tuple[pd.Timestamp, int]] = []
    if (
        not strat_df.empty
        and "entry_date" in strat_df.columns
        and "exit_date" in strat_df.columns
    ):
        entries = pd.to_datetime(strat_df["entry_date"], errors="coerce").dropna()
        exits = pd.to_datetime(strat_df["exit_date"], errors="coerce").dropna()
        for date in entries:
            events.append((date, 1))
        for date in exits:
            events.append((date, -1))

    for trade in strat_active:
        entry_date_val = trade.get("entry_date")
        if entry_date_val:
            try:
                parsed_date = pd.to_datetime(entry_date_val)
                events.append((parsed_date, 1))
            except (ValueError, TypeError):
                continue
    return events


def _calculate_95th_percentile(events: list[tuple[pd.Timestamp, int]]) -> int:
    """Calculates 95th percentile daily maximum concurrent trades."""

    def normalize_ts(ts: pd.Timestamp) -> pd.Timestamp:
        return (
            ts.tz_convert(None).normalize() if ts.tzinfo is not None else ts.normalize()
        )

    events_by_day: dict[pd.Timestamp, list[tuple[pd.Timestamp, int]]] = {}
    for date, change in events:
        day = normalize_ts(date)
        events_by_day.setdefault(day, []).append((date, change))

    sorted_days = sorted(events_by_day.keys())
    if not sorted_days:
        return 0

    daily_range = pd.date_range(start=sorted_days[0], end=sorted_days[-1], freq="D")
    daily_max: dict[pd.Timestamp, int] = {}
    current_open = 0
    for day in daily_range:
        day_events = events_by_day.get(day, [])
        if day_events:
            day_events.sort(key=lambda x: (x[0], x[1]))
            max_on_day = current_open
            for _, change in day_events:
                current_open += change
                max_on_day = max(max_on_day, current_open)
            daily_max[day] = max_on_day
        else:
            daily_max[day] = current_open

    return int(round(np.percentile(list(daily_max.values()), 95)))


def calculate_monthly_matrix_data(
    dataframe: pd.DataFrame,
    selected_year: int,
    strategy_groups: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Calculates all matrix rows, portfolio rows, and model comparison rows for a year."""
    year_df = (
        dataframe[dataframe["exit_date_dt"].dt.year == selected_year].copy()
        if not dataframe.empty
        else pd.DataFrame()
    )
    resolved_strategies = (
        year_df["strategy"].apply(lambda s: STRATEGY_ALIASES.get(str(s).lower(), s))
        if not year_df.empty
        else pd.Series(dtype=object)
    )

    matrix_rows = _build_strategy_matrix_rows(
        year_df, resolved_strategies, strategy_groups
    )
    portfolio_row = _build_portfolio_matrix_row(matrix_rows)
    portfolio_models_rows = _build_portfolio_models_matrix_rows(
        dataframe, selected_year, strategy_groups, matrix_rows, portfolio_row["months"]
    )
    return matrix_rows, portfolio_row, portfolio_models_rows


def _build_strategy_matrix_rows(
    year_df: pd.DataFrame,
    resolved_strategies: pd.Series,
    strategy_groups: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Builds monthly matrix rows for individual strategies."""
    matrix_rows: list[dict[str, Any]] = []
    for name, filters in strategy_groups.items():
        strat_df = (
            year_df[resolved_strategies.isin(filters)].copy()
            if not year_df.empty
            else pd.DataFrame()
        )
        monthly_pcts: list[float] = []
        for month in range(1, 13):
            month_df = (
                strat_df[strat_df["exit_date_dt"].dt.month == month]
                if not strat_df.empty
                else pd.DataFrame()
            )
            monthly_pcts.append(round(calculate_unweighted_monthly_pct(month_df), 1))

        factor = 1.0
        for val in monthly_pcts:
            factor *= 1.0 + val / 100.0
        gesamt_pct = (factor - 1.0) * 100.0
        matrix_rows.append(
            {"name": name, "months": monthly_pcts, "gesamt": round(gesamt_pct, 1)}
        )
    return matrix_rows


def _build_portfolio_matrix_row(matrix_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Builds equal-weighted aggregate portfolio row."""
    num_strats = len(matrix_rows)
    portfolio_monthly_pcts: list[float] = []
    for month_idx in range(12):
        month_sum = (
            sum(row["months"][month_idx] for row in matrix_rows)
            if num_strats > 0
            else 0.0
        )
        portfolio_monthly_pcts.append(
            round(month_sum / num_strats if num_strats > 0 else 0.0, 1)
        )

    factor = 1.0
    for val in portfolio_monthly_pcts:
        factor *= 1.0 + val / 100.0
    return {
        "name": "Portfolio",
        "months": portfolio_monthly_pcts,
        "gesamt": round((factor - 1.0) * 100.0, 1),
    }


def _build_portfolio_models_matrix_rows(
    dataframe: pd.DataFrame,
    selected_year: int,
    strategy_groups: dict[str, list[Any]],
    matrix_rows: list[dict[str, Any]],
    standard_monthly: list[float],
) -> list[dict[str, Any]]:
    """Builds model comparison rows (Standard, EV/M, Max-Sharpe, Risk Parity)."""
    evm_pcts, ms_pcts, rp_pcts = [], [], []

    for month_idx in range(12):
        if month_idx == 0:
            evm_pcts.append(standard_monthly[0])
            ms_pcts.append(standard_monthly[0])
            rp_pcts.append(standard_monthly[0])
            continue

        prior_month_end = pd.Timestamp(
            year=selected_year, month=month_idx, day=1
        ) + pd.offsets.MonthEnd(0)
        slice_df = (
            dataframe[dataframe["exit_date_dt"] <= prior_month_end]
            if not dataframe.empty
            else pd.DataFrame()
        )

        ev_w = calculate_evm_allocations(slice_df, strategy_groups)
        ms_w = calculate_mean_variance_allocations(
            slice_df, strategy_groups, model_type="max_sharpe"
        )
        rp_w = calculate_mean_variance_allocations(
            slice_df, strategy_groups, model_type="risk_parity"
        )

        evm_pcts.append(
            sum(ev_w[row["name"]] * row["months"][month_idx] for row in matrix_rows)
        )
        ms_pcts.append(
            sum(ms_w[row["name"]] * row["months"][month_idx] for row in matrix_rows)
        )
        rp_pcts.append(
            sum(rp_w[row["name"]] * row["months"][month_idx] for row in matrix_rows)
        )

    def compound(pcts: list[float]) -> float:
        factor = 1.0
        for p in pcts:
            factor *= 1.0 + p / 100.0
        return (factor - 1.0) * 100.0

    return [
        {
            "name": "Standard",
            "months": standard_monthly,
            "gesamt": round(compound(standard_monthly), 1),
        },
        {
            "name": "Frequenz-Modell (EV/M)",
            "months": [round(p, 1) for p in evm_pcts],
            "gesamt": round(compound(evm_pcts), 1),
        },
        {
            "name": "Risikoadjustiert (Max-Sharpe)",
            "months": [round(p, 1) for p in ms_pcts],
            "gesamt": round(compound(ms_pcts), 1),
        },
        {
            "name": "Risikoadjustiert (Risk Parity)",
            "months": [round(p, 1) for p in rp_pcts],
            "gesamt": round(compound(rp_pcts), 1),
        },
    ]


def calculate_strategy_risk_and_expectancy(
    strategy_dataframe: pd.DataFrame, initial_capital: float
) -> tuple[str, str]:
    """Computes average percentage risk and expectancy in R-multiples for a strategy.

    Args:
        strategy_dataframe: DataFrame containing trade records with entry_price, stop_loss, initial_size.
        initial_capital: Base starting portfolio capital.

    Returns:
        tuple[str, str]: Formatted average risk percentage and expectancy in R-multiples (or ('N/A', 'N/A')).
    """
    if strategy_dataframe.empty or initial_capital <= 0.0:
        return "N/A", "N/A"

    entry_prices = pd.to_numeric(
        strategy_dataframe["entry_price"], errors="coerce"
    ).fillna(0.0)
    stop_losses = pd.to_numeric(
        strategy_dataframe["stop_loss"], errors="coerce"
    ).fillna(0.0)
    initial_sizes = pd.to_numeric(
        strategy_dataframe["initial_size"], errors="coerce"
    ).fillna(0.0)

    valid_mask = (entry_prices > 0.0) & (stop_losses > 0.0) & (initial_sizes > 0.0)
    risks = (entry_prices - stop_losses).abs() * initial_sizes
    valid_risks = risks[valid_mask & (risks > 0.0)]

    if valid_risks.empty:
        return "N/A", "N/A"

    average_risk_dollar = float(valid_risks.mean())
    average_risk_percentage = (average_risk_dollar / initial_capital) * 100.0
    expectancy = metrics.calculate_expectancy(strategy_dataframe["realized_pnl"])
    expectancy_r = (
        expectancy / average_risk_dollar if average_risk_dollar > 0.0 else 0.0
    )
    return f"{average_risk_percentage:.2f}%", f"{expectancy_r:.2f} R"


def calculate_kelly_metrics(
    roi_series: pd.Series, strategy_active_trades: list[dict[str, Any]]
) -> tuple[float, float, float]:
    """Calculates Win Rate, Risk Reward Ratio, and Kelly Criterion from closed & active ROIs.

    Args:
        roi_series: Series of ROI values for closed trades.
        strategy_active_trades: List of active trade dictionaries.

    Returns:
        tuple[float, float, float]: Win rate, risk reward ratio, and Kelly criterion.
    """
    active_rois: list[float] = []
    for trade in strategy_active_trades:
        unrealized_pnl = float(trade.get("unrealized_pnl") or 0.0)
        entry_price = float(trade.get("entry_price") or 0.0)
        quantity = float(trade.get("quantity") or trade.get("initial_size") or 0.0)
        invested_capital = entry_price * quantity
        active_rois.append(
            unrealized_pnl / invested_capital if invested_capital > 0.0 else 0.0
        )

    combined_roi = pd.Series(roi_series.tolist() + active_rois, dtype=float)
    if combined_roi.empty:
        return 0.0, 0.0, 0.0

    win_rate = metrics.calculate_win_rate(combined_roi)
    risk_reward_ratio = metrics.calculate_risk_reward_ratio(combined_roi)
    kelly_criterion = metrics.calculate_kelly_criterion(win_rate, risk_reward_ratio)
    return win_rate, risk_reward_ratio, kelly_criterion


def calculate_win_loss_rois(
    strategy_dataframe: pd.DataFrame,
) -> tuple[float, float]:
    """Computes arithmetic average win ROI and loss ROI percentages.

    Args:
        strategy_dataframe: DataFrame of closed trades for a strategy.

    Returns:
        tuple[float, float]: Average win ROI percentage and average loss ROI percentage.
    """
    if strategy_dataframe.empty:
        return 0.0, 0.0

    win_trades = strategy_dataframe[strategy_dataframe["realized_pnl"] > 0]
    loss_trades = strategy_dataframe[strategy_dataframe["realized_pnl"] < 0]

    win_roi_series = extract_roi_series(win_trades)
    loss_roi_series = extract_roi_series(loss_trades)

    average_win_roi = (
        float(win_roi_series.mean() * 100.0) if not win_roi_series.empty else 0.0
    )
    average_loss_roi = (
        float(loss_roi_series.mean() * 100.0) if not loss_roi_series.empty else 0.0
    )
    return average_win_roi, average_loss_roi


def apply_depot_and_ev_allocations(
    strategies_data: list[dict[str, Any]],
    initial_capital: float = 100_000.0,
) -> None:
    """Scales Kelly values to depot limit and computes EV/M allocation percentages.

    Args:
        strategies_data: List of strategy dictionary metrics to mutate with allocations.
        initial_capital: Base portfolio capital (default: 100,000.0).
    """
    total_proposed = sum(
        max(0.0, float(strategy["metrics"]["kelly_criterion"]))
        for strategy in strategies_data
    )
    depot_multiplier = 1.0 / total_proposed if total_proposed > 1.0 else 1.0

    total_ev = sum(
        max(0.0, float(strategy["metrics"]["ev_per_month"]))
        for strategy in strategies_data
    )

    for strategy in strategies_data:
        raw_kelly = float(strategy["metrics"]["kelly_criterion"])
        strategy["metrics"]["suggested_allocation"] = (
            raw_kelly * depot_multiplier if raw_kelly > 0.0 else 0.0
        )
        raw_ev = float(strategy["metrics"]["ev_per_month"])
        ev_weight = (raw_ev / total_ev) if (raw_ev > 0.0 and total_ev > 0.0) else 0.0
        strategy["metrics"]["ev_allocation"] = ev_weight
        strategy["metrics"]["ev_allocation_100k"] = ev_weight * initial_capital


def calculate_benchmark_monthly_returns(
    benchmark_dataframe: pd.DataFrame,
    label: str,
    selected_year: int,
) -> dict[str, Any]:
    """Calculates monthly percentage returns and total annual return for a benchmark index.

    Args:
        benchmark_dataframe: Historical price DataFrame for the benchmark symbol.
        label: Benchmark display label (e.g. 'SPY (S&P 500)').
        selected_year: Target calendar year.

    Returns:
        dict[str, Any]: Dictionary containing label, 12-month returns, and total return.
    """
    if benchmark_dataframe.empty or "date" not in benchmark_dataframe.columns:
        return {"name": label, "months": [0.0] * 12, "gesamt": 0.0}

    working_dataframe = benchmark_dataframe.copy()
    working_dataframe["date_datetime"] = pd.to_datetime(
        working_dataframe["date"], errors="coerce"
    )
    working_dataframe = working_dataframe[
        working_dataframe["date_datetime"].dt.year == selected_year
    ].sort_values("date_datetime")

    if working_dataframe.empty:
        return {"name": label, "months": [0.0] * 12, "gesamt": 0.0}

    monthly_returns: list[float] = []
    for month in range(1, 13):
        month_dataframe = working_dataframe[
            working_dataframe["date_datetime"].dt.month == month
        ]
        if not month_dataframe.empty:
            open_price = float(
                month_dataframe.iloc[0]["open"] or month_dataframe.iloc[0]["close"]
            )
            close_price = float(month_dataframe.iloc[-1]["close"])
            percentage_return = (
                ((close_price - open_price) / open_price * 100.0)
                if open_price > 0.0
                else 0.0
            )
        else:
            percentage_return = 0.0
        monthly_returns.append(round(percentage_return, 1))

    year_open = float(
        working_dataframe.iloc[0]["open"] or working_dataframe.iloc[0]["close"]
    )
    year_close = float(working_dataframe.iloc[-1]["close"])
    total_annual_return = (
        ((year_close - year_open) / year_open * 100.0) if year_open > 0.0 else 0.0
    )
    return {
        "name": label,
        "months": monthly_returns,
        "gesamt": round(total_annual_return, 1),
    }


def calculate_monthly_trend_data(
    dataframe: pd.DataFrame,
    strategy_groups: dict[str, list[Any]],
    today: pd.Timestamp,
) -> dict[str, Any]:
    """Builds cumulative monthly performance series for Plotly charts.

    Calculates the YTD month-by-month cumulative realized PnL across all closed
    trades for the total portfolio and per individual strategy group.

    Args:
        dataframe: DataFrame of closed trades containing 'exit_date_dt', 'realized_pnl', 'strategy'.
        strategy_groups: Mapping from strategy group name to list of matching strategy identifiers.
        today: Current evaluation timestamp.

    Returns:
        dict[str, Any]: Formatted dictionary with 'dates', 'month_labels', 'aggregate', and 'strategies'.
    """
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    date_range = pd.date_range(start=start_of_year, end=today, freq="MS")
    dates_formatted = [date_val.strftime("%Y-%m-%d") for date_val in date_range]
    month_labels = [
        f"{GERMAN_MONTH_NAMES[date_val.month]} {date_val.year}"
        for date_val in date_range
    ]

    chart_dataframe = (
        dataframe[dataframe["exit_date_dt"] >= start_of_year].copy()
        if not dataframe.empty and "exit_date_dt" in dataframe.columns
        else pd.DataFrame()
    )

    if chart_dataframe.empty:
        return {
            "dates": dates_formatted,
            "month_labels": month_labels,
            "aggregate": [0.0] * len(date_range),
            "strategies": {name: [0.0] * len(date_range) for name in strategy_groups},
        }

    resolved_strategies = chart_dataframe["strategy"].apply(
        lambda s: STRATEGY_ALIASES.get(str(s).lower(), s)
    )

    aggregate_series: list[float] = []
    strategies_series: dict[str, list[float]] = {name: [] for name in strategy_groups}

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )
        month_slice = chart_dataframe[chart_dataframe["exit_date_dt"] <= slice_end]
        total_month_pnl = (
            float(pd.to_numeric(month_slice["realized_pnl"], errors="coerce").sum())
            if not month_slice.empty
            else 0.0
        )
        aggregate_series.append(total_month_pnl)

        for name, filters in strategy_groups.items():
            matching_rows = resolved_strategies.loc[month_slice.index].isin(filters)
            strat_slice = month_slice[matching_rows]
            strat_pnl = (
                float(pd.to_numeric(strat_slice["realized_pnl"], errors="coerce").sum())
                if not strat_slice.empty
                else 0.0
            )
            strategies_series[name].append(strat_pnl)

    return {
        "dates": dates_formatted,
        "month_labels": month_labels,
        "aggregate": aggregate_series,
        "strategies": strategies_series,
    }


def calculate_monthly_drawdown_max_intramonth(
    dataframe: pd.DataFrame,
    initial_capital: float,
    today: pd.Timestamp,
) -> dict[str, Any]:
    """Calculates monthly maximum intramonth portfolio drawdown series.

    For each month in YTD, determines the maximum peak-to-trough decline (lowest
    intramonth percentage drawdown) achieved by the cumulative portfolio equity.

    Args:
        dataframe: DataFrame of closed trades containing 'exit_date_dt' and 'realized_pnl'.
        initial_capital: Base starting portfolio capital.
        today: Current evaluation timestamp.

    Returns:
        dict[str, Any]: Formatted dictionary with 'dates', 'month_labels', and 'aggregate' drawdown ratios.
    """
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    date_range = pd.date_range(start=start_of_year, end=today, freq="MS")
    dates_formatted = [date_val.strftime("%Y-%m-%d") for date_val in date_range]
    month_labels = [
        f"{GERMAN_MONTH_NAMES[date_val.month]} {date_val.year}"
        for date_val in date_range
    ]

    if (
        dataframe.empty
        or "exit_date_dt" not in dataframe.columns
        or initial_capital <= 0.0
    ):
        return {
            "dates": dates_formatted,
            "month_labels": month_labels,
            "aggregate": [0.0] * len(date_range),
        }

    ytd_trades = (
        dataframe[dataframe["exit_date_dt"] >= start_of_year]
        .sort_values("exit_date_dt")
        .copy()
    )

    monthly_max_drawdowns: list[float] = []
    current_equity = initial_capital
    peak_equity = initial_capital

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        slice_end = (
            today
            if month_start.year == today.year and month_start.month == today.month
            else month_end
        )

        month_trades = ytd_trades[
            (ytd_trades["exit_date_dt"] >= month_start)
            & (ytd_trades["exit_date_dt"] <= slice_end)
        ]

        start_month_dd = (
            (current_equity - peak_equity) / peak_equity if peak_equity > 0.0 else 0.0
        )
        worst_month_dd = min(0.0, start_month_dd)

        if not month_trades.empty:
            for _, row in month_trades.iterrows():
                pnl = float(row.get("realized_pnl", 0.0) or 0.0)
                current_equity += pnl
                peak_equity = max(peak_equity, current_equity)
                trade_dd = (
                    (current_equity - peak_equity) / peak_equity
                    if peak_equity > 0.0
                    else 0.0
                )
                worst_month_dd = min(worst_month_dd, trade_dd)

        monthly_max_drawdowns.append(round(worst_month_dd, 4))

    return {
        "dates": dates_formatted,
        "month_labels": month_labels,
        "aggregate": monthly_max_drawdowns,
    }


def calculate_rolling_3m_metrics(
    dataframe: pd.DataFrame,
    initial_capital: float,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Calculates summary performance KPIs for the rolling 3-month period.

    Args:
        dataframe: DataFrame of closed trades with exit_date_dt, realized_pnl, etc.
        initial_capital: Base starting portfolio capital.
        as_of_date: Reference calculation date (defaults to current date).

    Returns:
        dict[str, object]: Dictionary containing rolling 3-month performance KPIs.
    """
    current_date = as_of_date if as_of_date is not None else pd.Timestamp.now()
    start_date = (current_date - pd.DateOffset(months=3)).normalize()

    if dataframe.empty or "exit_date_dt" not in dataframe.columns:
        period_df = pd.DataFrame(columns=dataframe.columns)
    else:
        period_df = dataframe[
            (dataframe["exit_date_dt"] >= start_date)
            & (dataframe["exit_date_dt"] <= current_date)
        ]

    date_range_label = (
        f"{start_date.strftime('%d.%m.%Y')} – {current_date.strftime('%d.%m.%Y')}"
    )

    if period_df.empty:
        return {
            "return_pct": 0.0,
            "net_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "sortino_ratio": 0.0,
            "avg_roi": 0.0,
            "trades_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "start_date": start_date.strftime("%d.%m.%Y"),
            "end_date": current_date.strftime("%d.%m.%Y"),
            "date_range_label": date_range_label,
        }

    pnl_series = pd.to_numeric(period_df["realized_pnl"], errors="coerce").fillna(0.0)
    net_pnl = float(pnl_series.sum())
    return_pct = (net_pnl / initial_capital) * 100.0 if initial_capital > 0 else 0.0

    roi_series = extract_roi_series(period_df)
    avg_roi = float(roi_series.mean() * 100.0) if not roi_series.empty else 0.0

    active_months = calculate_active_months(period_df)
    trades_per_year = (
        (len(roi_series) / active_months) * 12.0 if active_months > 0.0 else 252.0
    )

    sharpe_ratio = metrics.calculate_sharpe_ratio_from_roi(
        roi_series, trades_per_year=trades_per_year
    )
    sortino_ratio = metrics.calculate_sortino_ratio_from_roi(
        roi_series, trades_per_year=trades_per_year
    )
    profit_factor = metrics.calculate_profit_factor(pnl_series)

    win_count = int((pnl_series > 0.0).sum())
    loss_count = int((pnl_series < 0.0).sum())

    return {
        "return_pct": return_pct,
        "net_pnl": net_pnl,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "sortino_ratio": sortino_ratio,
        "avg_roi": avg_roi,
        "trades_count": len(period_df),
        "win_count": win_count,
        "loss_count": loss_count,
        "start_date": start_date.strftime("%d.%m.%Y"),
        "end_date": current_date.strftime("%d.%m.%Y"),
        "date_range_label": date_range_label,
    }
