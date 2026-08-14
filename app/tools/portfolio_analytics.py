"""Pure domain and mathematical calculations for portfolio analytics and allocation models.

Follows the Functional Core pattern: deterministic calculations without side effects or I/O.
"""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]  # Standard un-typed library import

from ..const import STRATEGY_ALIASES
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
