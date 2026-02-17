
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional

# --- Palette ---
# Constants
COLOR_RED = "#ef4444"
COLOR_AMBER = "#f59e0b"
COLOR_GREEN = "#10b981" # Emerald
COLOR_BLUE = "#3b82f6"
COLOR_TRACK = "#f1f5f9" # Light gray track
COLOR_DARK = "#0f172a" 
COLOR_SLATE = "#64748b" 
COLOR_VIOLET = "#8b5cf6"
COLOR_GOLD = "#d4af37"

def _generate_minimal_gauge(
        value: float, 
        minimum_value: float, 
        maximum_value: float, 
        bar_color: str,
        suffix: str = "", 
        thresholds: Optional[List[float]] = None
    ) -> str:
    """
    Advanced Gauge v4: 
    - Full-length Light Slate track (#f1f5f9)
    - Active Layer Overlay
    - 1px White Separators at thresholds using 'white steps'
    - Thinner stroke (thickness 0.17)
    """
    
    # We will build the track using snippets of gray, separated by gaps.
    track_color = COLOR_TRACK
    gap_size = (maximum_value - minimum_value) * 0.008 # 0.8% gap for refined visibility
    
    current_start = minimum_value
    sorted_thresholds = sorted(thresholds) if thresholds else []
    
    track_steps = []
    
    # Force background to white (implicit), so gaps in 'steps' appear white
    
    if not sorted_thresholds:
         track_steps.append({'range': [minimum_value, maximum_value], 'color': track_color})
    else:
        for threshold in sorted_thresholds:
            # Segment up to threshold minus half gap
            end_segment = threshold - (gap_size / 2)
            if end_segment > current_start:
                track_steps.append({'range': [current_start, end_segment], 'color': track_color})
            
            # Start next segment after gap
            current_start = threshold + (gap_size / 2)
            
        # Final segment
        if current_start < maximum_value:
            track_steps.append({'range': [current_start, maximum_value], 'color': track_color})
            
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {
            "font": {"size": 34, "color": COLOR_DARK, "family": "Inter, sans-serif", "weight": 300},
            "suffix": suffix,
            "valueformat": ".2f" if not suffix else ".1f"
        },
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {
                'range': [minimum_value, maximum_value], 
                'visible': False, 
                'tickmode': 'array',
                'tickvals': []
            },
            'bar': {'color': bar_color, 'thickness': 0.17}, # Thinner stroke (15% reduction)
            'bgcolor': "white", 
            'borderwidth': 0,
            'shape': "angular",
            'steps': track_steps, 
            'threshold': {
                'line': {'color': "white", 'width': 0}, 
                'thickness': 0,
                'value': value 
            }
        }
    ))
    
    fig.update_layout(
        margin=dict(l=25, r=25, t=25, b=25),
        paper_bgcolor='rgba(0,0,0,0)',
        height=150,
        font={'family': "Inter, sans-serif"}
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})


def generate_backtest_charts(dates: pd.Series, equity: pd.Series, drawdown: pd.Series, benchmark_df: Optional[pd.DataFrame] = None, id_prefix: str = "base") -> tuple[str, str]:
    """
    Generates Equity Curve and Drawdown (Underwater) charts.
    Style: Spline, Gradient-like Fill (15% -> 0%), Minimal Grid.
    """
    if dates.empty:
        return "<div>No Data</div>", "<div>No Data</div>"
        
    # Create a local dataframe for sorting and processing
    dataframe = pd.DataFrame({
        'date': dates, 
        'equity': equity,
        'drawdown_pct': drawdown
    })
    
    # Ensure correct sorting
    sort_cols = []
    if 'date' in dataframe.columns:
        sort_cols.append('date')
        date_col = 'date'
    else:
        # Fallback if somehow date not in columns (shouldn't happen with our manual creation above)
        date_col = 'date'
        
    dataframe = dataframe.sort_values(sort_cols, ascending=True)
    dates = dataframe[date_col]
    equity = dataframe['equity']

    # Calculate Drawdown if needed (Always re-calculate to ensure consistency)
    peak = equity.cummax()
    drawdown = ((equity - peak) / peak) * 100 # Percentage (e.g. -5.0)

    
    # --- 1. Equity Curve ---
    fig_eq = go.Figure()
    
    # Add Benchmark first (so it's behind strategy)
    if benchmark_df is not None and not benchmark_df.empty:
        fig_eq.add_trace(go.Scatter(
            x=benchmark_df['date'].tolist(),
            y=benchmark_df['equity'].tolist(),
            mode='lines',
            name='SPY (Buy & Hold)',
            line=dict(color='#94a3b8', width=2, dash='dot'), # Slate-400, Dotted
            hovertemplate="<b>SPY</b>: $%{y:,.0f}<extra></extra>"
        ))
    
    fig_eq.add_trace(go.Scatter(
        x=dates.tolist(), 
        y=equity.tolist(),
        mode='lines',
        name='Strategy Equity',
        line=dict(color=COLOR_BLUE, width=3, shape='spline', smoothing=1.3),
        fill='tozeroy', 
        # Gradient: User wants 15% opacity. 
        fillcolor='rgba(59, 130, 246, 0.15)',
        hovertemplate="<b>Equity</b>: $%{y:,.0f}<extra></extra>"
    ))
    
    fig_eq.update_layout(
         margin=dict(l=60, r=60, t=40, b=40),
         paper_bgcolor='rgba(0,0,0,0)',
         plot_bgcolor='rgba(0,0,0,0)',
         height=280,
         autosize=True,
         font=dict(family="Inter, sans-serif", color="#94a3b8"),
         xaxis=dict(showgrid=False, zeroline=False),
         # Horizontal Grid: #f1f5f9, 1px width
         yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False, gridwidth=1, tickprefix="$", tickformat=",.0f"), 
         hovermode="x unified",
         showlegend=True,
         legend=dict(
             orientation="h",
             yanchor="bottom",
             y=1.02,
             xanchor="center",
             x=0.5
         )
    )
    
    # --- 2. Drawdown (Underwater Chart) ---
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=dates, 
        y=drawdown.tolist(),
        mode='lines',
        name='Drawdown',
        line=dict(color=COLOR_RED, width=1),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)', # Visible Red Fill
        hovertemplate="<b>Drawdown</b>: %{y:.2f}%<extra></extra>"
    ))
    
    fig_dd.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(showgrid=False, zeroline=False),
        # Y-Axis: 0 at top (max), negative below
        yaxis=dict(
            title="Drawdown %", 
            showgrid=True, 
            gridcolor='rgba(241, 245, 249, 0.5)', 
            zeroline=True, 
            zerolinecolor='#cbd5e1',
            gridwidth=1
        ),
        hovermode="x unified",
        showlegend=False
    )
    
    fig_eq.update_layout(autosize=True)
    fig_dd.update_layout(autosize=True)

    html_eq = fig_eq.to_html(full_html=False, include_plotlyjs=False, div_id=f'{id_prefix}-equity-curve-chart', config={'displayModeBar': False, 'responsive': True})
    html_dd = fig_dd.to_html(full_html=False, include_plotlyjs=False, div_id=f'{id_prefix}-underwater-chart', config={'displayModeBar': False, 'responsive': True})
    
    return html_eq, html_dd

def generate_profit_factor_gauge(value: float) -> str:
    # Logic: <1 Red, 1-1.5 Amber, 1.5-2 Emerald (Green), >2 Blue
    if value < 1.0:
        color = COLOR_RED
    elif value < 1.5:
        color = COLOR_AMBER
    elif value < 2.0:
        color = COLOR_GREEN
    else:
        color = COLOR_BLUE
    
    # Thresholds: 1.0, 1.5, 2.0
    return _generate_minimal_gauge(value, 0, 5.0, color, "", thresholds=[1.0, 1.5, 2.0])

def generate_win_rate_gauge(value_pct: float) -> str:
    # Logic: <35 Red, 35-60 Amber, >60 Blue (Matches SQN visual)
    if value_pct < 35.0:
        color = COLOR_RED
    elif value_pct < 60.0:
        color = COLOR_AMBER
    else:
        color = COLOR_BLUE
    
    # Thresholds: 35, 60 (Removed 75 to merge top buckets)
    return _generate_minimal_gauge(value_pct, 0, 100, color, "%", thresholds=[35.0, 60.0])

def generate_sqn_gauge(value: float) -> str:
    # Use centralized classification from analytics
    from .analytics import get_system_quality_classification
    
    classification = get_system_quality_classification(value)
    
    # Thresholds: 1.0, 2.0, 3.0, 5.0, 7.0
    return _generate_minimal_gauge(
        value, 
        0, 
        10.0, 
        classification.color, 
        "", 
        thresholds=[1.0, 2.0, 3.0, 5.0, 7.0]
    )

def generate_regime_overlay_chart(daily_df: pd.DataFrame) -> str:
    """
    Overlays Equity Curve (Line) with VIX (Area) and Safety Zones.
    """
    if daily_df.empty:
        return "<div>No Data</div>"

    
    fig = go.Figure()

    # Sanitize Data for VIX Plotting
    # Ensure strict sorting by date and numeric types to prevent "accumulating" artifacts
    dataframe = daily_df.copy()
    if 'date' in dataframe.columns:
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe.sort_values('date', inplace=True)
    
    # Ensure VIX is numeric and fill gaps
    if 'vix' in dataframe.columns:
        # Strict conversion to float
        dataframe['vix'] = pd.to_numeric(dataframe['vix'], errors='coerce').fillna(0.0).astype(float)
        # Cap VIX at 85 visually if needed, though axis range handles part of it
        dataframe['vix'] = dataframe['vix'].clip(0, 85)
    
    
    # Secondary Y-Axis for VIX (Inverted area?)
    # Let's use a filled area for VIX on secondary axis
    
    # Explicit conversion to list to remove any index ambiguity
    vix_data = dataframe['vix'].tolist()
    
    fig.add_trace(go.Scatter(
        x=dataframe['date'],
        y=vix_data,
        name="VIX Index",
        fill='tozeroy',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(148, 163, 184, 0.2)', # Slate-400, low opacity
        yaxis="y2",
        hovertemplate="<b>VIX</b>: %{y:.2f}<extra></extra>"
    ))

    # Equity Line
    fig.add_trace(go.Scatter(
        x=dataframe['date'].tolist(),
        y=dataframe['equity'].tolist(),
        name="Equity",
        line=dict(color=COLOR_BLUE, width=2.5),
        yaxis="y1",
        hovertemplate="<b>Equity</b>: $%{y:,.0f}<extra></extra>"
    ))
    
    # Highlight Safety Zones (where safety_active is True)
    safety_days = dataframe[dataframe['safety_active']]
    if not safety_days.empty:
        fig.add_trace(go.Scatter(
            x=safety_days['date'].tolist(),
            y=[dataframe['equity'].max()] * len(safety_days), # Place markers at top
            mode='markers',
            name='Safety Active',
            marker=dict(symbol='triangle-down', size=8, color=COLOR_RED),
            yaxis='y1',
            hoverinfo='skip'
        ))

    fig.update_layout(
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title="Equity ($)", 
            showgrid=True, 
            gridcolor='#f1f5f9',
            tickprefix="$",
            tickformat=",.0f"
        ),
        yaxis2=dict(
            title="VIX", 
            overlaying="y", 
            side="right", 
            showgrid=False,
            range=[0, 85], # FIXED RANGE [0, 85] per User Request/Best Practice
            fixedrange=True
        ),
        showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    autosize=True
)
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False, 'responsive': True})

def generate_price_of_safety_chart(
        unconstrained_dataframe: pd.DataFrame, 
        constrained_dataframe: pd.DataFrame, 
        safety_dataframe: pd.DataFrame,
        spy_dataframe: pd.DataFrame = pd.DataFrame(),
        qqq_dataframe: pd.DataFrame = pd.DataFrame()
    ) -> str:
    """
    Compares Equity Curves: Unconstrained vs Constrained vs Safety Switch vs Benchmarks.
    """
    fig = go.Figure()
    
    # 1. Benchmarks
    if not spy_dataframe.empty:
        fig.add_trace(go.Scatter(x=spy_dataframe['date'].tolist(), y=spy_dataframe['equity'].tolist(), name="SPY", line=dict(color="#94a3b8")))
    if not qqq_dataframe.empty:
        fig.add_trace(go.Scatter(x=qqq_dataframe['date'].tolist(), y=qqq_dataframe['equity'].tolist(), name="QQQ", line=dict(color="#a855f7")))

    # 2. Strategies
    if not unconstrained_dataframe.empty:
        x_col = 'date' if 'date' in unconstrained_dataframe.columns else 'exit_date'
        fig.add_trace(go.Scatter(x=unconstrained_dataframe[x_col].tolist(), y=unconstrained_dataframe['equity'].tolist(), name="Unconstrained (Leveraged Kelly)", line=dict(color=COLOR_AMBER, dash='dot')))
        
    if not constrained_dataframe.empty:
        x_col = 'date' if 'date' in constrained_dataframe.columns else 'exit_date'
        fig.add_trace(go.Scatter(x=constrained_dataframe[x_col].tolist(), y=constrained_dataframe['equity'].tolist(), name="Constrained (Reality / Budget Cap)", line=dict(color=COLOR_BLUE)))
        
    if not safety_dataframe.empty:
        x_col = 'date' if 'date' in safety_dataframe.columns else 'exit_date'
        fig.add_trace(go.Scatter(x=safety_dataframe[x_col].tolist(), y=safety_dataframe['equity'].tolist(), name="With Safety Switch", line=dict(color=COLOR_GREEN, width=3)))
        
    fig.update_layout(
        height=320,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Equity ($)", 
            tickprefix="$",
            tickformat=",.0f",
            gridcolor='#f1f5f9'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
        autosize=True
    )
    
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False, 'responsive': True})

def generate_exposure_heatmap(daily_df: pd.DataFrame, pressure_pct: float = 0.0) -> str:
    """
    Stacked Area Chart of Exposure per Strategy + Overflow (Ghost Layer).
    """
    if daily_df.empty: return "<div>No Data</div>"

    
    fig = go.Figure()
    
    import numpy as np
    
    # Identify exposure columns
    exp_cols = [c for c in daily_df.columns if c.startswith("exposure_")]
    # Identify overflow columns
    overflow_cols = [c.replace("exposure_", "overflow_") for c in exp_cols]
    has_overflow = any(c in daily_df.columns for c in overflow_cols)
    
    fig = go.Figure()
    
    dates = daily_df['date'].values
    n = len(dates)
    cumulative_y = np.zeros(n)
    total_denied_y = np.zeros(n)

    # 1. Strategies (Capped at 100% cumulative)
    # High Contrast Palette: Blue, Cyan, Amber, Green, Pink, Gold, Red
    palette = ["#3b82f6", "#06b6d4", "#f59e0b", "#10b981", "#ec4899", "#d4af37", "#ef4444"]
    
    for i, col in enumerate(exp_cols):
        strat_name = col.replace("exposure_", "")
        color = palette[i % len(palette)]
        
        raw_vals = daily_df[col].fillna(0).to_numpy() * 100
        
        # Calculate what fits in the 100% budget
        room = np.maximum(0, 100 - cumulative_y)
        fits = np.minimum(raw_vals, room)
        excess = np.maximum(0, raw_vals - room)
        
        total_denied_y += excess
        
        # New top of this layer
        new_cumulative = cumulative_y + fits
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=new_cumulative,
            name=strat_name,
            mode='lines',
            line=dict(width=0), # Remove lines for cleaner area look
            fill='tonexty' if len(fig.data) > 0 else 'tozeroy',
            fillcolor=color, # Solid color
            hovertemplate=f"<b>{strat_name}</b><br>Stacked: %{{y:.1f}}%<extra></extra>"
        ))
        cumulative_y = new_cumulative

    # 2. Add theoretical overflows to denied trace
    if has_overflow:
        for col in overflow_cols:
            if col in daily_df.columns:
                total_denied_y += daily_df[col].fillna(0).to_numpy() * 100

    # 3. Denied Layer (Solid Slate, starting at current cumulative top)
    if np.sum(total_denied_y) > 0:
        denied_top = cumulative_y + total_denied_y
        fig.add_trace(go.Scatter(
            x=dates,
            y=denied_top,
            name="Total Denied (Excess Demand)",
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='#475569', # Solid Slate-600 (distinct from background)
            hovertemplate="<b>Total Denied</b><br>Stacked: %{y:.1f}%<extra></extra>"
        ))

    # Add Budget Limit Line (Hard Cap at 100%) - Dotted Red
    fig.add_trace(go.Scatter(
        x=dates,
        y=[100] * n,
        name="Budget Cap (100%)",
        mode='lines',
        line=dict(color=COLOR_RED, width=2, dash="dash"),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        yaxis=dict(
            title="Utilization %", 
            range=[0, 150], 
            showgrid=True, 
            gridcolor='#f1f5f9'
        ),
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.2, 
            xanchor="center", 
            x=0.5,
            itemwidth=40, # Increased for better spacing
            itemsizing="constant",
            traceorder="normal", # Maintain entry order (Actuals then Denied)
            font=dict(size=11)
        ),
        width=None
    )
    fig.update_layout(autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False, 'responsive': True})
    
def generate_risk_reward_scatter(daily_df: pd.DataFrame) -> str:
    """
    Monthly Scatter Plot: Return vs MaxDD.
    """
    if daily_df.empty: return "<div>No Data</div>"
    
    try:
        # Resample to Monthly
        dataframe = daily_df.copy()
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe.set_index('date', inplace=True)
        
        # Pandas < 2.2 uses 'M', >= 2.2 uses 'ME'
        try: monthly = dataframe['equity'].resample('ME').last()
        except ValueError: monthly = dataframe['equity'].resample('M').last()
        
        if monthly.empty: 
             return "<div>No Data (Insufficient History)</div>"
        
        # Calculate Percentage Change
        # Standard pct_change() makes the first item NaN (losing Jan return)
        # We manually compute returns relative to the first available equity value (Day 1 EOD)
        
        start_equity = dataframe['equity'].iloc[0]
        monthly_ret = monthly.pct_change()
        
        # Fix the first NaN: (Month1_End - Start_Equity) / Start_Equity
        if len(monthly) > 0:
            first_ret = (monthly.iloc[0] - start_equity) / start_equity
            monthly_ret.iloc[0] = first_ret
            
        monthly_ret = monthly_ret * 100
        
    except Exception as e:
        return f"<div>Error generating chart: {e}</div>"
    
    # For now, let's just plot Return distribution
    fig = go.Figure()
    
    # Flatten values and convert to standard Python list to disable binary encoding
    y_values = monthly_ret.values.flatten().tolist() if hasattr(monthly_ret, 'values') else list(monthly_ret)
    
    fig.add_trace(go.Box(
        y=y_values,
        name="Monthly Returns",
        boxpoints='all',
        jitter=0.5,
        pointpos=-1.8, # Points to the left
        marker=dict(color=COLOR_BLUE, size=4),
        line=dict(color=COLOR_BLUE)
    ))
    
    fig.update_layout(
        margin=dict(l=60, r=60, t=40, b=40), # Standardized margins
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        showlegend=False,
        yaxis=dict(
            title="Return %", 
            showgrid=True, 
            gridcolor='rgba(241, 245, 249, 0.5)', # Subtle grid
            zeroline=True,
            zerolinecolor='#000000', # Solid Black as requested
            zerolinewidth=2
        ),
        autosize=True
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False, 'responsive': True})
