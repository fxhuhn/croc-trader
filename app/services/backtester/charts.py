
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, List, Dict, Optional

# --- Palette ---
# Constants
COLOR_RED = "#ef4444"
COLOR_AMBER = "#f59e0b"
COLOR_GREEN = "#10b981" # Emerald
COLOR_BLUE = "#3b82f6"
COLOR_TRACK = "#f1f5f9" # Light gray track
COLOR_DARK = "#0f172a" 

def _generate_minimal_gauge(
        value: float, 
        min_val: float, 
        max_val: float, 
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
    gap_size = (max_val - min_val) * 0.008 # 0.8% gap for refined visibility
    
    current_start = min_val
    sorted_thresholds = sorted(thresholds) if thresholds else []
    
    track_steps = []
    
    # Force background to white (implicit), so gaps in 'steps' appear white
    
    if not sorted_thresholds:
         track_steps.append({'range': [min_val, max_val], 'color': track_color})
    else:
        for th in sorted_thresholds:
            # Segment up to threshold minus half gap
            end_segment = th - (gap_size / 2)
            if end_segment > current_start:
                track_steps.append({'range': [current_start, end_segment], 'color': track_color})
            
            # Start next segment after gap
            current_start = th + (gap_size / 2)
            
        # Final segment
        if current_start < max_val:
            track_steps.append({'range': [current_start, max_val], 'color': track_color})
            
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
                'range': [min_val, max_val], 
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


def generate_backtest_charts(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Generates Equity Curve and Drawdown charts.
    Style: Spline, Gradient-like Fill (15% -> 0%), Minimal Grid.
    """
    if df.empty:
        return "<div>No Data</div>", "<div>No Data</div>"
        
    dates = df['exit_date']
    equity = df['equity']
    drawdown = df['drawdown_pct'] * 100 
    
    # --- 1. Equity Curve ---
    fig_eq = go.Figure()
    
    fig_eq.add_trace(go.Scatter(
        x=dates, 
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color=COLOR_BLUE, width=2, shape='spline', smoothing=1.3),
        fill='tozeroy', 
        # Gradient: User wants 15% opacity. 
        fillcolor='rgba(59, 130, 246, 0.15)' 
    ))
    
    fig_eq.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(showgrid=False, zeroline=False),
        # Horizontal Grid: #f1f5f9, 1px width
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False, gridwidth=1), 
        hovermode="x unified",
        showlegend=False
    )
    
    # --- 2. Drawdown ---
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=dates, 
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color=COLOR_RED, width=1, shape='spline', smoothing=1.3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)' # Ultra light fill
    ))
    
    fig_dd.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(241, 245, 249, 0.5)', zeroline=False, gridwidth=1),
        hovermode="x unified",
        showlegend=False
    )
    
    html_eq = fig_eq.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})
    html_dd = fig_dd.to_html(full_html=False, include_plotlyjs=False, config={'displayModeBar': False})
    
    return html_eq, html_dd

def generate_profit_factor_gauge(value: float) -> str:
    # Logic: <1 Red, 1-1.5 Amber, 1.5-2 Emerald (Green), >2 Blue
    if value < 1.0: color = COLOR_RED
    elif value < 1.5: color = COLOR_AMBER
    elif value < 2.0: color = COLOR_GREEN
    else: color = COLOR_BLUE
    
    # Thresholds: 1.0, 1.5, 2.0
    return _generate_minimal_gauge(value, 0, 5.0, color, "", thresholds=[1.0, 1.5, 2.0])

def generate_win_rate_gauge(value_pct: float) -> str:
    # Logic: <35 Red, 35-60 Amber, >60 Blue (Matches SQN visual)
    if value_pct < 35.0: color = COLOR_RED
    elif value_pct < 60.0: color = COLOR_AMBER
    else: color = COLOR_BLUE
    
    # Thresholds: 35, 60 (Removed 75 to merge top buckets)
    return _generate_minimal_gauge(value_pct, 0, 100, color, "%", thresholds=[35.0, 60.0])

def generate_sqn_gauge(value: float) -> str:
    # Logic: <1.7 Red, 1.7-2 Amber, 2-3 Green, >3 Blue
    if value < 1.7: color = COLOR_RED
    elif value < 2.0: color = COLOR_AMBER
    elif value < 3.0: color = COLOR_GREEN
    else: color = COLOR_BLUE
    
    # Thresholds: 1.7, 2.0, 3.0
    return _generate_minimal_gauge(value, 0, 6.0, color, "", thresholds=[1.7, 2.0, 3.0])
