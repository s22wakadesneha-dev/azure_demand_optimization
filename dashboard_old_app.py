"""
Milestone 4 - Task 2: Azure Demand Forecast Dashboard
Infosys Springboard | Azure Consumer Demand Forecasting Project
Built with Streamlit + Plotly for professional interactive visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Azure Demand Forecast | Infosys",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Professional Dark Azure Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* Main background */
  .stApp { background: #0a0e1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid #1e2d4a;
  }

  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stMultiSelect label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #c8d8f0 !important;
  }

  /* Header banner */
  .dashboard-header {
    background: linear-gradient(135deg, #0d2045 0%, #0a3d7a 50%, #0078d4 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid #1a5fa8;
    box-shadow: 0 4px 32px rgba(0,120,212,0.25);
  }

  .header-left h1 {
    font-size: 26px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
  }

  .header-left p {
    font-size: 13px;
    color: rgba(255,255,255,0.65);
    margin: 4px 0 0 0;
  }

  .header-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 8px;
    padding: 10px 18px;
    text-align: right;
  }

  .header-badge .badge-label {
    font-size: 11px;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .header-badge .badge-value {
    font-size: 15px;
    font-weight: 600;
    color: #7ecfff;
    margin-top: 2px;
  }

  /* KPI Cards */
  .kpi-card {
    background: #111827;
    border-radius: 10px;
    padding: 20px 22px;
    border: 1px solid #1e2d4a;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
  }

  .kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,120,212,0.2);
  }

  .kpi-card .accent-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 10px 10px 0 0;
  }

  .kpi-card .kpi-icon { font-size: 22px; margin-bottom: 8px; }
  .kpi-card .kpi-label { font-size: 11px; color: #6b7a99; text-transform: uppercase; letter-spacing: 1px; }
  .kpi-card .kpi-value { font-size: 30px; font-weight: 700; color: #e8f0ff; line-height: 1.1; margin: 4px 0; }
  .kpi-card .kpi-delta { font-size: 12px; }
  .kpi-card .kpi-delta.up   { color: #3fb950; }
  .kpi-card .kpi-delta.down { color: #f85149; }
  .kpi-card .kpi-delta.neutral { color: #8b949e; }

  /* Section headers */
  .section-title {
    font-size: 14px;
    font-weight: 600;
    color: #7ecfff;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2d4a;
  }

  /* Alert boxes */
  .alert-box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 13px;
  }

  .alert-warning {
    background: rgba(248,161,0,0.1);
    border: 1px solid rgba(248,161,0,0.3);
    color: #f8c200;
  }

  .alert-success {
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.3);
    color: #3fb950;
  }

  .alert-danger {
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.3);
    color: #f85149;
  }

  /* Table styling */
  .stDataFrame { border-radius: 8px; overflow: hidden; }
  .stDataFrame thead { background: #1a2640 !important; }

  /* Metric override */
  [data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 16px;
  }

  /* Divider */
  hr { border-color: #1e2d4a; }

  /* Status pill */
  .status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }

  .status-live   { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
  .status-drift  { background: rgba(248,81,73,0.15); color: #f85149; border: 1px solid #f85149; }
  .status-warn   { background: rgba(248,161,0,0.15); color: #f8a100; border: 1px solid #f8a100; }

  /* Sidebar divider */
  .sidebar-section {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3a4a6a;
    padding: 12px 0 6px 0;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 10px;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
REGIONS   = ['Central-India', 'East-US', 'West-US', 'East-Asia', 'UK-South']
SERVICES  = ['Compute', 'Storage', 'Networking']
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='IBM Plex Sans', color='#c8d8f0', size=12),
    xaxis=dict(gridcolor='#1a2640', linecolor='#1e2d4a', tickfont=dict(color='#6b7a99')),
    yaxis=dict(gridcolor='#1a2640', linecolor='#1e2d4a', tickfont=dict(color='#6b7a99')),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2d4a', borderwidth=1,
                font=dict(color='#c8d8f0')),
    margin=dict(l=0, r=0, t=30, b=0),
)

COLORS = {
    'primary':   '#0078d4',
    'accent':    '#7ecfff',
    'forecast':  '#f78166',
    'actual':    '#3fb950',
    'warning':   '#f8a100',
    'danger':    '#f85149',
    'purple':    '#a371f7',
    'teal':      '#39d353',
}
def hex_to_rgba(hex_color, alpha=0.7):
    hex_color = hex_color.replace('#', '')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 1000
    start = datetime(2021, 1, 1)
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=i * 3)
        region  = REGIONS[i % len(REGIONS)]
        service = SERVICES[i % len(SERVICES)]
        actual  = 200 + np.random.uniform(0, 300) + np.sin(i / 24) * 60
        prov    = actual * np.random.uniform(1.1, 1.5)
        pred    = actual * np.random.uniform(0.85, 1.15)
        rows.append({
            'timestamp':           ts,
            'region':              region,
            'service_type':        service,
            'usage_units':         round(actual, 2),
            'prediction':     round(pred, 2),
            'provisioned_capacity':round(prov, 2),
            'cost_usd':            round(actual * 0.45, 2),
            'availability_pct':    round(np.random.uniform(96, 100), 2),
            'economic_growth_index': round(np.random.uniform(0.8, 1.4), 3),
            'marketing_index':     round(np.random.uniform(0.5, 1.2), 3),
            'it_spending_growth':  round(np.random.uniform(0.9, 1.3), 3),
            'is_holiday':          int(np.random.choice([0, 1], p=[0.9, 0.1])),
        })
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df


def compute_metrics(df):
    if 'prediction' not in df.columns:
        df['prediction'] = df['usage_units'] * np.random.uniform(0.9, 1.1, len(df))

    actual    = df['usage_units'].dropna()
    predicted = df['prediction'].dropna()
    n = min(len(actual), len(predicted))
    a, p = actual.iloc[:n].values, predicted.iloc[:n].values
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mae  = float(np.mean(np.abs(a - p)))
    bias = float(np.mean(p - a))
    da   = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(p)))) * 100
    util = float(df['prediction'].sum() / df['provisioned_capacity'].replace(0, np.nan).sum()) * 100 \
           if 'provisioned_capacity' in df.columns else 0
    return dict(rmse=rmse, mae=mae, bias=bias, directional_accuracy=da, utilisation=util)


def kpi_card(icon, label, value, delta_text, delta_type, accent_color):
    return f"""
    <div class="kpi-card">
      <div class="accent-bar" style="background:{accent_color}"></div>
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta {delta_type}">{delta_text}</div>
    </div>
    """


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
      <div style="font-size:36px">☁️</div>
      <div style="font-size:15px; font-weight:700; color:#7ecfff; margin-top:6px;">Azure Forecast</div>
      <div style="font-size:11px; color:#4a5a7a; margin-top:2px;">Infosys Springboard · M4</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">📁 Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio(
    "Data Source",
    ["📊 Use Demo Data", "📂 Upload CSV"],
    label_visibility="collapsed"
)

    df_raw = None
    if data_source == "📂 Upload CSV":
        uploaded = st.file_uploader("Upload forecast_output.csv", type=["csv"], label_visibility="collapsed")
        if uploaded:
            df_raw = load_csv(uploaded)
            st.success(f"✅ Loaded {len(df_raw):,} rows")
        else:
            st.info("Upload your `forecast_output.csv`")
    else:
        df_raw = generate_demo_data()
        st.success("✅ Demo data loaded (1,000 records)")

    if df_raw is not None:
        st.markdown('<div class="sidebar-section">🔍 Filters</div>', unsafe_allow_html=True)

        all_regions  = sorted(df_raw['region'].dropna().unique().tolist()) if 'region' in df_raw.columns else []
        all_services = sorted(df_raw['service_type'].dropna().unique().tolist()) if 'service_type' in df_raw.columns else []

        sel_regions = st.multiselect("Region", all_regions, default=all_regions)
        sel_services = st.multiselect("Service Type", all_services, default=all_services)

        if 'timestamp' in df_raw.columns:
            min_d = df_raw['timestamp'].min().date()
            max_d = df_raw['timestamp'].max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d),
                                       min_value=min_d, max_value=max_d)
        else:
            date_range = None

        st.markdown('<div class="sidebar-section">⚙️ Settings</div>', unsafe_allow_html=True)
        rmse_baseline  = st.number_input("RMSE Baseline", value=130.0, step=1.0)
        alert_margin   = st.slider("Drift Alert Margin %", 10, 50, 20)
        capacity_alert = st.slider("Capacity Alert Threshold %", 50, 100, 80)

        st.markdown('<div class="sidebar-section">📄 Export</div>', unsafe_allow_html=True)
        if st.button("⬇️ Download Filtered Data as CSV", use_container_width=True):
            st.download_button(
                label="Save CSV",
                data=df_raw.to_csv(index=False),
                file_name="filtered_forecast.csv",
                mime="text/csv"
            )

    st.markdown("""
    <div style="position:fixed; bottom:20px; font-size:10px; color:#2a3a5a; text-align:center; width:220px;">
      Azure Demand Forecast Dashboard v1.0<br/>Infosys Springboard · Milestone 4
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# NO DATA STATE
# ─────────────────────────────────────────────
if df_raw is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
      <div style="font-size:64px; margin-bottom:16px;">☁️</div>
      <h2 style="color:#7ecfff; font-size:24px;">Azure Demand Forecast Dashboard</h2>
      <p style="color:#6b7a99; font-size:15px; margin-top:8px;">
        Select <strong style="color:#c8d8f0">Demo Data</strong> or upload your 
        <strong style="color:#c8d8f0">forecast_output.csv</strong> to get started.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()
if sel_regions  and 'region'       in df.columns: df = df[df['region'].isin(sel_regions)]
if sel_services and 'service_type' in df.columns: df = df[df['service_type'].isin(sel_services)]
if date_range and len(date_range) == 2 and 'timestamp' in df.columns:
    df = df[(df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])]

if df.empty:
    st.warning("⚠️ No data matches the current filters. Adjust the filters in the sidebar.")
    st.stop()


# ─────────────────────────────────────────────
# COMPUTE METRICS
# ─────────────────────────────────────────────
metrics = compute_metrics(df)
total_forecast = df['prediction'].sum()
peak_row       = df.loc[df['prediction'].idxmax()]
peak_date      = str(peak_row['timestamp'])[:10] if 'timestamp' in df.columns else '—'
peak_value     = peak_row['prediction']
avg_actual     = df['usage_units'].mean()
avg_pred       = df['prediction'].mean()
growth_pct     = ((avg_pred - avg_actual) / avg_actual * 100) if avg_actual > 0 else 0
drift_detected = metrics['rmse'] > rmse_baseline * (1 + alert_margin / 100)
n_breaches     = int((df['prediction'] / df['provisioned_capacity'].replace(0, np.nan) > capacity_alert / 100).sum()) \
                 if 'provisioned_capacity' in df.columns else 0


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
now_str = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
status_html = '<span class="status-pill status-drift">⚠ DRIFT</span>' if drift_detected \
              else '<span class="status-pill status-live">● LIVE</span>'

st.markdown(f"""
<div class="dashboard-header">
  <div class="header-left">
    <h1>☁️ Azure Demand Forecast Dashboard</h1>
    <p>Infosys Springboard &nbsp;·&nbsp; Milestone 4 &nbsp;·&nbsp; Forecast Integration & Capacity Planning &nbsp;·&nbsp; {status_html}</p>
  </div>
  <div style="display:flex; gap:16px;">
    <div class="header-badge">
      <div class="badge-label">Last Refresh</div>
      <div class="badge-value">{now_str}</div>
    </div>
    <div class="header-badge">
      <div class="badge-label">Records</div>
      <div class="badge-value">{len(df):,}</div>
    </div>
    <div class="header-badge">
      <div class="badge-label">Regions Active</div>
      <div class="badge-value">{df['region'].nunique() if 'region' in df.columns else '—'}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────
if drift_detected:
    st.markdown(f"""
    <div class="alert-box alert-danger">
      <span style="font-size:18px">🚨</span>
      <div><strong>Model Drift Detected</strong> — Current RMSE ({metrics['rmse']:.1f}) exceeds the 
      {alert_margin}% alert threshold above baseline ({rmse_baseline:.1f}). 
      <strong>Retraining recommended.</strong> Run <code>python monitoring.py --retrain</code></div>
    </div>
    """, unsafe_allow_html=True)

if n_breaches > 0:
    st.markdown(f"""
    <div class="alert-box alert-warning">
      <span style="font-size:18px">⚠️</span>
      <div><strong>Capacity Alert</strong> — {n_breaches} forecast records exceed 
      {capacity_alert}% of provisioned capacity. Review infrastructure provisioning immediately.</div>
    </div>
    """, unsafe_allow_html=True)

if not drift_detected and n_breaches == 0:
    st.markdown("""
    <div class="alert-box alert-success">
      <span style="font-size:18px">✅</span>
      <div><strong>All Systems Normal</strong> — Model performance within acceptable limits. 
      No capacity breaches detected.</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)

cards = [
    (c1, "📦", "Total Forecast Demand",  f"{total_forecast:,.0f}",
     f"Across {len(df):,} records", "neutral", COLORS['primary']),
    (c2, "📅", "Peak Forecast Date",     peak_date,
     f"Peak: {peak_value:,.0f} units", "neutral", COLORS['accent']),
    (c3, "📈", "Forecast Growth",        f"{growth_pct:+.1f}%",
     "vs actual average", "up" if growth_pct > 0 else "down", COLORS['teal']),
    (c4, "🎯", "Model RMSE",             f"{metrics['rmse']:.1f}",
     f"Baseline: {rmse_baseline:.1f}", "danger" if drift_detected else "up",
     COLORS['danger'] if drift_detected else COLORS['primary']),
    (c5, "📐", "MAE",                    f"{metrics['mae']:.1f}",
     "Mean Absolute Error", "neutral", COLORS['purple']),
    (c6, "🧭", "Directional Accuracy",   f"{metrics['directional_accuracy']:.1f}%",
     "Correct trend direction", "up" if metrics['directional_accuracy'] > 55 else "down",
     COLORS['teal']),
]

for col, icon, label, value, delta, dtype, color in cards:
    with col:
        st.markdown(kpi_card(icon, label, value, delta, dtype, color), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📉 Analysis Panels</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Actual vs Forecast",
    "🌍  Regional Analysis",
    "⚙️  Service Breakdown",
    "🔍  Monitoring & Drift",
    "📋  Data Explorer",
])


# ══════════════════════════════════════════
# TAB 1 — TIME SERIES
# ══════════════════════════════════════════
with tab1:
    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt2:
        agg_by    = st.selectbox("Aggregate By", ["Hourly", "Daily", "Weekly", "Monthly"])
        show_band = st.checkbox("Show Confidence Band", value=True)

    ts_df = df[['timestamp', 'usage_units', 'prediction']].dropna().copy()
    ts_df = ts_df.sort_values('timestamp')

    freq_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_map[agg_by]
    ts_agg = ts_df.set_index('timestamp').resample(freq).agg({
        'usage_units':     ['mean', 'min', 'max'],
        'prediction': ['mean', 'min', 'max'],
    }).reset_index()
    ts_agg.columns = ['timestamp', 'actual_mean', 'actual_min', 'actual_max',
                      'pred_mean', 'pred_min', 'pred_max']

    fig_ts = go.Figure()

    if show_band:
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([ts_agg['timestamp'], ts_agg['timestamp'][::-1]]),
            y=pd.concat([ts_agg['pred_max'], ts_agg['pred_min'][::-1]]),
            fill='toself',
            fillcolor='rgba(247,129,102,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Forecast Band',
            showlegend=True
        ))
        fig_ts.add_trace(go.Scatter(
            x=pd.concat([ts_agg['timestamp'], ts_agg['timestamp'][::-1]]),
            y=pd.concat([ts_agg['actual_max'], ts_agg['actual_min'][::-1]]),
            fill='toself',
            fillcolor='rgba(0,120,212,0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Actual Band',
            showlegend=True
        ))

    fig_ts.add_trace(go.Scatter(
        x=ts_agg['timestamp'], y=ts_agg['actual_mean'],
        name='Actual Demand', line=dict(color=COLORS['primary'], width=2),
        mode='lines'
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_agg['timestamp'], y=ts_agg['pred_mean'],
        name='Predicted Demand',
        line=dict(color=COLORS['forecast'], width=2, dash='dot'),
        mode='lines'
    ))

    fig_ts.update_layout(
    **PLOTLY_THEME,
    height=420,
    title=dict(
        text=f"Actual vs Predicted Demand ({agg_by})",
        font=dict(size=15, color='#c8d8f0')
    ),
    hovermode='x unified',
)
    fig_ts.update_layout(
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Residual Plot
    with col_opt1:
        st.caption("💡 Dashed = Predicted · Solid = Actual · Band = min/max range")

    st.markdown("**Forecast Residuals (Error = Actual − Predicted)**")
    ts_res = ts_agg.copy()
    ts_res['residual'] = ts_res['actual_mean'] - ts_res['pred_mean']
    fig_res = go.Figure()
    fig_res.add_hline(y=0, line_dash='dash', line_color='#3a4a6a')
    fig_res.add_trace(go.Bar(
        x=ts_res['timestamp'], y=ts_res['residual'],
        marker_color=[COLORS['primary'] if v >= 0 else COLORS['danger'] for v in ts_res['residual']],
        name='Residual'
    ))
    fig_res.update_layout(**PLOTLY_THEME, height=220,
                          title=dict(text="Residuals Over Time", font=dict(size=13, color='#c8d8f0')))
    st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════
# TAB 2 — REGIONAL ANALYSIS
# ══════════════════════════════════════════
with tab2:
    if 'region' not in df.columns:
        st.warning("No 'region' column found.")
    else:
        reg_df = df.groupby('region').agg(
            actual_mean   = ('usage_units',     'mean'),
            predicted_mean= ('prediction', 'mean'),
            predicted_sum = ('prediction', 'sum'),
            count         = ('usage_units',     'count'),
        ).reset_index()

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(
                x=reg_df['region'], y=reg_df['actual_mean'],
                name='Avg Actual', marker_color=COLORS['primary'],
                marker_line=dict(color='#1a3a6a', width=1),
            ))
            fig_reg.add_trace(go.Bar(
                x=reg_df['region'], y=reg_df['predicted_mean'],
                name='Avg Predicted', marker_color=COLORS['forecast'],
                marker_line=dict(color='#5a2a1a', width=1),
            ))
            fig_reg.update_layout(
                **PLOTLY_THEME, barmode='group', height=360,
                title=dict(text="Avg Actual vs Predicted by Region", font=dict(size=13, color='#c8d8f0'))
            )
            st.plotly_chart(fig_reg, use_container_width=True)

        with col_r2:
            fig_pie = go.Figure(go.Pie(
                labels=reg_df['region'],
                values=reg_df['predicted_sum'],
                hole=0.55,
                marker=dict(colors=['#0078d4','#7ecfff','#3fb950','#a371f7','#f78166'],
                            line=dict(color='#0a0e1a', width=2)),
                textfont=dict(color='#c8d8f0'),
            ))
            fig_pie.update_layout(
                **PLOTLY_THEME, height=360,
                title=dict(text="Total Forecast Share by Region", font=dict(size=13, color='#c8d8f0')),
                annotations=[dict(text='Forecast<br>Share', x=0.5, y=0.5,
                                  font=dict(size=13, color='#7ecfff'), showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Regional utilisation heatmap
        if 'provisioned_capacity' in df.columns and 'timestamp' in df.columns:
            st.markdown("**Capacity Utilisation Heatmap by Region**")
            df['month'] = df['timestamp'].dt.strftime('%b %Y')
            heat_df = df.groupby(['region','month']).apply(
                lambda x: (x['prediction'].sum() / x['provisioned_capacity'].replace(0,np.nan).sum() * 100)
            ).reset_index(name='utilisation_pct')
            months_order = df['month'].unique().tolist()
            heat_pivot = heat_df.pivot(index='region', columns='month', values='utilisation_pct')
            heat_pivot = heat_pivot.reindex(columns=[m for m in months_order if m in heat_pivot.columns])

            fig_heat = go.Figure(go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=[[0,'#0a2040'],[0.5,'#0078d4'],[1,'#f85149']],
                text=[[f"{v:.0f}%" if not np.isnan(v) else '' for v in row] for row in heat_pivot.values],
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(tickfont=dict(color='#c8d8f0'), len=0.8)
            ))
            fig_heat.update_layout(
    **PLOTLY_THEME,
    height=280
)

            fig_heat.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3 — SERVICE TYPE
# ══════════════════════════════════════════
with tab3:
    if 'service_type' not in df.columns:
        st.warning("No 'service_type' column found.")
    else:
        svc_df = df.groupby('service_type').agg(
            actual_total   = ('usage_units',     'sum'),
            predicted_total= ('prediction', 'sum'),
            avg_cost       = ('cost_usd',        'mean'),
        ).reset_index()

        col_s1, col_s2 = st.columns(2)
        svc_colors = ['#0078d4','#3fb950','#a371f7','#f78166','#ffa657']

        with col_s1:
            fig_svc = go.Figure()
            for i, row in svc_df.iterrows():
                fig_svc.add_trace(go.Bar(
                    name=row['service_type'],
                    x=['Actual', 'Predicted'],
                    y=[row['actual_total'], row['predicted_total']],
                    marker_color=svc_colors[i % len(svc_colors)]
                ))
            fig_svc.update_layout(
                **PLOTLY_THEME, barmode='group', height=380,
                title=dict(text="Total Demand by Service Type", font=dict(size=13, color='#c8d8f0'))
            )
            st.plotly_chart(fig_svc, use_container_width=True)

        with col_s2:
            # Stacked area over time
            if 'timestamp' in df.columns:
                svc_ts = df.groupby(['timestamp','service_type'])['prediction'].mean().reset_index()
                svc_ts = svc_ts.sort_values('timestamp')
                fig_area = go.Figure()
                for i, svc in enumerate(svc_df['service_type']):
                    sub = svc_ts[svc_ts['service_type'] == svc]
                    fig_area.add_trace(go.Scatter(
                    x=sub['timestamp'],
                    y=sub['prediction'],
                    name=svc,
                    stackgroup='one',
                    fillcolor=hex_to_rgba(svc_colors[i % len(svc_colors)], 0.7),
                    line=dict(width=0.5, color=svc_colors[i % len(svc_colors)]),
                    mode='lines'
)) 
                fig_area.update_layout(
                    **PLOTLY_THEME, height=380,
                    title=dict(text="Stacked Predicted Demand by Service", font=dict(size=13, color='#c8d8f0'))
                )
                st.plotly_chart(fig_area, use_container_width=True)

        # Cost analysis
        if 'cost_usd' in df.columns:
            st.markdown("**Average Cost (USD) by Service Type**")
            fig_cost = go.Figure(go.Bar(
                x=svc_df['service_type'], y=svc_df['avg_cost'].round(2),
                marker_color=svc_colors[:len(svc_df)],
                text=svc_df['avg_cost'].round(2),
                texttemplate='$%{text}',
                textposition='outside',
                textfont=dict(color='#c8d8f0')
            ))
            fig_cost.update_layout(**PLOTLY_THEME, height=360,
                                   title=dict(text="Avg Cost per Record by Service Type",
                                              font=dict(size=13, color='#c8d8f0')))
            st.plotly_chart(fig_cost, use_container_width=True)


# ══════════════════════════════════════════
# TAB 4 — MONITORING & DRIFT
# ══════════════════════════════════════════
with tab4:
    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("**Model Health Summary**")
        rmse_thresh = rmse_baseline * (1 + alert_margin / 100)
        status_color = COLORS['danger'] if drift_detected else COLORS['teal']

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['rmse'],
            delta={'reference': rmse_baseline, 'valueformat': '.1f',
                   'increasing': {'color': COLORS['danger']},
                   'decreasing': {'color': COLORS['teal']}},
            gauge={
                'axis': {'range': [0, rmse_thresh * 1.5], 'tickcolor': '#6b7a99',
                         'tickfont': {'color': '#6b7a99'}},
                'bar': {'color': status_color},
                'bgcolor': '#111827',
                'steps': [
                    {'range': [0, rmse_baseline], 'color': 'rgba(63,185,80,0.1)'},
                    {'range': [rmse_baseline, rmse_thresh], 'color': 'rgba(248,161,0,0.1)'},
                    {'range': [rmse_thresh, rmse_thresh * 1.5], 'color': 'rgba(248,81,73,0.1)'},
                ],
                'threshold': {'line': {'color': COLORS['warning'], 'width': 2},
                              'thickness': 0.8, 'value': rmse_thresh}
            },
            number={'font': {'color': status_color, 'size': 36}, 'suffix': ' RMSE'},
            title={'text': "Live Model RMSE vs Baseline", 'font': {'color': '#c8d8f0', 'size': 13}}
        ))
        fig_gauge.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_m2:
        st.markdown("**Metrics Breakdown**")
        metric_rows = [
            ("RMSE",                f"{metrics['rmse']:.2f}",    "Lower is better", "🎯"),
            ("MAE",                 f"{metrics['mae']:.2f}",     "Mean Abs Error",  "📐"),
            ("Forecast Bias",       f"{metrics['bias']:+.2f}",   "+ve = over-forecast, -ve = under", "⚖️"),
            ("Directional Accuracy",f"{metrics['directional_accuracy']:.1f}%", "Correct trend %", "🧭"),
            ("Capacity Utilisation",f"{metrics['utilisation']:.1f}%", "Predicted / Provisioned", "📊"),
        ]
        for name, val, desc, icon in metric_rows:
            c_i, c_n, c_v, c_d = st.columns([0.3, 2, 1.2, 2.5])
            c_i.markdown(f"<span style='font-size:18px'>{icon}</span>", unsafe_allow_html=True)
            c_n.markdown(f"<span style='color:#c8d8f0; font-size:13px; font-weight:600'>{name}</span>",
                         unsafe_allow_html=True)
            c_v.markdown(f"<span style='color:#7ecfff; font-family:IBM Plex Mono; font-size:14px'>{val}</span>",
                         unsafe_allow_html=True)
            c_d.markdown(f"<span style='color:#4a5a7a; font-size:11px'>{desc}</span>", unsafe_allow_html=True)

    # RMSE log if exists
    st.markdown("---")
    if os.path.exists("rmse_history.csv"):
        st.markdown("**📋 RMSE History Log**")
        rmse_hist = pd.read_csv("rmse_history.csv")
        rmse_hist['timestamp'] = pd.to_datetime(rmse_hist['timestamp'])
        fig_rh = go.Figure()
        fig_rh.add_hline(y=rmse_baseline, line_dash='dash', line_color=COLORS['warning'],
                         annotation_text=f"Baseline ({rmse_baseline})")
        fig_rh.add_hline(y=rmse_thresh, line_dash='dot', line_color=COLORS['danger'],
                         annotation_text=f"Alert Threshold ({rmse_thresh:.1f})")
        fig_rh.add_trace(go.Scatter(
            x=rmse_hist['timestamp'], y=rmse_hist['rmse'],
            mode='lines+markers', name='RMSE',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(color=[COLORS['danger'] if v > rmse_thresh else COLORS['primary']
                               for v in rmse_hist['rmse']], size=8)
        ))
        fig_rh.update_layout(**PLOTLY_THEME, height=380,
                             title=dict(text="RMSE Over Time (from monitoring_log)", font=dict(size=13)))
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("💡 Run `python monitoring.py` to generate RMSE history. It will appear here automatically.")

    # Model registry
    if os.path.exists("model_registry.json"):
        st.markdown("**📁 Model Registry (Retraining History)**")
        with open("model_registry.json") as f:
            registry = json.load(f)
        reg_df = pd.DataFrame(registry)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
    else:
        st.info("💡 Run `python monitoring.py --retrain` to populate the model registry.")


# ══════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════
with tab5:
    col_e1, col_e2 = st.columns([2, 1])

    with col_e1:
        st.markdown(f"**Filtered Dataset — {len(df):,} records**")
        display_cols = [c for c in ['timestamp','region','service_type','usage_units',
                                     'prediction','provisioned_capacity','cost_usd',
                                     'availability_pct'] if c in df.columns]
        st.dataframe(
            df[display_cols].sort_values('timestamp', ascending=False).head(500).reset_index(drop=True),
            use_container_width=True, height=380, hide_index=True
        )

    with col_e2:
        st.markdown("**Summary Statistics**")
        stats_cols = ['usage_units','prediction']
        if 'provisioned_capacity' in df.columns:
            stats_cols.append('provisioned_capacity')
        st.dataframe(
            df[stats_cols].describe().round(2),
            use_container_width=True, height=380
        )

    # Download
    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            data=df.to_csv(index=False),
            file_name=f"filtered_forecast_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col2:
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "records": len(df),
            "metrics": metrics,
            "drift_detected": drift_detected,
            "capacity_breaches": n_breaches,
        }
        st.download_button(
            "⬇️ Download Metrics Report (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"metrics_report_{datetime.utcnow().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )


