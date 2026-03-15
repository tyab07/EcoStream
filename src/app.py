import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))
from analysis.task1_pca import load_pca_data, run_pca, compute_centroids, FEATURE_COLS
from analysis.task2_temporal import get_temporal_data, build_monthly_pivot
from analysis.task3_distribution import (
    get_distribution_data, compute_statistics,
    build_histogram_fig, build_survival_fig, THRESHOLD
)
from analysis.task4_visual_integrity import (
    load_station_pm25, build_audit_df
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EcoStream | Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Glassmorphic Premium CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --glass-bg: rgba(17, 24, 39, 0.6);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.main {
    background: linear-gradient(160deg, #0a0e17 0%, #0f172a 30%, #111827 60%, #0a0e17 100%);
    color: var(--text-primary);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--bg-secondary); }

/* ── Glass Card ── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(17, 24, 39, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.metric-card .value sup {
    font-size: 0.6em;
    color: var(--text-secondary);
}

/* ── Section Title ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 3px solid var(--accent-blue);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(17, 24, 39, 0.4);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--glass-border);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.15) !important;
    color: var(--accent-blue) !important;
    border-bottom: none !important;
}

/* ── Headings ── */
h1, h2, h3, h4 { color: var(--text-primary); font-weight: 600; }

/* ── Dashboard Header ── */
.dash-header {
    text-align: center;
    padding: 32px 0 24px;
}
.dash-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.dash-header p {
    color: var(--text-muted);
    font-size: 0.9rem;
}

/* ── Plotly Overrides ── */
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader { color: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading (delegated to task modules) ─────────────────
@st.cache_data
def cached_pca_data():
    df = load_pca_data()
    df_pca, pca_obj, loadings = run_pca(df.copy())
    centroids = compute_centroids(df_pca)
    return df_pca, pca_obj, loadings, centroids

# ── Plotly dark template ─────────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    margin=dict(l=20, r=20, t=50, b=20),
)

# ── Main ─────────────────────────────────────────────────────
def main():
    st.markdown("""
        <div class="dash-header">
            <h1>🌍 EcoStream</h1>
            <p>Air Quality Intelligence Dashboard — Tufte-Certified Visualizations</p>
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Dimensionality Reduction",
        "📊 Temporal Analysis",
        "📈 Distribution Modeling",
        "🛡️ Visual Integrity Audit"
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — PCA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        st.markdown('<div class="section-title">Principal Component Analysis (PCA)</div>', unsafe_allow_html=True)

        with st.spinner("Loading pollutant data..."):
            df_pca, pca_obj, loadings, centroids = cached_pca_data()

        var_pc1 = pca_obj.explained_variance_ratio_[0]
        var_pc2 = pca_obj.explained_variance_ratio_[1]
        total_var = var_pc1 + var_pc2

        centroid_gap = centroids['centroid_gap']

        # Metrics
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="label">Total Records</div><div class="value">{len(df_pca):,}</div></div>
            <div class="metric-card"><div class="label">Variance Captured</div><div class="value">{total_var:.1%}</div></div>
            <div class="metric-card"><div class="label">PC1 Variance</div><div class="value">{var_pc1:.1%}</div></div>
            <div class="metric-card"><div class="label">PC2 Variance</div><div class="value">{var_pc2:.1%}</div></div>
            <div class="metric-card"><div class="label">Centroid Gap</div><div class="value">{centroid_gap:.2f}</div></div>
        </div>
        """, unsafe_allow_html=True)

        plot_col, info_col = st.columns([3, 1])

        with plot_col:
            color_map = {'Residential': '#3b82f6', 'Industrial': '#ef4444'}
            fig = px.scatter(
                df_pca, x='PC1', y='PC2', color='Zone',
                color_discrete_map=color_map,
                hover_data=['location_name'],
                opacity=0.45,
                labels={
                    "PC1": f"PC1 ({var_pc1:.1%} variance)",
                    "PC2": f"PC2 ({var_pc2:.1%} variance)"
                }
            )
            # Centroids
            fig.add_scatter(
                x=[centroids['res_pc1']], y=[centroids['res_pc2']],
                mode='markers+text', text=["Residential"],
                textposition="top center",
                marker=dict(symbol='diamond', size=14, color='#3b82f6',
                            line=dict(width=2, color='white')),
                name='Residential Centroid',
                textfont=dict(color='white', size=11)
            )
            fig.add_scatter(
                x=[centroids['ind_pc1']], y=[centroids['ind_pc2']],
                mode='markers+text', text=["Industrial"],
                textposition="top center",
                marker=dict(symbol='diamond', size=14, color='#ef4444',
                            line=dict(width=2, color='white')),
                name='Industrial Centroid',
                textfont=dict(color='white', size=11)
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="PCA Projection — Industrial vs Residential Pollution Signatures",
                xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#1e293b'),
                yaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#1e293b'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center',
                            font=dict(color='#94a3b8')),
                height=520,
            )
            st.plotly_chart(fig, use_container_width=True)

        with info_col:
            st.markdown('<div class="section-title">Feature Loadings</div>', unsafe_allow_html=True)
            styled = loadings.style.background_gradient(cmap='Blues', axis=None).format("{:.3f}")
            st.dataframe(styled, use_container_width=True, height=280)
            st.markdown("""
            <div class="glass-card" style="font-size:0.82rem; color:#94a3b8;">
                <b>Interpretation:</b> Loadings show how much each pollutant
                contributes to each principal component. Higher absolute values
                indicate stronger influence on that axis.
            </div>
            """, unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — Temporal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown('<div class="section-title">High-Density Temporal Analysis</div>', unsafe_allow_html=True)
        df_temporal = get_temporal_data()

        if df_temporal is not None and not df_temporal.empty:
            df_temporal['label'] = df_temporal['location_name'].apply(lambda x: x.split(',')[0])
            MONTH_MAP = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                         7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

            mv = df_temporal.groupby(['label', 'month'])['violation'].mean().reset_index()
            mv['month_name'] = mv['month'].map(MONTH_MAP)
            pivot = mv.pivot_table(index='label', columns='month_name', values='violation')
            ordered = [m for m in MONTH_MAP.values() if m in pivot.columns]
            pivot = pivot[ordered]
            pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

            fig_heat = px.imshow(
                pivot,
                color_continuous_scale='YlOrRd',
                labels=dict(x="Month", y="Monitoring Station", color="Violation Rate"),
                text_auto=".0%",
                aspect="auto"
            )
            fig_heat.update_layout(
                **PLOTLY_LAYOUT,
                xaxis=dict(showgrid=False, side='top'),
                yaxis=dict(showgrid=False, title_text="Monitoring Station"),
                height=max(400, len(pivot) * 28),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("No temporal data available.")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — Distribution
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.markdown('<div class="section-title">Distribution Modeling & Tail Integrity</div>', unsafe_allow_html=True)
        station_t3, pm25_arr = get_distribution_data()
        stats = compute_statistics(pm25_arr)

        # Metrics
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="label">Target Station</div><div class="value" style="font-size:1rem;">{station_t3[:28]}</div></div>
            <div class="metric-card"><div class="label">Valid Observations</div><div class="value">{stats['n']:,}</div></div>
            <div class="metric-card"><div class="label">99th Percentile</div><div class="value">{stats['p99']:.1f}<sup> μg/m³</sup></div></div>
            <div class="metric-card"><div class="label">Extreme Events (>200)</div><div class="value">{stats['count_ext']} <sup>({stats['pct_ext']:.3f}%)</sup></div></div>
            <div class="metric-card"><div class="label">Skewness</div><div class="value">{stats['skewness']:.3f}</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Plot A — Standard Histogram**")
            fig_h = build_histogram_fig(pm25_arr, station_t3)
            fig_h.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_h, use_container_width=True)
        with col2:
            st.markdown("**Plot B — Log-Scaled Survival CDF**")
            fig_s = build_survival_fig(pm25_arr, station_t3, stats['p99'])
            fig_s.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_s, use_container_width=True)

        with st.expander("📝 Analytical Explanation — Tail Integrity & Extreme Risk"):
            st.markdown("""
**1. Why histograms hide tail behaviour**
Standard histograms use equal-width bins on a linear y-axis. The bulk of data clusters
near low concentrations, making extreme events (>200 μg/m³) visually invisible.

**2. Why the Survival Function is superior**
S(x) = P(X > x) directly quantifies exceedance probability — the question regulators
actually care about.

**3. Why log-scaling restores Tail Integrity**
On a linear axis, P=0.0001 and P=0.001 look identical. Log-scaling spreads them across
a full decade of visual space, ensuring all risk levels are fairly represented.
            """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4 — Visual Integrity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.markdown('<div class="section-title">Visual Integrity Audit</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="border-left: 3px solid #ef4444;">
            <b style="color:#ef4444;">REJECTED:</b>
            <span style="color:#94a3b8;">3D Bar Charts violate visual honesty — perspective distortion, occlusion,
            and low data-ink ratio make them unsuitable for scientific communication.</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Plot 1 — Bivariate Bubble Map**")
        st.caption("Color = PM2.5 intensity (Plasma); Bubble area = Population Density")

        station_means = load_station_pm25()
        df_t4 = build_audit_df(station_means)

        fig_b = px.scatter(
            df_t4, x='pop_density', y='pm25_mean',
            size='pop_density', color='pm25_mean',
            color_continuous_scale='Plasma', size_max=55,
            text='short_name',
            labels={
                "pop_density": "Population Density (persons/km²)",
                "pm25_mean": "Mean PM2.5 (μg/m³)"
            },
            title="PM2.5 vs Population Density — Bivariate Bubble Map"
        )
        fig_b.update_traces(
            textposition='top center',
            textfont=dict(size=9, color='#e2e8f0')
        )
        fig_b.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(showgrid=False, tickformat=",.0s"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            height=550,
        )
        st.plotly_chart(fig_b, use_container_width=True)

        with st.expander("Why 2D Bubble Maps win over 3D Bar Charts"):
            st.markdown("""
- **No Occlusion**: Every data point is visible; bars in 3D hide each other.
- **Linear Scaling**: Bubble area scales honestly with data values.
- **High Data-Ink**: Color _and_ size each encode a variable — no wasted pixels.
- **Perceptual Uniformity**: Plasma colormap ensures magnitude honesty.
            """)

if __name__ == "__main__":
    main()
