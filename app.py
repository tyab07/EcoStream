import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Air Quality PCA Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_zone(loc_name):
    loc_lower = loc_name.lower()
    if "manali" in loc_lower or "talkatora" in loc_lower or "solapur" in loc_lower or "industrial" in loc_lower:
        return "Industrial"
    return "Residential"

@st.cache_data
def load_and_preprocess_data():
    # ── Use a memory-capped connection with disk-spilling to avoid OutOfMemoryException
    #    on the ~1.7 GB CSV dataset.
    con = duckdb.connect()
    
    # Ensure temporary directory exists for out-of-core processing
    import os
    if not os.path.exists('duckdb_temp'):
        os.makedirs('duckdb_temp')
    
    con.execute("PRAGMA temp_directory='duckdb_temp';")
    con.execute("SET memory_limit='1GB';")
    con.execute("SET threads TO 2;")

    # ── Step 1: find stations that have all 6 required parameters ─────────
    # Use fast OS-level directory scan to avoid DuckDB OUT OF MEMORY
    # when doing read_csv_auto across millions of raw rows
    import os
    valid_stations = []
    raw_dir = os.path.join("data", "raw")
    if os.path.exists(raw_dir):
        for folder in os.listdir(raw_dir):
            folder_path = os.path.join(raw_dir, folder)
            if not os.path.isdir(folder_path): continue
            
            p_found = set()
            for fn in os.listdir(folder_path):
                if not fn.endswith('.csv'): continue
                lname = fn.lower()
                if 'pm2.5' in lname or 'pm25' in lname: p_found.add('pm25')
                elif 'pm10' in lname: p_found.add('pm10')
                elif 'no2' in lname: p_found.add('no2')
                elif 'o3' in lname or 'ozone' in lname: p_found.add('ozone')
                elif 'temperature' in lname: p_found.add('temperature')
                elif 'humidity' in lname or 'relativehumidity' in lname: p_found.add('humidity')
            
            if len(p_found) >= 6:
                base_name = folder.rsplit('_', 1)[0]
                valid_stations.append(base_name)

    if not valid_stations:
        con.close()
        return pd.DataFrame(), []

    # ── Step 2: register station list so DuckDB can JOIN safely ───────────
    con.execute("CREATE TEMP TABLE valid_st (location_name VARCHAR)")
    con.executemany("INSERT INTO valid_st VALUES (?)", [[s] for s in valid_stations])

    # ── Step 3: do ALL processing inside a single SQL query ───────────────
    #   • filter to valid stations via INNER JOIN (no full table scan to pandas)
    #   • normalise parameter names
    #   • parse timestamps (handles plain ISO and JSON-dict formats)
    #   • skip Feb-29 leap-day rows
    #   • floor to hourly bucket; remap year to 2025
    #   • aggregate to hourly means per station+parameter
    #   • pivot wide (CASE WHEN … END) — no pandas pivot_table
    #   • drop rows where any required parameter is NULL
    # Only the final small model DataFrame comes back to Python → no OOM.
    df_model = con.execute(r"""
        WITH src AS (
            SELECT
                r.location_name,
                CASE
                    WHEN lower(r.parameter) IN ('pm2.5','pm25')                THEN 'pm25'
                    WHEN lower(r.parameter) = 'pm10'                           THEN 'pm10'
                    WHEN lower(r.parameter) = 'no2'                            THEN 'no2'
                    WHEN lower(r.parameter) IN ('o3','ozone')                  THEN 'ozone'
                    WHEN lower(r.parameter) = 'temperature'                    THEN 'temperature'
                    WHEN lower(r.parameter) IN ('humidity','relativehumidity') THEN 'humidity'
                    ELSE NULL
                END AS param_norm,
                date_trunc('hour',
                    TRY_CAST(
                        CASE
                            WHEN r.timestamp LIKE '{%'
                            THEN regexp_extract(r.timestamp,
                                    '''utc'':\s*''([^'']+)''', 1)
                            ELSE r.timestamp
                        END
                    AS TIMESTAMP)
                ) AS ts,
                r.value
            FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=true) r
            INNER JOIN valid_st v ON r.location_name = v.location_name
            WHERE r.value IS NOT NULL
        ),
        filtered AS (
            SELECT *
            FROM src
            WHERE param_norm IS NOT NULL
              AND ts IS NOT NULL
              AND NOT (month(ts) = 2 AND day(ts) = 29)
        ),
        hourly AS (
            SELECT
                location_name,
                param_norm,
                make_timestamp(2025, month(ts), day(ts), hour(ts), 0, 0) AS ts_2025,
                avg(value) AS val
            FROM filtered
            GROUP BY location_name, param_norm, month(ts), day(ts), hour(ts)
        ),
        pivoted AS (
            SELECT
                location_name,
                ts_2025,
                avg(CASE WHEN param_norm = 'pm25'        THEN val END) AS pm25,
                avg(CASE WHEN param_norm = 'pm10'        THEN val END) AS pm10,
                avg(CASE WHEN param_norm = 'no2'         THEN val END) AS no2,
                avg(CASE WHEN param_norm = 'ozone'       THEN val END) AS ozone,
                avg(CASE WHEN param_norm = 'temperature' THEN val END) AS temperature,
                avg(CASE WHEN param_norm = 'humidity'    THEN val END) AS humidity
            FROM hourly
            GROUP BY location_name, ts_2025
        )
        SELECT *
        FROM pivoted
        WHERE pm25        IS NOT NULL
          AND pm10        IS NOT NULL
          AND no2         IS NOT NULL
          AND ozone       IS NOT NULL
          AND temperature IS NOT NULL
          AND humidity    IS NOT NULL
        ORDER BY location_name, ts_2025
    """).df()

    con.close()

    required_cols = ['pm25', 'pm10', 'no2', 'ozone', 'temperature', 'humidity']
    df_model['Zone'] = df_model['location_name'].apply(get_zone)
    return df_model, required_cols


def compute_pca(df, cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df['PC1'] = pcs[:, 0]
    df['PC2'] = pcs[:, 1]
    
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=cols)
    return df, pca, loadings

def main():
    st.title("🌍 Smart City Environmental Intelligence")
    
    with st.spinner("Loading and Preprocessing Data..."):
        df_model, required_cols = load_and_preprocess_data()
        
    # --- Tab Layout ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dim Reduction (PCA)",
        "High-Density Temporal Analysis",
        "Distribution Modeling (Task 3)",
        "Visual Integrity Audit (Task 4)"
    ])
    
    # ==========================================
    # TAB 1: PCA Analysis
    # ==========================================
    with tab1:
        st.markdown("### Dimensionality Reduction Analysis")
        df_pca, pca, loadings = compute_pca(df_model.copy(), required_cols)
        
        var_pc1 = pca.explained_variance_ratio_[0]
        var_pc2 = pca.explained_variance_ratio_[1]
        
        # Calculate Centroids
        res_pc1_mean = df_pca[df_pca['Zone'] == 'Residential']['PC1'].mean()
        res_pc2_mean = df_pca[df_pca['Zone'] == 'Residential']['PC2'].mean()
        ind_pc1_mean = df_pca[df_pca['Zone'] == 'Industrial']['PC1'].mean()
        ind_pc2_mean = df_pca[df_pca['Zone'] == 'Industrial']['PC2'].mean()
        distance = np.sqrt((ind_pc1_mean - res_pc1_mean)**2 + (ind_pc2_mean - res_pc2_mean)**2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_pca):,}")
        col2.metric("Total Variance Explained", f"{(var_pc1 + var_pc2):.2%}")
        col3.metric("Centroid Distance", f"{distance:.2f}")
        col4.metric("PC1 Mean Diff (Ind-Res)", f"{ind_pc1_mean - res_pc1_mean:.2f}")

        st.divider()

        plot_col, stats_col = st.columns([2, 1])
        
        with plot_col:
            st.subheader("Interactive PCA Projection")
            color_map = {'Residential': '#3498db', 'Industrial': '#e74c3c'}
            fig = px.scatter(
                df_pca, x='PC1', y='PC2', color='Zone', color_discrete_map=color_map,
                hover_data=['location_name', 'ts_2025'], opacity=0.5,
                title="Air Quality Profiles (Dim. Reduction)",
                labels={"PC1": f"Principal Component 1 ({var_pc1:.1%} var)", "PC2": f"Principal Component 2 ({var_pc2:.1%} var)"}
            )
            # Add Centroids
            fig.add_scatter(x=[res_pc1_mean], y=[res_pc2_mean], mode='markers', marker=dict(symbol='x', size=15, color='white', line=dict(width=2, color='black')), name='Residential Centroid')
            fig.add_scatter(x=[ind_pc1_mean], y=[ind_pc2_mean], mode='markers', marker=dict(symbol='x', size=15, color='white', line=dict(width=2, color='black')), name='Industrial Centroid')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=1, zerolinecolor='#555'), yaxis=dict(showgrid=False, zeroline=True, zerolinewidth=1, zerolinecolor='#555'))
            st.plotly_chart(fig, use_container_width=True)

        with stats_col:
            st.subheader("Component Loadings")
            st.dataframe(loadings.style.background_gradient(cmap='Blues'), use_container_width=True)
            st.subheader("Zone Centroids")
            st.markdown(f"- **Residential:** ({res_pc1_mean:.2f}, {res_pc2_mean:.2f})\n- **Industrial:** ({ind_pc1_mean:.2f}, {ind_pc2_mean:.2f})")
            
        with st.expander("📝 Analytical Details"):
            st.write("Provides empirical evidence of structural differentiation in pollution profiles. Industrial zones lean strongly toward extreme PC1 values.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: High-Density Temporal Analysis
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("## 🕐 Task 2 — High-Density Temporal Analysis")
        st.caption(
            "Tracking PM2.5 Health Threshold Violations (> 35 μg/m³) across all stations "
            "to identify neighborhoods that consistently exceed safe limits and reveal "
            "the **periodic signature** of pollution events."
        )

        from temporal_analysis import get_temporal_data

        with st.spinner("⏳ Loading PM2.5 temporal data from all stations…"):
            df_temporal = get_temporal_data()

        # ── Guard: nothing to show ─────────────────────────────────────────
        if df_temporal is None or df_temporal.empty:
            st.error(
                "⚠️ No PM2.5 data could be loaded. "
                "Check that `data/raw/*/*.csv` files exist and contain pm2.5 readings."
            )
            st.stop()

        # ── Ensure correct dtypes ──────────────────────────────────────────
        df_temporal['month']     = df_temporal['month'].astype(int)
        df_temporal['hour']      = df_temporal['hour'].astype(int)
        df_temporal['violation'] = df_temporal['violation'].astype(float)

        # ── Better short label: use city portion after last comma ──────────
        # e.g. "Talkatora District Industries Center, Lucknow - CPCB" → "Lucknow"
        # Fall back to first word of full name if no comma present
        def _short(name: str) -> str:
            if ',' in name:
                city = name.split(',')[-1].strip()          # "Lucknow - CPCB"
                city = city.split('-')[0].strip()           # "Lucknow"
                return city if city else name.split()[0]
            return name.split()[0]

        df_temporal['label'] = df_temporal['location_name'].apply(_short)

        MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        # ══════════════════════════════════════════════════════════════
        # CHART 1 — Station × Month Heatmap
        # (High-density: all stations on y, all months on x, colour = violation rate)
        # ══════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("📊 Chart 1 — High-Density Heatmap: Stations × Month")
        st.caption(
            "Each row = one monitoring station. Each column = one month. "
            "Colour intensity = fraction of hours that exceeded 35 μg/m³. "
            "**Dark red vertical bands** = city-wide pollution events. "
            "**Dark red horizontal streaks** = persistently polluted stations."
        )

        monthly_viol = (
            df_temporal
            .groupby(['label', 'month'])['violation']
            .mean()
            .reset_index()
        )
        monthly_viol['month_name'] = monthly_viol['month'].map(MONTH_NAMES)

        # pivot: rows = station labels, cols = month names
        pivot_m = monthly_viol.pivot_table(
            index='label', columns='month_name', values='violation', aggfunc='mean'
        )
        ordered_months = [m for m in MONTH_NAMES.values() if m in pivot_m.columns]
        pivot_m = pivot_m[ordered_months]

        # sort rows by mean violation rate (worst at top)
        row_means = monthly_viol.groupby('label')['violation'].mean().sort_values(ascending=False)
        pivot_m   = pivot_m.reindex(row_means.index).dropna(how='all')

        n_rows = len(pivot_m)
        hmap_h = max(420, n_rows * 28 + 120)

        HEAT_SCALE = [
            [0.00, "#f7f7f7"],   # near-zero  → light grey-white
            [0.20, "#fce8c3"],   # 20%        → pale amber
            [0.40, "#f6a623"],   # 40%        → amber
            [0.65, "#d7191c"],   # 65%        → red
            [1.00, "#67000d"],   # 100%       → dark crimson
        ]

        fig_heat = px.imshow(
            pivot_m,
            labels=dict(x="Month", y="Monitoring Station", color="Violation Rate"),
            x=pivot_m.columns.tolist(),
            y=pivot_m.index.tolist(),
            color_continuous_scale=HEAT_SCALE,
            zmin=0, zmax=1,
            aspect="auto",
            text_auto=".0%",
            title=(
                "PM2.5 Health Violation Rate — Stations × Month  "
                f"(n = {n_rows} stations, threshold > 35 μg/m³)"
            ),
        )
        fig_heat.update_traces(
            textfont=dict(size=8, color="#222"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Month: <b>%{x}</b><br>"
                "Violation Rate: <b>%{z:.1%}</b>"
                "<extra></extra>"
            ),
        )
        fig_heat.update_coloraxes(
            colorbar=dict(
                title=dict(text="Violation<br>Rate", font=dict(size=11)),
                tickformat=".0%",
                thickness=14,
                len=0.80,
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0%","25%","50%","75%","100%"],
            )
        )
        fig_heat.update_layout(
            height=hmap_h,
            margin=dict(l=10, r=130, t=60, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                side="bottom",
                title=dict(text="Month of Year", font=dict(size=13)),
                tickfont=dict(size=12, color="#222"),
                showgrid=False,
            ),
            yaxis=dict(
                title=dict(text="Monitoring Station", font=dict(size=12)),
                tickfont=dict(size=10, color="#222"),
                showgrid=False,
                automargin=True,
            ),
            font=dict(family="Inter, Arial, sans-serif"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ══════════════════════════════════════════════════════════════
        # CHART 2 & 3 — Periodicity: Diurnal + Seasonal
        # ══════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("📈 Chart 2 & 3 — Periodic Signature of Pollution Events")
        st.caption(
            "Determines whether PM2.5 violations are driven by **daily 24-hour traffic cycles** "
            "or **monthly seasonal weather patterns**."
        )

        col_diurnal, col_seasonal = st.columns(2)

        # ── Diurnal (24-hour) ─────────────────────────────────────────
        with col_diurnal:
            hourly_stats = (
                df_temporal.groupby('hour')['violation']
                .mean()
                .reset_index()
                .rename(columns={'violation': 'rate'})
            )
            # highlight the peak hour
            peak_h = int(hourly_stats.loc[hourly_stats['rate'].idxmax(), 'hour'])

            fig_hr = px.bar(
                hourly_stats, x='hour', y='rate',
                color='rate',
                color_continuous_scale=[
                    [0.0, "#f7fbff"], [0.4, "#6baed6"],
                    [0.7, "#2171b5"], [1.0, "#08306b"]
                ],
                labels={"hour": "Hour of Day (0–23)", "rate": "Avg Violation Rate"},
                title="⏰ Diurnal Cycle — 24-Hour Pattern",
            )
            fig_hr.add_vline(
                x=peak_h, line_dash="dash", line_color="#d7191c", line_width=2,
                annotation_text=f"Peak: {peak_h:02d}:00",
                annotation_font_color="#d7191c", annotation_position="top right"
            )
            fig_hr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis=dict(tickmode='linear', tick0=0, dtick=2, showgrid=False,
                           tickfont=dict(size=10)),
                yaxis=dict(tickformat=".0%", showgrid=True,
                           gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10)),
                margin=dict(l=0, r=10, t=50, b=10),
                height=340,
            )
            st.plotly_chart(fig_hr, use_container_width=True)
            peak_label = f"{peak_h:02d}:00–{(peak_h+1)%24:02d}:00"
            st.info(f"🔺 **Peak violation hour:** {peak_label}", icon="🕐")

        # ── Seasonal (monthly) ────────────────────────────────────────
        with col_seasonal:
            monthly_stats = (
                df_temporal.groupby('month')['violation']
                .mean()
                .reset_index()
                .rename(columns={'violation': 'rate'})
            )
            monthly_stats['month_name'] = monthly_stats['month'].map(MONTH_NAMES)
            peak_mon_row = monthly_stats.loc[monthly_stats['rate'].idxmax()]
            peak_mon_name = peak_mon_row['month_name']

            fig_mon = px.bar(
                monthly_stats, x='month', y='rate',
                color='rate',
                color_continuous_scale=[
                    [0.0, "#fff5f0"], [0.4, "#fc8d59"],
                    [0.7, "#d7301f"], [1.0, "#67000d"]
                ],
                labels={"month": "Month", "rate": "Avg Violation Rate"},
                title="📅 Seasonal Cycle — Monthly Pattern",
                hover_data={"month_name": True, "month": False},
            )
            fig_mon.update_xaxes(
                tickvals=monthly_stats['month'].tolist(),
                ticktext=monthly_stats['month_name'].tolist(),
                showgrid=False, tickfont=dict(size=10)
            )
            fig_mon.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                yaxis=dict(tickformat=".0%", showgrid=True,
                           gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=10)),
                margin=dict(l=0, r=10, t=50, b=10),
                height=340,
            )
            st.plotly_chart(fig_mon, use_container_width=True)
            st.info(f"🔺 **Peak violation month:** {peak_mon_name}", icon="📅")

        # ══════════════════════════════════════════════════════════════
        # Analytical Interpretation (assignment requirement)
        # ══════════════════════════════════════════════════════════════
        st.divider()

        # top 5 most-violated stations
        top5 = row_means.head(5)
        top5_txt = "\n".join(
            f"  {i+1}. **{s}** — {v:.1%} avg violation rate"
            for i,(s,v) in enumerate(top5.items())
        )

        st.markdown("### 📝 Analytical Interpretation")
        st.markdown(f"""
**Why a heatmap instead of a line chart?**
A standard line chart with {n_rows} overlapping lines produces unreadable clutter. 
The Station × Month heatmap encodes all {n_rows} time-series simultaneously in a compact grid —
colour saturation replaces line height, enabling instant identification of which 
stations and which time windows are most hazardous.

**High-violation neighbourhoods (top 5 stations):**
{top5_txt}

**Periodic Signature — What drives the violations?**

* 🕐 **Diurnal (24-hour) pattern**: The diurnal chart (Chart 2) reveals the daily cycle.  
  Peak violations at **{peak_label}** align with morning traffic rush hours combined with  
  nocturnal temperature inversions that trap pollutants near the surface until midday mixing clears them.

* 📅 **Seasonal (30-day) pattern**: The seasonal chart (Chart 3) shows that **{peak_mon_name}**  
  carries the highest violation burden. Winter months exhibit stronger violations due to:  
  (i) suppressed planetary boundary-layer height trapping emissions close to ground level,  
  (ii) increased residential biomass/coal heating, and  
  (iii) weaker wind speeds reducing horizontal dispersion.

**Interpretation for the Mayor:**
Vertical dark-red bands in the heatmap indicate *city-wide synoptic pollution events* 
affecting all stations simultaneously — likely driven by stagnant weather systems.  
Horizontal dark-red streaks identify *persistent local point-source emitters* 
(industrial zones, high-traffic corridors) that require station-specific intervention regardless of season.
""")


    # ══════════════════════════════════════════════════════════════════
    # TAB 3: Distribution Modeling (Task 3)
    # ══════════════════════════════════════════════════════════════════

    with tab3:
        from task3_distribution import (
            get_distribution_data, compute_statistics,
            build_histogram_fig, build_survival_fig, THRESHOLD
        )

        st.markdown("### Distribution Modeling with Log-Scaled Tail Integrity")
        st.caption("Quantifying the probability of Extreme Hazard events where PM2.5 > 200 μg/m³")

        with st.spinner("Loading PM2.5 distribution data..."):
            station_t3, pm25_arr = get_distribution_data()
            stats = compute_statistics(pm25_arr)

        # ── Key metrics row ─────────────────────────────────
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Target Station", station_t3.split(',')[0])
        c2.metric("Valid Observations", f"{stats['n']:,}")
        c3.metric("99th Percentile", f"{stats['p99']:.1f} μg/m³")
        c4.metric("Extreme Events (>200)", f"{stats['count_ext']:,}  ({stats['pct_ext']:.3f}%)")
        c5.metric("Skewness", f"{stats['skewness']:.3f}")
        st.divider()

        # ── Charts side by side ──────────────────────────────
        col_hist, col_surv = st.columns(2)

        with col_hist:
            st.subheader("Plot A — Standard Histogram")
            fig_hist = build_histogram_fig(pm25_arr, station_t3.split(',')[0])
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_surv:
            st.subheader("Plot B — Log-Scaled Survival CDF")
            fig_surv = build_survival_fig(pm25_arr, station_t3.split(',')[0], stats['p99'])
            st.plotly_chart(fig_surv, use_container_width=True)

        # ── Analytical Explanation ───────────────────────────
        st.divider()
        with st.expander("📝 Analytical Explanation — Tail Integrity & Extreme Risk", expanded=True):
            st.markdown("""
**1. Why histograms hide tail behaviour**  
Standard histograms use equal-width bins on a linear y-axis. When the bulk of data
clusters near low concentrations, bins for extreme values (>200 μg/m³) become
visually indistinguishable from zero — rare but health-critical events are literally
invisible.

**2. Why the Survival Function is superior for extreme risk**  
S(x) = P(X > x) = 1 − ECDF(x) directly quantifies the exceedance probability at
every threshold. This matches how regulatory and epidemiological risk models are
actually framed — "how likely is the environment to exceed this level at any hour?"

**3. Why log-scaling improves tail visibility**  
On a linear y-axis, P = 0.0001 and P = 0.001 look identical (both near zero). A
log y-axis stretches them across an entire visual decade, making one order-of-magnitude
differences clearly readable — the core principle of **Tail Integrity**.

**4. 99th percentile in environmental regulation**  
The 99th percentile is used by CPCB / US EPA as the "design value" — the worst-case
concentration that infrastructure must be engineered to handle. A 99th-pct >> WHO
limits signals chronic structural failure, not isolated anomalies.

**5. Heavy right skew and pollution risk**  
Large positive skewness confirms that the mean dramatically under-estimates peak
exposure. Simple mean-based compliance metrics are misleading; tail-sensitive
statistics (CVaR, GEV/GPD distributions) are required.

**6. Tail Integrity**  
A visualization with high tail integrity faithfully represents rare, extreme values
without scale compression or clipping. The log-scaled Survival CDF achieves this by
displaying probabilities across multiple orders of magnitude simultaneously.
            """)

    # ==========================================
    # TAB 4: Visual Integrity Audit (Task 4)
    # ==========================================
    with tab4:
        from task4_visual_integrity import (
            load_station_pm25, build_audit_df,
            PART1_DECISION, PART3_COLORSCALE, CONCLUSION
        )


        st.markdown("### Visual Integrity Audit")
        st.caption("REJECT 3D bar charts. Implement Tufte-compliant 2D alternatives using plasma sequential colormap.")

        with st.spinner("Loading station-level PM2.5 data..."):
            station_means_t4 = load_station_pm25()
            df_t4 = build_audit_df(station_means_t4)

        # ── Part 1: Decision ────────────────────────────────
        with st.expander("Part 1 — Decision: REJECT 3D Bar Chart", expanded=True):
            st.error("**VERDICT: REJECT** the 3D Bar Chart proposal")
            st.markdown("""
| Violation | Detail |
|---|---|
| **Lie Factor >> 1** | Volume scales as h³ — a 2× taller bar looks 8× larger |
| **Low Data-Ink Ratio** | Depth face, side face, perspective lines add zero data |
| **Occlusion** | Front bars physically hide rear bars — data suppression |
| **Perspective distortion** | Rear bars appear smaller even with identical values |
            """)

        st.divider()

        # ── Part 2: Plot 1 — Bubble Bivariate ───────────────
        st.subheader("Plot 1 — Bivariate Bubble Map")
        st.caption("Color = PM2.5 intensity (plasma); Bubble area = Population Density")

        fig_bubble = px.scatter(
            df_t4,
            x='pop_density', y='pm25_mean',
            size='pop_density', color='pm25_mean',
            text='short_name',
            color_continuous_scale='Plasma',
            # No symbol= so all markers are uniform circles
            size_max=55,
            labels={
                'pop_density': 'Population Density (persons/km²)',
                'pm25_mean'  : 'Mean PM2.5 (μg/m³)',
                'zone'       : 'Zone'
            },
            hover_data={'short_name': True, 'zone': True,
                        'pm25_mean': ':.1f', 'pop_density': ':,'},
            title='PM2.5 vs Population Density — Bivariate Bubble Map'
        )
        fig_bubble.update_traces(textposition='top center', textfont_size=9,
                                  marker=dict(symbol='circle', opacity=0.85,
                                              line=dict(width=1, color='rgba(255,255,255,0.4)')))
        fig_bubble.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
            coloraxis_colorbar=dict(title='PM2.5 (μg/m³)', thickness=15, xpad=10),
            margin=dict(l=0, r=100, t=50, b=0),  # right margin for colorbar
            legend=dict(orientation='h', yanchor='bottom', y=-0.15)
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        # ── Part 3: Color Scale Justification ────────────────
        st.divider()
        with st.expander("Part 3 — Color Scale Justification: Plasma vs Rainbow", expanded=True):
            st.markdown("""
| Property | Plasma (Sequential) | Rainbow / Jet |
|---|---|---|
| **Luminance monotonicity** | Strictly increasing (dark → bright) | Multiple peaks & valleys |
| **Perceptual ordering** | Matches biological low→high prior | Requires ROYGBIV memorisation |
| **Magnitude accuracy** | Equal perceptual steps per data step | False categorical boundaries |
| **CVD accessibility** | Safe for all colour vision types | Red-green invisible to 8% of males |
| **Greyscale robustness** | Monotonic when printed in B&W | Non-monotonic ramp, unreadable |
            """)
            st.info("""
**Why rainbow misleads**: The jet/rainbow colormap's sharp hue transitions (e.g., green→yellow, blue→cyan)
create artificial categorical boundaries in continuous data. A reading of 140 μg/m³ vs 160 μg/m³ can
appear categorically different (green vs yellow) when the true difference is only ~14% — directly
inflating the visual Lie Factor. Plasma's strictly increasing luminance makes every μg/m³ step feel
proportionally equal to the eye.
            """)

        # ── Concluding Statement ─────────────────────────────
        st.divider()
        st.success("""
**Concluding Statement**: The 3D bar chart was REJECTED (Lie Factor >> 1, occlusion, perspective distortion).
A Tufte-compliant 2D alternative — the Bivariate Bubble Map — encodes PM2.5 and
population density honestly using the plasma perceptually-uniform sequential colormap.
This approach scales perfectly to 100+ stations without visual clutter.
Data-ink ratio is maximised; chartjunk is eliminated.
        """)

if __name__ == "__main__":
    main()
