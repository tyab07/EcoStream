# -*- coding: utf-8 -*-
"""
Task 3 - Distribution Modeling with Log-Scaled Tail Integrity
Senior Environmental Data Scientist | Smart City Environmental Intelligence
"""

import sys
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from scipy.stats import skew

# Ensure UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# STEP 1 — Data Load & Preparation
# ─────────────────────────────────────────────

def load_pm25() -> pd.DataFrame:
    """Load PM2.5 data via DuckDB with a memory cap, return clean DataFrame."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1.5GB';")
    con.execute("SET threads TO 2;")

    # Step 1: find stations with all 6 parameters using extremely fast OS scan
    # to avoid the DuckDB OOM error on read_csv_auto scanning millions of rows
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
        return pd.DataFrame()

    # Step 2: register station list for safe JOIN
    con.execute("CREATE TEMP TABLE valid_st (location_name VARCHAR)")
    con.executemany("INSERT INTO valid_st VALUES (?)", [[s] for s in valid_stations])

    # Step 3: pull only PM2.5 rows via JOIN — no full-table pandas pull
    df = con.execute(r"""
        SELECT
            r.location_name,
            TRY_CAST(
                CASE
                    WHEN r.timestamp LIKE '{%'
                    THEN regexp_extract(r.timestamp, '''utc'':\s*''([^'']+)''', 1)
                    ELSE r.timestamp
                END
            AS TIMESTAMP) AS ts,
            r.value
        FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=true) r
        INNER JOIN valid_st v ON r.location_name = v.location_name
        WHERE lower(r.parameter) IN ('pm2.5','pm25')
          AND r.value IS NOT NULL
          AND CAST(r.value AS DOUBLE) >= 0
    """).df()
    con.close()

    # Lightweight pandas cleanup on the small result
    df.dropna(subset=['ts'], inplace=True)
    df['ts'] = df['ts'].dt.floor('h')
    # Remove Feb 29 to avoid year-relabelling errors
    df = df[~((df['ts'].dt.month == 2) & (df['ts'].dt.day == 29))]
    ts = df['ts']
    df['ts_2025'] = pd.to_datetime({
        'year': 2025,
        'month': ts.dt.month,
        'day': ts.dt.day,
        'hour': ts.dt.hour
    })
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['value'], inplace=True)

    return df


# Target industrial station for Task 3 analysis
TARGET_STATION = 'Talkatora District Industries Center, Lucknow - CPCB'


def select_target_station(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Select Talkatora industrial station; falls back to highest-mean if not found."""
    if TARGET_STATION in df['location_name'].values:
        station = TARGET_STATION
    else:
        # Graceful fallback
        station = df.groupby('location_name')['value'].mean().idxmax()
        print(f"  [WARN] Target station not found. Falling back to: {station}")
    pm25_series = df[df['location_name'] == station]['value'].reset_index(drop=True)
    return station, pm25_series


def get_distribution_data():
    """Public API: returns (station_name, pm25_array) for use by app.py."""
    df = load_pm25()
    station, pm25_series = select_target_station(df)
    return station, pm25_series.values.astype(float)


# ─────────────────────────────────────────────
# STEP 2 — Distribution Visualizations
# ─────────────────────────────────────────────

THRESHOLD = 200.0   # μg/m³ "Extreme Hazard" level
ACCENT    = '#e84118'  # single honest accent color used for threshold markers

def freedman_diaconis_bins(data: np.ndarray) -> int:
    """Optimal bin count via Freedman-Diaconis rule."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(len(data)))
    bin_width = 2.0 * iqr * len(data) ** (-1 / 3)
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(n_bins, 10)


def compute_statistics(pm25: np.ndarray) -> dict:
    """Return a dict of all required extreme-event statistics."""
    p99      = float(np.percentile(pm25, 99))
    prob_ext = float(np.mean(pm25 > THRESHOLD))
    return {
        'n'         : len(pm25),
        'p99'       : p99,
        'prob_ext'  : prob_ext,
        'count_ext' : int(np.sum(pm25 > THRESHOLD)),
        'pct_ext'   : prob_ext * 100,
        'skewness'  : float(skew(pm25)),
    }


def build_histogram_fig(pm25: np.ndarray, station: str) -> go.Figure:
    """Plotly interactive histogram for the dashboard."""
    n_bins = freedman_diaconis_bins(pm25)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pm25, nbinsx=n_bins,
        marker_color='#2980b9', opacity=0.85,
        name='PM2.5 Observations'
    ))
    fig.add_vline(
        x=THRESHOLD, line_dash='dash', line_color=ACCENT, line_width=2,
        annotation_text=f'Extreme Hazard ({int(THRESHOLD)} μg/m³)',
        annotation_position='top right', annotation_font_color=ACCENT
    )
    fig.update_layout(
        title=f'PM2.5 Distribution — {station}<br><sup>Freedman-Diaconis Bins: {n_bins}</sup>',
        xaxis_title='PM2.5 Concentration (μg/m³)',
        yaxis_title='Frequency (Count)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return fig


def build_survival_fig(pm25: np.ndarray, station: str, p99: float) -> go.Figure:
    """Plotly interactive log-scale survival CDF for the dashboard."""
    sorted_pm25 = np.sort(pm25)
    n = len(sorted_pm25)
    survival = np.clip(1.0 - np.arange(1, n + 1) / n, 1e-6, 1.0)

    fig = go.Figure()

    # Main survival curve
    fig.add_trace(go.Scatter(
        x=sorted_pm25, y=survival,
        mode='lines', line=dict(color='#2980b9', width=2),
        name='Survival Function S(x) = P(X > x)'
    ))

    # Extreme tail shading
    mask = sorted_pm25 >= THRESHOLD
    if mask.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([sorted_pm25[mask], sorted_pm25[mask][::-1]]),
            y=np.concatenate([survival[mask], np.full(mask.sum(), 1e-6)]),
            fill='toself', fillcolor='rgba(232,65,24,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Extreme Tail (PM2.5 > 200)'
        ))

    # Threshold vertical
    fig.add_vline(
        x=THRESHOLD, line_dash='dash', line_color=ACCENT, line_width=2,
        annotation_text=f'{int(THRESHOLD)} μg/m³ threshold',
        annotation_font_color=ACCENT
    )

    # 99th percentile marker
    p99_surv = max(float(np.mean(pm25 > p99)), 1e-6)
    fig.add_trace(go.Scatter(
        x=[p99], y=[p99_surv],
        mode='markers+text',
        marker=dict(color='#f39c12', size=10, symbol='circle'),
        text=[f'99th pct<br>{p99:.1f} μg/m³'],
        textposition='top right',
        textfont=dict(color='#f39c12', size=10),
        name=f'99th Percentile ({p99:.1f} μg/m³)'
    ))

    fig.update_layout(
        title=f'Log-Scaled Survival Function — {station}<br>'
              '<sup>Reveals rare extreme events invisible in linear histograms</sup>',
        xaxis_title='PM2.5 Concentration (μg/m³)',
        yaxis_title='Survival Probability  P(PM2.5 > x)  [log scale]',
        yaxis_type='log',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=0, r=0, t=65, b=0),
    )
    return fig


def plot_histogram(pm25: np.ndarray, station: str) -> None:
    """PLOT A — Standard histogram with Freedman-Diaconis binning."""
    n_bins = freedman_diaconis_bins(pm25)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')

    ax.hist(pm25, bins=n_bins, color='#2980b9', alpha=0.85, edgecolor='none',
            linewidth=0, label='PM2.5 Observations')

    # Vertical line at 200 μg/m³
    ax.axvline(THRESHOLD, color=ACCENT, linewidth=1.8, linestyle='--',
               label=f'Extreme Hazard Threshold ({int(THRESHOLD)} μg/m³)')

    # Annotation
    ylim_top = ax.get_ylim()[1]
    ax.annotate(f'≥ {int(THRESHOLD)} μg/m³\n(Extreme Hazard)',
                xy=(THRESHOLD, ylim_top * 0.7),
                xytext=(THRESHOLD + max(pm25) * 0.04, ylim_top * 0.72),
                fontsize=9, color=ACCENT,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.2))

    ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Frequency (Count)', fontsize=11)
    ax.set_title(f'PM2.5 Distribution — {station}\n'
                 f'Freedman–Diaconis Bins: {n_bins}', fontsize=12, pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: histogram.png")


def plot_log_survival_cdf(pm25: np.ndarray, station: str,
                          p99: float) -> None:
    """PLOT B — Log-scaled Survival Function (1 - ECDF)."""
    sorted_pm25 = np.sort(pm25)
    n = len(sorted_pm25)
    # Survival = 1 - ECDF (probability of exceeding x)
    survival = 1.0 - np.arange(1, n + 1) / n

    # Clip to avoid log(0)
    survival = np.clip(survival, 1e-6, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')

    # Main curve
    ax.semilogy(sorted_pm25, survival, color='#2980b9', linewidth=1.5,
                label='Survival Function (1 − ECDF)')

    # Vertical line at 200 μg/m³
    ax.axvline(THRESHOLD, color=ACCENT, linewidth=1.8, linestyle='--',
               label=f'Extreme Hazard Threshold ({int(THRESHOLD)} μg/m³)')

    # Shade the extreme tail region
    mask = sorted_pm25 >= THRESHOLD
    if mask.any():
        ax.fill_betweenx(survival[mask],
                         sorted_pm25[mask],
                         sorted_pm25.max(),
                         alpha=0.18, color=ACCENT,
                         label='Extreme Tail (PM2.5 > 200)')

    # Mark 99th percentile
    p99_survival = np.mean(pm25 > p99)
    p99_survival = max(p99_survival, 1e-6)
    ax.scatter([p99], [p99_survival], color='#f39c12', zorder=5, s=60,
               label=f'99th Percentile ({p99:.1f} μg/m³)')
    ax.annotate(f'99th pct\n{p99:.1f} μg/m³',
                xy=(p99, p99_survival),
                xytext=(p99 + max(sorted_pm25) * 0.04, p99_survival * 3),
                fontsize=8.5, color='#f39c12',
                arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.1))

    ax.set_xlabel('PM2.5 Concentration (μg/m³)', fontsize=11)
    ax.set_ylabel('Survival Probability  P(PM2.5 > x)  [log scale]', fontsize=11)
    ax.set_title(f'Log-Scaled Survival Function — {station}\n'
                 f'Reveals rare extreme events invisible in linear histograms', fontsize=12, pad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(which='both', axis='y', linestyle='--', alpha=0.2, linewidth=0.7)
    ax.grid(which='major', axis='x', linestyle='--', alpha=0.15, linewidth=0.7)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig('log_scaled_survival_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: log_scaled_survival_cdf.png")


# ─────────────────────────────────────────────
# STEP 3 — Statistical Calculations
# ─────────────────────────────────────────────

def print_statistics(pm25: np.ndarray, station: str) -> float:
    """Compute and print extreme-event statistics. Returns 99th percentile."""
    p99       = np.percentile(pm25, 99)
    prob_ext  = np.mean(pm25 > THRESHOLD)
    count_ext = np.sum(pm25 > THRESHOLD)
    pct_ext   = 100.0 * prob_ext
    skewness  = skew(pm25)

    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  Statistical Summary — {station}")
    print(sep)
    print(f"  Total valid observations      : {len(pm25):,}")
    print(f"  99th Percentile               : {p99:.4f} μg/m³")
    print(f"  P(PM2.5 > {int(THRESHOLD)})              : {prob_ext:.6f}  ({prob_ext*100:.4f}%)")
    print(f"  Count of extreme events       : {count_ext:,}")
    print(f"  Percentage of dataset         : {pct_ext:.4f}%")
    print(f"  Skewness coefficient          : {skewness:.4f}")
    print(sep)
    return p99


# ─────────────────────────────────────────────
# STEP 4 — Analytical Explanation
# ─────────────────────────────────────────────

ANALYTICAL_EXPLANATION = """
=======================================================
  ANALYTICAL EXPLANATION - Tail Integrity & Extreme Risk
=======================================================

1. WHY HISTOGRAMS HIDE TAIL BEHAVIOUR
   Standard histograms use equal-width bins on a linear y-axis. When the
   bulk of data clusters near low concentrations (0–150 μg/m³), the bin
   heights for extreme concentrations (>200 μg/m³) are so small they become
   visually indistinguishable from zero. Rare but health-critical events are
   literally invisible, leading decision-makers to underestimate hazard risk.

2. WHY THE SURVIVAL FUNCTION IS SUPERIOR FOR EXTREME RISK
   The Survival Function S(x) = P(X > x) = 1 − ECDF(x) directly quantifies
   the probability of exceeding any concentration threshold. Instead of
   summarising how many observations fall in a bin, it asks: "How likely is
   the environment to exceed this level at any given hour?" This framing
   matches regulatory and epidemiological risk models that care about
   exceedance probabilities, not density counts.

3. WHY LOG-SCALING IMPROVES TAIL VISIBILITY
   On a linear y-axis, P = 0.0001 and P = 0.001 are visually identical (both
   near zero). A log10 y-axis stretches these out across an entire decade of
   visual space, making one order-of-magnitude difference clearly readable.
   Tail events at probabilities of 1-in-10,000 become just as visually
   accessible as events at 1-in-10, restoring proportional representation of
   all risk levels — the core of "Tail Integrity".

4. INTERPRETATION OF THE 99TH PERCENTILE IN ENVIRONMENTAL REGULATION
   The 99th percentile represents the threshold that 99% of hourly readings
   fall below. Regulatory agencies (e.g., India's CPCB, US EPA) use high
   percentile statistics to set "design values" — concentrations that air
   quality control equipment and urban planning must be able to handle. A
   99th-percentile concentration well above WHO/NAAQS limits signals chronic
   structural air quality failure, not isolated anomalies.

5. WHAT HEAVY RIGHT SKEW IMPLIES ABOUT POLLUTION RISK
   A large positive skewness coefficient confirms a right-skewed distribution:
   most observations are moderate, but the tail extends far to the right with
   high-concentration extreme events. In environmental risk, this implies:
   • Average PM2.5 significantly under-estimates population exposure during
     peak events.
   • Simple mean-based compliance metrics are inadequate and misleading.
   • Risk models must be built on tail-sensitive statistics (e.g., CVaR,
     extreme value theory — GEV or GPD distributions).

6. CONCEPT OF "TAIL INTEGRITY"
   Tail Integrity refers to the property of a visualization or statistical
   summary that faithfully represents the behaviour of rare, extreme values
   without distorting or erasing them due to scale compression or
   over-smoothing. A visualization with high tail integrity:
   • Displays probabilities across multiple orders of magnitude (log scale).
   • Does not clip or aggregate extreme observations into a single overflow bin.
   • Honestly encodes the survival probability at every observed concentration.
   The log-scaled survival CDF achieves all three, making it the gold-standard
   technique for high-stake environmental and financial risk communication.

=======================================================
"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("==== TASK 3 - Distribution Modeling with Log-Scaled Tail Integrity ====\n")

    # Step 1
    print("[Step 1] Loading PM2.5 data via DuckDB...")
    df = load_pm25()
    station, pm25_series = select_target_station(df)
    pm25 = pm25_series.values.astype(float)
    print(f"  Target station   : {station}")
    print(f"  Total valid obs  : {len(pm25):,}")

    # Step 3 — compute stats first (p99 needed for plots)
    print("\n[Step 3] Statistical calculations...")
    p99 = print_statistics(pm25, station)

    # Step 2 — plots
    print("\n[Step 2] Generating visualizations...")
    plot_histogram(pm25, station)
    plot_log_survival_cdf(pm25, station, p99)

    # Step 4 — print explanation
    print(ANALYTICAL_EXPLANATION)
    print("==== TASK 3 COMPLETE ====")


if __name__ == '__main__':
    main()
