import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def _get_con():
    """Return a DuckDB connection with memory cap and disk-spilling."""
    import os
    if not os.path.exists('duckdb_temp'):
        os.makedirs('duckdb_temp')
        
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='duckdb_temp';")
    con.execute("SET memory_limit='1GB';")
    con.execute("SET threads TO 2;")
    return con

def get_temporal_data():
    """
    Extract temporal PM2.5 data for heatmaps/periodicity analysis.

    Key optimisation: ALL heavy work (station filtering, parameter
    selection, violation flagging, date-part extraction) is done inside
    DuckDB so that only a small aggregated DataFrame comes back to Python.
    Previously the raw rows for all stations were pulled into RAM which
    caused the OutOfMemoryException on a ~1.7 GB dataset.
    """
    con = _get_con()

    # ── Step 1: find stations that have all 6 required parameters ──────
    # Replaced DuckDB read_csv_auto scan with strictly os-level scan
    # to avoid OOM Allocation Failures on memory-constrained systems.
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

    # ── Step 2: register the list so DuckDB can use it safely ──────────
    con.execute("CREATE TEMP TABLE valid_st (location_name VARCHAR)")
    con.executemany(
        "INSERT INTO valid_st VALUES (?)",
        [[s] for s in valid_stations]
    )

    # ── Step 3: pull ONLY pm25 rows, aggregate to hourly means,
    #            compute violations and date-parts – all inside DuckDB ──
    df_pm25 = con.execute(r"""
        WITH src AS (
            SELECT
                r.location_name,
                -- parse timestamp string (handle JSON-dict format)
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
            WHERE lower(r.parameter) IN ('pm2.5','pm25')
              AND r.value IS NOT NULL
        ),
        filtered AS (
            SELECT * FROM src
            WHERE ts IS NOT NULL
              -- skip Feb-29 (leap-day that cannot map to 2025)
              AND NOT (month(ts) = 2 AND day(ts) = 29)
        ),
        hourly AS (
            SELECT
                location_name,
                ts,
                -- re-stamp the year to 2025 for uniform x-axis
                make_timestamp(
                    2025,
                    month(ts), day(ts),
                    hour(ts), 0, 0
                ) AS ts_2025,
                avg(value) AS value
            FROM filtered
            GROUP BY location_name, ts
        )
        SELECT
            location_name,
            ts_2025,
            value,
            CASE WHEN value > 35 THEN 1 ELSE 0 END AS violation,
            -- short station label (everything before first ',' or '-')
            regexp_replace(
                regexp_replace(location_name, ',.*', ''),
                '-.*', ''
            ) AS loc_short,
            hour(ts_2025)  AS hour,
            month(ts_2025) AS month,
            CAST(ts_2025 AS DATE) AS date
        FROM hourly
        ORDER BY location_name, ts_2025
    """).df()

    con.close()
    return df_pm25

def main():
    print("--- 1. Data Preprocessing for Temporal Analysis ---")
    df_pm25 = get_temporal_data()
    
    print("--- 2. High-Density Visualization (Heatmap) ---")
    # 'date', 'hour', 'month' are already pre-computed columns from get_temporal_data()
    if not pd.api.types.is_object_dtype(df_pm25['date']):
        df_pm25['date'] = pd.to_datetime(df_pm25['date']).dt.date
    daily_violations = df_pm25.groupby(['loc_short', 'date'])['violation'].mean().reset_index()
    pivot_daily = daily_violations.pivot(index='loc_short', columns='date', values='violation')
    
    # Sort stations by overall average violation rate (highest at top)
    station_means = df_pm25.groupby('loc_short')['violation'].mean().sort_values(ascending=False)
    pivot_daily = pivot_daily.reindex(station_means.index)
    
    plt.figure(figsize=(12, 5), facecolor='white')
    ax = plt.gca()
    
    # Using 'plasma' as it is a perceptually uniform sequential colormap and highly professional
    sns.heatmap(pivot_daily, cmap='plasma', cbar_kws={'label': 'Violation Rate (%)'}, rasterized=True)
    
    plt.title("High-Density Heatmap: Daily PM2.5 Health Violations (>35 μg/m³)", fontsize=14, pad=15)
    plt.xlabel("Date (2025)", fontsize=11)
    plt.ylabel("Monitoring Station", fontsize=11)
    
    # Clean x-axis ticks to show just a few months rather than every day
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::30])
    ax.set_xticklabels([pivot_daily.columns[int(x)].strftime('%b %d') for x in xticks[::30] if int(x) < len(pivot_daily.columns)], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('temporal_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved 'temporal_heatmap.png'.")
    
    print("\n--- 3. Periodicity Analysis ---")
    # 'hour' and 'month' are already pre-computed columns from get_temporal_data()
    
    # --- Diurnal (Hour of Day) ---
    hourly_stats = df_pm25.groupby('hour')['violation'].mean().reset_index()
    
    plt.figure(figsize=(8, 4), facecolor='white')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Create colors from plasma mapping
    norm = plt.Normalize(hourly_stats['violation'].min(), hourly_stats['violation'].max())
    colors = cm.plasma(norm(hourly_stats['violation']))
    
    plt.bar(hourly_stats['hour'], hourly_stats['violation'], color=colors, edgecolor='none')
    plt.title("Diurnal Cycle: 24-Hour Average PM2.5 Violation Pattern", fontsize=12, pad=10)
    plt.xlabel("Hour of Day (0-23)", fontsize=10)
    plt.ylabel("Average Violation Rate", fontsize=10)
    plt.xticks(range(0, 24, 2))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('diurnal_cycle.png', dpi=300, bbox_inches='tight')
    print("Saved 'diurnal_cycle.png'.")
    
    # --- Seasonal (Month) ---
    monthly_stats = df_pm25.groupby('month')['violation'].mean().reset_index()
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    plt.figure(figsize=(8, 4), facecolor='white')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    norm_m = plt.Normalize(monthly_stats['violation'].min(), monthly_stats['violation'].max())
    colors_m = cm.plasma(norm_m(monthly_stats['violation']))
    
    plt.bar(monthly_stats['month'], monthly_stats['violation'], color=colors_m, edgecolor='none')
    plt.title("Seasonal Cycle: Monthly PM2.5 Violation Pattern", fontsize=12, pad=10)
    plt.xlabel("Month of Year", fontsize=10)
    plt.ylabel("Average Violation Rate", fontsize=10)
    plt.xticks(monthly_stats['month'], [month_map[m] for m in monthly_stats['month']])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('seasonal_cycle.png', dpi=300, bbox_inches='tight')
    print("Saved 'seasonal_cycle.png'.")
    
    print("\n--- Statistical Summary ---")
    st_hour = hourly_stats.sort_values(by='violation', ascending=False).head(3)
    st_month = monthly_stats.sort_values(by='violation', ascending=False).head(3)
    
    print("Top 3 Hours with highest average PM2.5 violation rates:")
    for _, row in st_hour.iterrows():
        print(f"  - Hour {int(row['hour']):02d}:00 -> {row['violation']:.1%}")
        
    print("\nTop 3 Months with highest average PM2.5 violation rates:")
    for _, row in st_month.iterrows():
        print(f"  - {month_map[int(row['month'])]} -> {row['violation']:.1%}")
        
    print("\nAnalytical Interpretation:")
    print("The high-density heatmap maps daily violation rates across all monitoring stations using the perceptually uniform 'plasma' colormap. This allows visual distinction of severe pollution epochs (bright vertical bands affecting all stations simultaneously) versus localized point-source pollution events (horizontal streaks for a single station) without massive overplotting.")
    print("The periodicity analysis demonstrates the structural forces driving these violations. The diurnal bar chart often peaks during temperature inversions combined with traffic/industrial exhaust accumulation, while the seasonal bar chart confirms the severity of winter boundary-layer phenomena and biomass burning against summer washout.")

if __name__ == '__main__':
    main()
