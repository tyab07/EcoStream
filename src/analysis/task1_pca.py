"""
Task 1 — Dimensionality Reduction via Principal Component Analysis (PCA)
========================================================================

Purpose:
    Reduce the 6-dimensional pollutant space (PM2.5, PM10, NO2, O3,
    Temperature, Humidity) into 2 principal components to reveal whether
    Industrial and Residential zones have distinguishable pollution signatures.

Method:
    1. Load all 6 parameters per station-hour via DuckDB.
    2. Standardize features (zero mean, unit variance).
    3. Apply PCA (n_components=2).
    4. Plot scatter with Zone coloring and centroid markers.
"""

import duckdb
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Zone classifier ──────────────────────────────────────────
def get_zone(loc_name: str) -> str:
    loc_lower = loc_name.lower()
    industrial_keywords = ["manali", "talkatora", "solapur", "industrial", "district", "midc"]
    return "Industrial" if any(k in loc_lower for k in industrial_keywords) else "Residential"


# ── Data Loading ─────────────────────────────────────────────
def load_pca_data() -> pd.DataFrame:
    """Load and pivot all 6 pollutant parameters into a single DataFrame."""
    con = duckdb.connect()
    if not os.path.exists('duckdb_temp'):
        os.makedirs('duckdb_temp')
    con.execute("PRAGMA temp_directory='duckdb_temp';")
    con.execute("SET memory_limit='1GB';")
    con.execute("SET threads TO 2;")

    con.execute(
        "CREATE TEMP TABLE metadata AS "
        "SELECT CAST(id AS VARCHAR) as loc_id, name AS location_name "
        "FROM read_json_auto('locations_metadata.json')"
    )

    sql = r"""
        WITH files AS (
            SELECT column0 AS value, filename, row_number() OVER () as rn
            FROM read_csv_auto('data/raw/*/*.csv', filename=true)
        ),
        parsed AS (
            SELECT value,
                   regexp_extract(filename, 'station=(\d+)', 1) AS loc_id,
                   regexp_extract(filename, '([^\/]+)\.csv$', 1) AS param_raw,
                   rn, filename
            FROM files
        ),
        ordered AS (
            SELECT *, row_number() OVER (PARTITION BY filename ORDER BY rn) as row_idx
            FROM parsed
        ),
        mapped AS (
            SELECT p.value, m.location_name,
                CASE
                    WHEN lower(p.param_raw) IN ('pm2.5','pm25') THEN 'pm25'
                    WHEN lower(p.param_raw) = 'pm10' THEN 'pm10'
                    WHEN lower(p.param_raw) = 'no2' THEN 'no2'
                    WHEN lower(p.param_raw) IN ('o3','ozone') THEN 'ozone'
                    WHEN lower(p.param_raw) = 'temperature' THEN 'temperature'
                    WHEN lower(p.param_raw) IN ('humidity','relativehumidity') THEN 'humidity'
                    ELSE NULL
                END AS param_norm,
                make_timestamp(2025,1,1,0,0,0) + INTERVAL (p.row_idx - 1) HOUR AS ts_2025
            FROM ordered p
            INNER JOIN metadata m ON p.loc_id = m.loc_id
            WHERE p.value IS NOT NULL
        ),
        pivoted AS (
            SELECT location_name, ts_2025,
                avg(CASE WHEN param_norm = 'pm25' THEN value END) AS pm25,
                avg(CASE WHEN param_norm = 'pm10' THEN value END) AS pm10,
                avg(CASE WHEN param_norm = 'no2' THEN value END) AS no2,
                avg(CASE WHEN param_norm = 'ozone' THEN value END) AS ozone,
                avg(CASE WHEN param_norm = 'temperature' THEN value END) AS temperature,
                avg(CASE WHEN param_norm = 'humidity' THEN value END) AS humidity
            FROM mapped WHERE param_norm IS NOT NULL
            GROUP BY location_name, ts_2025
        )
        SELECT * FROM pivoted
        WHERE pm25 IS NOT NULL AND pm10 IS NOT NULL
          AND no2 IS NOT NULL AND ozone IS NOT NULL
          AND temperature IS NOT NULL AND humidity IS NOT NULL
    """

    df = con.execute(sql).df()
    con.close()

    df['Zone'] = df['location_name'].apply(get_zone)
    return df


# ── PCA Computation ──────────────────────────────────────────
FEATURE_COLS = ['pm25', 'pm10', 'no2', 'ozone', 'temperature', 'humidity']


def run_pca(df: pd.DataFrame):
    """
    Standardize features and run PCA(n=2).

    Returns:
        df       — DataFrame with PC1, PC2 columns added
        pca_obj  — fitted PCA object
        loadings — DataFrame of feature loadings
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    df = df.copy()
    df['PC1'] = pcs[:, 0]
    df['PC2'] = pcs[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=FEATURE_COLS
    )
    return df, pca, loadings


def compute_centroids(df: pd.DataFrame) -> dict:
    """Compute zone centroids and separation distance."""
    res = df[df['Zone'] == 'Residential'][['PC1', 'PC2']].mean()
    ind = df[df['Zone'] == 'Industrial'][['PC1', 'PC2']].mean()
    gap = np.sqrt((ind['PC1'] - res['PC1'])**2 + (ind['PC2'] - res['PC2'])**2)
    return {
        'res_pc1': res['PC1'], 'res_pc2': res['PC2'],
        'ind_pc1': ind['PC1'], 'ind_pc2': ind['PC2'],
        'centroid_gap': gap
    }


# ── Standalone Execution ─────────────────────────────────────
if __name__ == '__main__':
    print("=== Task 1: PCA Dimensionality Reduction ===\n")
    df = load_pca_data()
    print(f"Loaded {len(df):,} records across {df['location_name'].nunique()} stations")

    df, pca_obj, loadings = run_pca(df)
    centroids = compute_centroids(df)

    print(f"\nVariance Captured: {sum(pca_obj.explained_variance_ratio_):.1%}")
    print(f"  PC1: {pca_obj.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2: {pca_obj.explained_variance_ratio_[1]:.1%}")
    print(f"Centroid Gap: {centroids['centroid_gap']:.3f}")
    print(f"\nFeature Loadings:\n{loadings}")
    print("\n=== Task 1 Complete ===")
