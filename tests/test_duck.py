import duckdb
import pandas as pd

q2 = """
    WITH files AS (
        SELECT 
            column0 AS value,
            filename,
            row_number() OVER () as rn
        FROM read_csv_auto('data/raw/*/*.csv', filename=true)
    ),
    parsed AS (
        SELECT 
            value,
            CAST(regexp_extract(filename, 'station=(\d+)', 1) AS INTEGER) AS location_id,
            regexp_extract(filename, '([^\\\/]+)\.csv$', 1) AS parameter,
            rn,
            row_number() OVER (PARTITION BY filename ORDER BY rn) as row_idx
        FROM files
    )
    SELECT
        location_id,
        parameter,
        value,
        make_timestamp(2025,1,1,0,0,0) + INTERVAL (row_idx - 1) HOUR AS ts
    FROM parsed
    LIMIT 20
"""
print(duckdb.query(q2).to_df())
