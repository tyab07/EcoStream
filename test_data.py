import duckdb
import pandas as pd
import numpy as np

query = """
    SELECT 
        location_name,
        TRY_CAST(timestamp AS TIMESTAMP) AS ts,
        parameter,
        value
    FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=true)
    WHERE TRY_CAST(timestamp AS TIMESTAMP) IS NOT NULL
"""
df_raw = duckdb.query(query).to_df()
df_raw['ts'] = pd.to_datetime(df_raw['ts'])
df_raw = df_raw[~((df_raw.ts.dt.month == 2) & (df_raw.ts.dt.day == 29))]
df_raw['ts_2025'] = df_raw['ts'].apply(lambda dt: pd.Timestamp(year=2025, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute, second=dt.second))

df_raw['parameter'] = df_raw['parameter'].str.lower()
df_raw['parameter'] = df_raw['parameter'].replace({
    'pm2.5': 'pm25', 'pm10': 'pm10', 'no2': 'no2', 'o3': 'ozone', 
    'temperature': 'temperature', 'relativehumidity': 'humidity', 'humidity': 'humidity'
})

print(df_raw['parameter'].value_counts())

df_pivot = df_raw.pivot_table(index=['location_name', 'ts_2025'], columns='parameter', values='value', aggfunc='mean').reset_index()
print(df_pivot.head())
print("Missing values in pivot:")
print(df_pivot.isna().sum())

print("Rows before dropna:", len(df_pivot))
df_valid = df_pivot.dropna(subset=['pm25', 'pm10', 'no2', 'ozone', 'temperature', 'humidity'])
print("Rows after dropna:", len(df_valid))
