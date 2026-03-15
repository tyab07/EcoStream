import duckdb
import pandas as pd
import numpy as np

def get_zone(loc_name):
    loc_lower = loc_name.lower()
    if "manali" in loc_lower or "talkatora" in loc_lower or "solapur" in loc_lower or "industrial" in loc_lower:
        return "Industrial"
    return "Residential"

query = """
    SELECT 
        location_name,
        parameter,
        COUNT(value) as c,
        COUNT(NULLIF(value, 'nan')) as valid_c
    FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=true)
    GROUP BY location_name, parameter
"""
df = duckdb.query(query).to_df()
df['parameter'] = df['parameter'].str.lower()
df['parameter'] = df['parameter'].replace({
    'pm2.5': 'pm25', 'pm10': 'pm10', 'no2': 'no2', 'o3': 'ozone', 
    'temperature': 'temperature', 'relativehumidity': 'humidity', 'humidity': 'humidity'
})

print("Missing params by location:")
for loc in df['location_name'].unique():
    subset = df[df['location_name'] == loc]
    params = subset['parameter'].unique()
    zone = get_zone(loc)
    print(f"[{zone}] {loc}: {len(params)} params -> {params}")
