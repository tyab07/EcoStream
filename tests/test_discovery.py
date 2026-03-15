import duckdb

query = """
WITH raw_data AS (
    SELECT 
        location_name,
        lower(parameter) as param
    FROM read_csv_auto('data/raw/*/*.csv', ignore_errors=true)
    WHERE value IS NOT NULL
),
normalized_params AS (
    SELECT 
        location_name,
        CASE 
            WHEN param IN ('pm2.5', 'pm25') THEN 'pm25'
            WHEN param = 'pm10' THEN 'pm10'
            WHEN param = 'no2' THEN 'no2'
            WHEN param IN ('o3', 'ozone') THEN 'ozone'
            WHEN param = 'temperature' THEN 'temperature'
            WHEN param IN ('humidity', 'relativehumidity') THEN 'humidity'
            ELSE 'other'
        END AS param_norm
    FROM raw_data
)
SELECT location_name, count(DISTINCT param_norm) as param_count
FROM normalized_params
WHERE param_norm != 'other'
GROUP BY location_name
HAVING param_count >= 6
"""
df = duckdb.query(query).to_df()
print(f"✅ DYNAMIC DISCOVERY SUCCESSFUL")
print(f"System scanned all 'data/raw/*/*.csv' combinations.")
print(f"Total stations currently found possessing all 6 parameters: {len(df)}")
for idx, row in df.iterrows():
    print(f" {idx+1}. {row['location_name']}")
