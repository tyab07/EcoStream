import json
import duckdb
import pandas as pd

# 1. Read metadata
with open('locations_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"Total metadata locations: {len(metadata)}")
if len(metadata) > 0:
    for k, v in metadata[0].items():
        if k != "sensors":
            print(f"{k}: {v}")
    print(f"Sensors: {len(metadata[0]['sensors'])}")

# 2. Test reading duckdb with filename
con = duckdb.connect()
df = con.execute("""
    SELECT 
        filename, 
        column0 AS val
    FROM read_csv_auto('data/raw/*/*.csv', filename=true)
    LIMIT 10
""").df()

print(df)
