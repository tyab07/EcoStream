import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from data_fetcher import OpenAQFetcher

# Get R K Puram station
with open('locations_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)
loc = [m for m in metadata if m['id'] == 17][0]
pm25_sensor = next(s for s in loc['sensors'] if s['parameter']['name'].lower() in ['pm2.5', 'pm25'])

print(f"Fetching PM2.5 from OpenAQ for {loc['name']}")
fetcher = OpenAQFetcher()
# The API might be returning the newest data first. We'll fetch 100 records
res = fetcher._make_request(f"sensors/{pm25_sensor['id']}/hours", {'limit': 100, 'datetimeFrom': '2025-01-01T00:00:00Z', 'datetimeTo': '2025-01-05T00:00:00Z'})
api_values = [x['value'] for x in res.get('results', [])]

# Load local CSV
local_csv = os.path.join("data", "raw", "station=17", "pm25.csv")
local_df = pd.read_csv(local_csv, header=None, names=['value'])
local_values = local_df['value'].head(100).tolist()

plt.figure(figsize=(10, 5))
plt.plot(local_values, label='Local CSV Data', marker='o', alpha=0.7)
plt.plot(api_values, label='OpenAQ API Data', marker='x', alpha=0.7)
plt.title('PM2.5 Comparison: Local vs OpenAQ API (First 100 hours)')
plt.xlabel('Hours (Index)')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.tight_layout()
plt.savefig('openaq_comparison.png')
print("Comparison chart saved to openaq_comparison.png")
