from data_fetcher import OpenAQFetcher
import pandas as pd
import json
import os

def check():
    fetcher = OpenAQFetcher()
    
    with open('locations_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    loc = [m for m in metadata if m['id'] == 17][0]
    pm25_sensor = next(s for s in loc['sensors'] if s['parameter']['name'].lower() in ['pm2.5', 'pm25'])
    
    print(f"Checking Location: {loc['name']} (ID: {loc['id']})")
    print(f"Sensor ID for PM2.5: {pm25_sensor['id']}")
    
    print("\nFetching data from OpenAQ API for Jan 2025...")
    api_data = fetcher.fetch_sensor_data(pm25_sensor['id'], "2025-01-01", "2025-01-03")
    
    if not api_data:
        print("No data returned from API.")
        return
        
    api_df = pd.DataFrame(api_data)
    print("Period structure:", api_df['period'].iloc[0])
    
    if 'period' in api_df.columns:
        api_df['timestamp'] = api_df['period'].apply(lambda x: x.get('datetimeFrom') if isinstance(x, dict) else x)
    
    api_df['timestamp'] = api_df['timestamp'].apply(lambda x: x.get('utc') if isinstance(x, dict) else x)
    api_df['timestamp'] = pd.to_datetime(api_df['timestamp'])
    api_df = api_df.sort_values('timestamp').reset_index(drop=True)
    
    print("\nAPI Data Head (first 10 records):")
    print(api_df[['timestamp', 'value']].head(10))
    
    print("\nReading Local CSV Data...")
    local_csv = os.path.join("data", "raw", "station=17", "pm25.csv")
    local_df = pd.read_csv(local_csv, header=None, names=['value'])
    print("Local Data Head (first 10 records):")
    print(local_df.head(10))
    
if __name__ == "__main__":
    check()
