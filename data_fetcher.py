import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class OpenAQFetcher:
    """
    Fetcher for OpenAQ API v3 data.
    """
    BASE_URL = "https://api.openaq.org/v3"
    
    # Required parameters mapping (name to IDs)
    # PM2.5 (2), PM10 (1), NO2 (5, 15), Ozone (3, 10, 32), Temperature (100), Humidity (98, 134)
    REQUIRED_PARAMS = {
        "pm25": [2],
        "pm10": [1],
        "no2": [5, 15],
        "o3": [3, 10, 32],
        "temperature": [100],
        "humidity": [98, 134]
    }

    def __init__(self):
        self.api_key = os.getenv("OPENAQ_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAQ_API_KEY not found in .env file")
        
        self.headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json"
        }
        
        # Ensure data directory exists
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint, params=None, retries=5):
        """Helper to make rate-limited API requests with robust retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        # Respect rate limits
        time.sleep(1.2) 
        
        response = None
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=45)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if retries > 0:
                print(f"  Network error ({e}). Retrying in 15s... ({retries} retry(s) left)")
                time.sleep(15)
                return self._make_request(endpoint, params, retries - 1)
            print(f"  Failed after max retries: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            status_code = response.status_code if response is not None else 0
            
            # Retryable Errors: 429 (Rate Limit), 408 (Timeout), 5xx (Server Error)
            if status_code in [408, 429] or (500 <= status_code < 600):
                if retries > 0:
                    wait_time = 70 if status_code == 429 else 20
                    print(f"  HTTP {status_code} Error. Waiting {wait_time}s and retrying... ({retries} retry(s) left)")
                    time.sleep(wait_time)
                    return self._make_request(endpoint, params, retries - 1)
            
            print(f"  HTTP Error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  Unexpected error requesting {url}: {e}")
            return None

    def discover_locations(self, count=100):
        """
        Find locations that measure all required parameters.
        Filtering for 2025 data availability is tricky via locations endpoint directly,
        so we'll fetch locations and verify they have recent data.
        """
        print(f"Discovering {count} global locations with required parameters...")
        locations = []
        page = 1
        limit = 100
        
        # We'll pull locations that have at least PM2.5 and PM10 as a starting point
        # then filter for others in the results.
        params = {
            "limit": limit,
            "page": page,
            "parameters_id": [2, 1], # PM2.5 and PM10
        }

        while len(locations) < count:
            data = self._make_request("locations", params)
            if not data or not data.get("results"):
                break
            
            for loc in data["results"]:
                available_params = {p["parameter"]["id"] for p in loc.get("sensors", [])}
                
                # Check if all 6 groups have at least one ID present
                met_requirements = True
                for group, ids in self.REQUIRED_PARAMS.items():
                    if not any(pid in available_params for pid in ids):
                        met_requirements = False
                        break
                
                if met_requirements:
                    locations.append(loc)
                    if len(locations) >= count:
                        break
            
            print(f"Checked page {page}, found {len(locations)} matching locations...")
            page += 1
            params["page"] = page
            
            # Safety break to avoid infinite loops if it's hard to find 100
            if page > 50: 
                break

        return locations[:count]

    def fetch_sensor_data(self, sensor_id, date_from, date_to):
        """Fetch hourly data for a specific sensor in chunks of 1 month."""
        all_measurements = []
        start_date = datetime.strptime(date_from, "%Y-%m-%d")
        end_dt = datetime.strptime(date_to, "%Y-%m-%d")
        
        current_start = start_date
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=30), end_dt)
            
            print(f"  Fetching sensor {sensor_id} from {current_start.date()} to {current_end.date()}")
            
            params = {
                "datetimeFrom": current_start.isoformat(),
                "datetimeTo": current_end.isoformat(),
                "limit": 1000
            }
            
            page = 1
            while page <= 10: # Safety cap: shouldn't need more than 10 pages for hourly data
                params["page"] = page
                # Using /sensors/{id}/hours for hourly averages
                data = self._make_request(f"sensors/{sensor_id}/hours", params)
                
                if not data or not data.get("results"):
                    break
                
                all_measurements.extend(data["results"])
                
                if len(data["results"]) < 1000:
                    break
                page += 1
            
            if page > 10:
                print(f"  Warning: Reached pagination cap for sensor {sensor_id}")
            
            current_start = current_end + timedelta(seconds=1)
            
        return all_measurements

    def run_single_loc(self, loc):
        """Process only required sensors for a single location and save immediately."""
        loc_id = loc["id"]
        loc_name = loc["name"].replace("/", "_").replace("\\", "_").replace(":", "_")
        
        # Create location-specific directory
        loc_dir = self.data_dir / f"{loc_name}_{loc_id}"
        
        # Map parameter IDs to names to track them
        param_id_to_group = {}
        for group, ids in self.REQUIRED_PARAMS.items():
            for pid in ids:
                param_id_to_group[pid] = group

        # PRE-CHECK: If folder exists, check if all 6 pollutant groups are there
        if loc_dir.exists():
            existing_files = [f.name.split("_")[0] for f in loc_dir.glob("*.csv")]
            # Map existing file names back to their groups
            present_groups = set()
            for found_name in existing_files:
                for group, ids in self.REQUIRED_PARAMS.items():
                    # We'll just check if the name starts with the group name 
                    # or matches specific known variations
                    if found_name.startswith(group) or \
                       (group == "humidity" and found_name == "relativehumidity") or \
                       (group == "o3" and found_name == "o3"):
                        present_groups.add(group)
            
            if len(present_groups) >= 6:
                print(f"Skipping Location: {loc_name} (All 6 params already present)")
                return True

        loc_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing Location: {loc_name} (ID: {loc_id})")
        
        # Track which parameters we've already fetched for this location
        processed_params = set()
        # Find already existing files in this specific run to avoid re-fetching within session
        for f in loc_dir.glob("*.csv"):
            p_name = f.name.split("_")[0]
            processed_params.add(p_name)

        for sensor in loc["sensors"]:
            p_id = sensor["id"]
            p_info = sensor["parameter"]
            p_param_id = p_info["id"]
            p_name = p_info["name"]
            p_units = p_info["units"]
            
            # Check if this sensor is one of our 6 required groups
            param_group = param_id_to_group.get(p_param_id)
            if not param_group:
                continue
            
            # If we already fetched this pollutant group, skip
            if param_group in processed_params:
                continue
                
            print(f"  Starting REQUIRED parameter: {p_name} (Sensor: {p_id})")
            measurements = self.fetch_sensor_data(p_id, "2025-01-01", "2025-12-31")
            
            if measurements:
                df = pd.DataFrame(measurements)
                df["parameter"] = p_name
                df["units"] = p_units
                df["location_id"] = loc_id
                df["location_name"] = loc_name
                
                if "period" in df.columns:
                    df["timestamp"] = df["period"].apply(lambda x: x.get("datetimeFrom") if isinstance(x, dict) else None)
                
                # Save this specific sensor using temp-rename pattern for atomic writes
                filename = loc_dir / f"{p_name}_sensor_{p_id}.csv"
                tmp_filename = filename.with_suffix(".tmp")
                
                cols = ["timestamp", "parameter", "value", "units", "location_name", "location_id"]
                df_to_save = df[[c for c in cols if c in df.columns]]
                
                # Write to temp file first
                df_to_save.to_csv(tmp_filename, index=False)
                
                # Atomic rename (stable file)
                if tmp_filename.exists():
                    os.replace(tmp_filename, filename)
                    print(f"    [STABLE SAVED] {len(df)} records to {filename.name}")
                processed_params.add(param_group)
        
        return True

    def run(self, count=100):
        """Main orchestrator."""
        locations = self.discover_locations(count)
        
        with open("locations_metadata.json", "w") as f:
            json.dump(locations, f, indent=2)
        
        total_locs = len(locations)
        for idx, loc in enumerate(locations):
            print(f"[{idx+1}/{total_locs}] ", end="")
            self.run_single_loc(loc)

if __name__ == "__main__":
    fetcher = OpenAQFetcher()
    # Run the full 100 location fetch with skip logic enabled
    fetcher.run(100)
