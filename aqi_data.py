import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()
HEADERS = {
    "X-API-Key": os.getenv("API_KEY")
}

def year_interval(datetime_from, datetime_to):
    start_year = int(datetime_from[:4])
    end_year = int(datetime_to[:4])
    intervals = []
    
    if start_year == end_year:
        return [(datetime_from, datetime_to)]
    
    intervals.append((datetime_from, f"{start_year + 1}-01-01T00:00:00Z"))
    for year in range(start_year + 1, end_year):
        intervals.append((f"{year}-01-01T00:00:00Z", f"{year}-12-31T00:00:00Z"))
    intervals.append((f"{end_year}-01-01T00:00:00Z", datetime_to))
    
    return intervals


def get_location_info(location_id):
    url = f"https://api.openaq.org/v3/locations/{location_id}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        result = response.json().get("results", [])
        os.makedirs(f'data/location_{location_id}', exist_ok=True)
        
        with open(f'data/location_{location_id}/info.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))


def get_sensor_measurements(sensor_id, datetime_from, datetime_to):
    
    def get_one_year_sensor_measurements(sensor_id, start_time, end_time):
        url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
        measurement_data = []
        
        timestamp_index = pd.date_range(start=start_time, end=end_time, freq='h').strftime("%Y-%m-%dT%H:%M:%SZ")[:-1]
        
        page = 1
        while True:
            params = {
                "datetime_from": start_time,
                "datetime_to": end_time,
                "limit": 1000,
                "page": page,
            }
            response = requests.get(url, headers=HEADERS, params=params).json()

            results = response.get("results", [])
            if not results:
                break
            
            for res in results:
                timestamp_data = {
                    "value": res["value"],
                    "timestamp": res["period"]["datetimeFrom"]["utc"]
                }
                measurement_data.append(timestamp_data)

            page += 1
            
        df = pd.DataFrame(measurement_data)
        df = df.set_index("timestamp").reindex(timestamp_index)
        
        return df

    df_list = []
    for start_time, end_time in year_interval(datetime_from, datetime_to):
        df_list.append(get_one_year_sensor_measurements(sensor_id, start_time, end_time))
    return pd.concat(df_list, axis=0)


def get_aqi_data(location_id, datetime_from, datetime_to):
    df_list = []
    
    filepath = f'data/location_{location_id}/info.json'
    with open(filepath, "r") as file:
        sensor_data = json.load(file)[0]["sensors"]
    
    sensor_ids = {sensor["parameter"]["name"]: sensor["id"] for sensor in sensor_data}
    pollutants = sorted(sensor_ids.keys())
    for pollutant in pollutants:
        print(f"Fetching {pollutant} info...")
        df = get_sensor_measurements(sensor_ids[pollutant], datetime_from, datetime_to)
        df_list.append(df.rename(columns={"value": pollutant}))
        
    final_df = pd.concat(df_list, axis=1)
    final_df.to_csv(f"data/location_{location_id}/aqi.csv")
    
        
if __name__ == "__main__":
    pass
    get_aqi_data(7740, "2019-01-01T00:00:00Z", "2025-12-08T00:00:00Z")