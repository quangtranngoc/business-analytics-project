import os
import json
from dotenv import load_dotenv
import requests
import pandas as pd

load_dotenv()
HEADERS = {
    "X-API-Key": os.getenv("API_KEY")
}

def get_location_info(location_id):
    url = f"https://api.openaq.org/v3/locations/{location_id}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        result = response.json().get("results", [])
        os.makedirs(f'data/location_{location_id}', exist_ok=True)
        
        with open(f'data/location_{location_id}/info.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))


def get_sensor_measurements(sensor_id):
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
    datetime_from = "2021-01-01T00:00:00Z"
    datetime_to = "2025-12-07T00:00:00Z"
    measurement_data = []
    
    timestamp_index = pd.date_range(start=datetime_from, end=datetime_to, freq='h').strftime("%Y-%m-%dT%H:%M:%SZ")[:-1]
    
    page = 1
    while True:
        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
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


def get_aqi_data(location_id):
    df_list = []
    
    filepath = f'data/location_{location_id}/info.json'
    with open(filepath, "r") as file:
        sensor_data = json.load(file)[0]["sensors"]
    
    sensor_ids = {sensor["parameter"]["name"]: sensor["id"] for sensor in sensor_data}
    pollutants = sorted(sensor_ids.keys())
    for pollutant in pollutants:
        df = get_sensor_measurements(sensor_ids[pollutant])
        df_list.append(df.rename(columns={"value": pollutant}))
        
    final_df = pd.concat(df_list, axis=1)
    final_df.to_csv(f"data/location_{location_id}/aqi2.csv")
    
        
if __name__ == "__main__":
    get_location_info(7740)
    get_aqi_data(7740)