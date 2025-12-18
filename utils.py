import os
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
import requests
import pandas as pd

load_dotenv()
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")


def datetime_str_to_unix(datetime_str, str_format):
    datetime_obj = datetime.strptime(datetime_str, str_format).replace(tzinfo=timezone.utc)
    return int(datetime_obj.timestamp())


def unix_to_datetime_str(unix_time, str_format):
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime(str_format)


def timestamp_index_list(datetime_from, datetime_to, str_format):
    return pd.date_range(start=datetime_from, end=datetime_to, freq='h').strftime(str_format)


def year_interval(datetime_from, datetime_to):
    start_year = int(datetime_from[:4])
    end_year = int(datetime_to[:4])
    intervals = []
    
    if start_year == end_year:
        return [(datetime_from, datetime_to)]
    
    intervals.append((datetime_from, f"{start_year}-12-31T23:00:00"))
    for year in range(start_year + 1, end_year):
        intervals.append((f"{year}-01-01T00:00:00", f"{year}-12-31T23:00:00"))
    intervals.append((f"{end_year}-01-01T00:00:00", datetime_to))

    return intervals


# def get_aqi_data(lat, lon, datetime_from, datetime_to, str_format="%Y-%m-%dT%H:%M:%S"):
#     df_list = []
    
#     for dt_from, dt_to in year_interval(datetime_from, datetime_to):
#         measurement_data = []
#         unix_from = datetime_str_to_unix(dt_from, str_format)
#         unix_to = datetime_str_to_unix(dt_to, str_format)
#         timestamp_index = timestamp_index_list(dt_from, dt_to, str_format)
        
#         url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
#         params = {
#             "lat": lat,
#             "lon": lon,
#             "start": unix_from,
#             "end": unix_to,
#             "appid": API_KEY
#         }
#         response = requests.get(url, params=params).json()
#         results = response.get("list", [])
        
#         for res in results:
#             timestamp = unix_to_datetime_str(res["dt"], str_format)
#             values = res["components"]
#             measurement_data.append({"timestamp": timestamp} | values)
        
#         df = pd.DataFrame(measurement_data)
#         df.drop(["no", "nh3"], axis=1, inplace=True)
#         df = df.set_index("timestamp").reindex(timestamp_index)

#         df_list.append(df)
    
#     final_df = pd.concat(df_list, axis=0)    
#     final_df.to_csv("data/aqi.csv")


def get_aqi_data(lat, lon, datetime_from, datetime_to, str_format="%Y-%m-%dT%H:%M:%S"):
    datetime_from = datetime_from[:10]
    datetime_to = datetime_to[:10]
    timestamp_index = timestamp_index_list(datetime_from, datetime_to, str_format)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": datetime_from,
        "end_date": datetime_to,
        "hourly": ["carbon_monoxide", "pm10", "pm2_5", "nitrogen_dioxide", "ozone", "sulphur_dioxide"],
        "timeformat": "unixtime"
    }
    
    response = requests.get(url, params=params).json()
    results = response.get("hourly", [])
    
    df = pd.DataFrame(results)
    df["timestamp"] = df["time"].apply(lambda unix_time: unix_to_datetime_str(unix_time, str_format))
    df.drop("time", axis=1, inplace=True)
    df = df.set_index("timestamp").reindex(timestamp_index)
    
    df.to_csv("data/aqi2.csv")
    
    
def get_weather_data(lat, lon, datetime_from, datetime_to, str_format="%Y-%m-%dT%H:%M:%S"):
    datetime_from = datetime_from[:10]
    datetime_to = datetime_to[:10]
    timestamp_index = timestamp_index_list(datetime_from, datetime_to, str_format)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": datetime_from,
        "end_date": datetime_to,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
                   "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
        "timeformat": "unixtime"
    }
    
    response = requests.get(url, params=params).json()
    results = response.get("hourly", [])
    
    df = pd.DataFrame(results)
    df["timestamp"] = df["time"].apply(lambda unix_time: unix_to_datetime_str(unix_time, str_format))
    df.drop("time", axis=1, inplace=True)
    df = df.set_index("timestamp").reindex(timestamp_index)
    
    df.to_csv("data/weather.csv")


if __name__ == "__main__":
    pass
    # with open("data/info.json", "r") as file:
    #     info = json.load(file)
        
    # get_aqi_data(**info)
    # get_weather_data(**info)