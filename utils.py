import os
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


def pm25_to_vn_aqi(pm25_value):
    """
    Convert PM2.5 concentration (μg/m³) to Vietnamese AQI category.
    
    Vietnamese AQI Standards for PM2.5:
    - Good: 0-25 μg/m³
    - Moderate: 26-50 μg/m³
    - Unhealthy for Sensitive Groups: 51-90 μg/m³
    - Unhealthy: 91-150 μg/m³
    - Very Unhealthy: 151-250 μg/m³
    - Hazardous: >250 μg/m³
    """
    if pm25_value <= 25:
        category = "Good"
        aqi = int((50 / 25) * pm25_value)
        color = "#00E400"  # Green
        health = "Air quality is good. Ideal for outdoor activities."
        icon = "😊"
    elif pm25_value <= 50:
        category = "Moderate"
        aqi = int(50 + ((50 / 25) * (pm25_value - 25)))
        color = "#FFFF00"  # Yellow
        health = "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion."
        icon = "😐"
    elif pm25_value <= 90:
        category = "Unhealthy for Sensitive Groups"
        aqi = int(100 + ((50 / 40) * (pm25_value - 50)))
        color = "#FF7E00"  # Orange
        health = "Sensitive groups (children, elderly, people with respiratory conditions) should reduce prolonged outdoor activities."
        icon = "😷"
    elif pm25_value <= 150:
        category = "Unhealthy"
        aqi = int(150 + ((50 / 60) * (pm25_value - 90)))
        color = "#FF0000"  # Red
        health = "Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activities."
        icon = "⚠️"
    elif pm25_value <= 250:
        category = "Very Unhealthy"
        aqi = int(200 + ((100 / 100) * (pm25_value - 150)))
        color = "#8F3F97"  # Purple
        health = "Health alert! Everyone should avoid prolonged outdoor activities. Wear masks if going outside."
        icon = "🚨"
    else:
        category = "Hazardous"
        aqi = int(300 + ((200 / 250) * min(pm25_value - 250, 250)))
        color = "#7E0023"  # Maroon
        health = "Health emergency! Everyone should avoid all outdoor activities. Stay indoors with air purifiers."
        icon = "☠️"
    
    return {
        "pm25": round(pm25_value, 2),
        "aqi": min(aqi, 500),
        "category": category,
        "color": color,
        "health_advisory": health,
        "icon": icon
    }


def get_health_recommendations(aqi_category):
    """Get detailed health recommendations based on AQI category."""
    recommendations = {
        "Good": {
            "general": "Perfect day for outdoor activities!",
            "sensitive": "No restrictions for sensitive groups.",
            "activities": "✅ Running, cycling, outdoor sports\n✅ Open windows for ventilation\n✅ All outdoor activities recommended"
        },
        "Moderate": {
            "general": "Air quality is acceptable for most people.",
            "sensitive": "Unusually sensitive individuals may experience minor symptoms.",
            "activities": "✅ Most outdoor activities OK\n⚠️ Sensitive groups: limit prolonged exertion\n✅ Normal outdoor activities for most"
        },
        "Unhealthy for Sensitive Groups": {
            "general": "General public can enjoy outdoor activities.",
            "sensitive": "Children, elderly, and people with heart/lung disease should reduce outdoor activities.",
            "activities": "⚠️ Sensitive groups: limit outdoor time\n⚠️ Consider moving strenuous activities indoors\n✅ General public: outdoor activities OK with awareness"
        },
        "Unhealthy": {
            "general": "Everyone may experience health effects.",
            "sensitive": "Sensitive groups may experience more serious effects. Avoid outdoor activities.",
            "activities": "❌ Sensitive groups: stay indoors\n⚠️ General public: reduce prolonged outdoor exertion\n🏠 Consider indoor alternatives for exercise\n😷 Wear N95 masks if going outside"
        },
        "Very Unhealthy": {
            "general": "Health alert! Everyone should limit outdoor activities.",
            "sensitive": "Sensitive groups should stay indoors and keep activity levels low.",
            "activities": "❌ Avoid all prolonged outdoor activities\n🏠 Stay indoors with windows closed\n😷 Wear N95/KF94 masks if must go outside\n💨 Use air purifiers indoors\n⚠️ Cancel outdoor events"
        },
        "Hazardous": {
            "general": "Health emergency! Everyone should avoid outdoor activities.",
            "sensitive": "Everyone should stay indoors and avoid all physical activities outdoors.",
            "activities": "🚨 STAY INDOORS - Health Emergency!\n❌ Cancel all outdoor activities\n😷 Wear N95 masks even for brief outdoor exposure\n💨 Use air purifiers at maximum setting\n🪟 Seal windows and doors\n🏥 Seek medical attention if experiencing symptoms"
        }
    }
    return recommendations.get(aqi_category, recommendations["Good"])


if __name__ == "__main__":
    pass
    # with open("data/info.json", "r") as file:
    #     info = json.load(file)
        
    # get_aqi_data(**info)
    # get_weather_data(**info)