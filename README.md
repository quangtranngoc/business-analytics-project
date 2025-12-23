# Business Analytics Project - Air Quality Nowcasting

## 🌫️ Hanoi Air Quality Nowcasting & Health Advisory

PM2.5 nowcasting web app for Hanoi with 1-6 hour forecasts and Vietnamese AQI health advisories.

## 🚀 Quick Start

```bash
# Run the dashboard
streamlit run app.py

# Or
python -m streamlit run app.py
```

Open browser to **http://localhost:8501**

## ✅ Features

- PM2.5 nowcasting (1-6 hours ahead) using ETS time series model
- Vietnamese AQI standards with color-coded categories
- Health advisories for general public & sensitive groups
- Interactive forecast charts with confidence intervals
- Automatic alerts for unhealthy conditions
- Hanoi map with HUST monitoring station
- CSV export and data refresh capabilities

## 📊 Vietnamese AQI Standards

| PM2.5 (μg/m³) | Category | Color |
|---------------|----------|-------|
| 0-25 | Good | 🟢 Green |
| 26-50 | Moderate | 🟡 Yellow |
| 51-90 | Unhealthy for Sensitive Groups | 🟠 Orange |
| 91-150 | Unhealthy | 🔴 Red |
| 151-250 | Very Unhealthy | 🟣 Purple |
| 250+ | Hazardous | 🟤 Maroon |

## 🎯 Business Analytics Topic 9

**Goal:** Nowcast PM2.5 and issue exposure advisories for Hanoi  
**Methods:** Time-series nowcasting with meteorological features  
**Deliverables:** Dashboard with alerts, AQI mapping, and health guidance

## 📁 Project Structure

```
├── app.py                    # Streamlit dashboard
├── utils.py                  # AQI conversion & API functions
├── test_aqi.py              # AQI system tests
├── data_cleaning.ipynb       # Data preprocessing
├── data_exploration.ipynb    # EDA and analysis
├── model.ipynb              # Model training
├── data/                    # Data files
│   ├── aqi.csv              # Air quality data
│   ├── weather.csv          # Weather data
│   └── processed/cleaned/   # Cleaned datasets
└── models/ets/              # Trained ETS model
```

## 🧪 Testing

```bash
# Test AQI conversion system
python test_aqi.py
```

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Main packages: streamlit, pandas, plotly, statsmodels, requests

## 🔧 Models

- **ETS (Exponential Smoothing)**: ✅ Active (univariate time series)
- **ARIMA**: 🔜 Placeholder (to be trained)
- **ARIMAX**: 🔜 Placeholder (with weather features)

---

**Hanoi University of Science and Technology (HUST) - December 2025**