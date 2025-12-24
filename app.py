import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta
import json
import os
from utils import pm25_to_vn_aqi, get_health_recommendations, get_latest_realtime_data

# Page configuration
st.set_page_config(
    page_title="Hanoi Air Quality Nowcasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load location info
@st.cache_data
def load_location_info():
    with open("data/info.json", "r") as f:
        return json.load(f)

# Load data
@st.cache_data
def load_cleaned_data():
    air_df = pd.read_csv("data/processed/cleaned/cleaned_air.csv", index_col=0, parse_dates=True)
    weather_df = pd.read_csv("data/processed/cleaned/cleaned_weather.csv", index_col=0, parse_dates=True)
    return air_df, weather_df

# Load real-time data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_realtime_data(lat, lon):
    """Load latest real-time data."""
    try:
        air_rt, weather_rt, last_update = get_latest_realtime_data(lat, lon)
        return air_rt, weather_rt, last_update, None
    except Exception as e:
        return None, None, None, str(e)

# Combine training and real-time data in memory
def get_combined_data(training_air, training_weather, realtime_air, realtime_weather):
    """Combine historical training data with real-time data."""
    if realtime_air is not None and not realtime_air.empty:
        combined_air = pd.concat([training_air, realtime_air]).sort_index()
        combined_air = combined_air[~combined_air.index.duplicated(keep='last')]
    else:
        combined_air = training_air
    
    if realtime_weather is not None and not realtime_weather.empty:
        combined_weather = pd.concat([training_weather, realtime_weather]).sort_index()
        combined_weather = combined_weather[~combined_weather.index.duplicated(keep='last')]
    else:
        combined_weather = training_weather
    
    return combined_air, combined_weather

# Load trained models
@st.cache_resource
def load_models():
    models = {}
    
    # Load ETS model
    if os.path.exists("models/ets/pm2_5.pickle"):
        with open("models/ets/pm2_5.pickle", "rb") as f:
            models["ETS"] = pickle.load(f)
    
    # Load ARIMA model
    if os.path.exists("models/arima/pm2_5.pickle"):
        with open("models/arima/pm2_5.pickle", "rb") as f:
            models["ARIMA"] = pickle.load(f)
    
    # Load ARIMAX model
    if os.path.exists("models/arimax/pm2_5.pickle"):
        with open("models/arimax/pm2_5.pickle", "rb") as f:
            models["ARIMAX"] = pickle.load(f)
    
    # Load VAR model (Air + Weather) - best for ozone
    if os.path.exists("models/var/var_air_weather.pickle"):
        with open("models/var/var_air_weather.pickle", "rb") as f:
            models["VAR (Air+Weather)"] = pickle.load(f)
    
    # Load VAR model (Air Only) - best for pm10, pm2_5
    if os.path.exists("models/var/var_air_only.pickle"):
        with open("models/var/var_air_only.pickle", "rb") as f:
            models["VAR (Air-Only)"] = pickle.load(f)
    
    return models

# Refit ETS model with recent data
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def refit_ets_model(training_air, lat, lon):
    """Refit ETS model with recent data from API + real-time data."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from utils import get_aqi_data, get_latest_realtime_data
    
    # Get date range for fetching recent data
    training_end = training_air.index[-1]
    now = datetime.now()
    
    try:
        # Fetch historical recent data
        recent_data = get_aqi_data(
            lat, lon,
            training_end.strftime("%Y-%m-%dT%H:%M:%S"),
            now.strftime("%Y-%m-%dT%H:%M:%S")
        )
        recent_data.index = pd.to_datetime(recent_data.index)
        
        # Fetch real-time data
        realtime_air, _, _ = get_latest_realtime_data(lat, lon)
        
        # Combine all three: training + recent archive + real-time
        combined = pd.concat([training_air, recent_data, realtime_air]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # Retrain ETS model with same configuration as pretrained
        pm25_series = combined['pm2_5'].dropna()
        model = ETSModel(
            pm25_series,
            error='add', 
            trend=None, 
            seasonal='mul',
            seasonal_periods=24
        )
        fitted_model = model.fit(disp=False)
        
        return fitted_model, combined, None
        
    except Exception as e:
        return None, training_air, str(e)

# Refit ARIMA model with recent data
@st.cache_resource(ttl=3600)
def refit_arima_model(training_air, lat, lon, order=(2, 0, 1)):
    """Refit ARIMA model at forecast origin with recent data.
    
    Uses trend='c' for forecast continuity (avoids mean-reversion issues).
    """
    from statsmodels.tsa.arima.model import ARIMA
    from utils import get_aqi_data, get_latest_realtime_data
    import warnings
    
    training_end = training_air.index[-1]
    now = datetime.now()
    
    try:
        # Fetch recent data
        recent_data = get_aqi_data(
            lat, lon,
            training_end.strftime("%Y-%m-%dT%H:%M:%S"),
            now.strftime("%Y-%m-%dT%H:%M:%S")
        )
        recent_data.index = pd.to_datetime(recent_data.index)
        
        # Fetch real-time data
        realtime_air, _, _ = get_latest_realtime_data(lat, lon)
        
        # Combine all data
        combined = pd.concat([training_air, recent_data, realtime_air]).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        
        # Prepare series
        pm25_series = combined['pm2_5'].dropna()
        pm25_series.index.freq = 'h'
        
        # Retrain ARIMA with trend='c' for forecast continuity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(pm25_series, order=order, trend='c')
            fitted_model = model.fit()
        
        return fitted_model, combined, None
        
    except Exception as e:
        return None, training_air, str(e)

# Refit ARIMAX model with recent data
@st.cache_resource(ttl=3600)
def refit_arimax_model(training_air, training_weather, lat, lon, order=(2, 0, 1)):
    """Refit ARIMAX model at forecast origin with recent data.
    Uses trend='c' for forecast continuity (avoids mean-reversion issues).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from utils import get_aqi_data, get_weather_data, get_latest_realtime_data
    import warnings
    
    training_end = training_air.index[-1]
    now = datetime.now()
    
    try:
        # Fetch recent air data
        recent_air = get_aqi_data(
            lat, lon,
            training_end.strftime("%Y-%m-%dT%H:%M:%S"),
            now.strftime("%Y-%m-%dT%H:%M:%S")
        )
        recent_air.index = pd.to_datetime(recent_air.index)
        
        # Fetch recent weather data
        recent_weather = get_weather_data(
            lat, lon,
            training_end.strftime("%Y-%m-%dT%H:%M:%S"),
            now.strftime("%Y-%m-%dT%H:%M:%S")
        )
        recent_weather.index = pd.to_datetime(recent_weather.index)
        
        # Fetch real-time data
        realtime_air, realtime_weather, _ = get_latest_realtime_data(lat, lon)
        
        # Combine air data
        combined_air = pd.concat([training_air, recent_air, realtime_air]).sort_index()
        combined_air = combined_air[~combined_air.index.duplicated(keep='last')]
        
        # Combine weather data
        combined_weather = pd.concat([training_weather, recent_weather, realtime_weather]).sort_index()
        combined_weather = combined_weather[~combined_weather.index.duplicated(keep='last')]
        
        # Align and prepare data
        pm25_series = combined_air['pm2_5'].dropna()
        pm25_series.index.freq = 'h'
        
        # Align exog with endog
        exog = combined_weather.reindex(pm25_series.index).ffill().bfill()
        exog = exog[training_weather.columns]  # Ensure column order
        
        # Retrain SARIMAX with trend='c' for forecast continuity
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(pm25_series, exog=exog, order=order, trend='c')
            fitted_model = model.fit(disp=False)
        
        return fitted_model, combined_air, None
        
    except Exception as e:
        return None, training_air, str(e)


# Generate ETS forecast
@st.cache_data(ttl=300)
def generate_ets_forecast(_model, steps=6):
    """Generate forecast using the ETS model."""
    forecast = _model.forecast(steps=steps)
    forecast_df = _model.get_prediction(
        start=len(_model.data.orig_endog),
        end=len(_model.data.orig_endog) + steps - 1
    ).summary_frame()
    
    return forecast, forecast_df

# Generate ARIMA forecast
@st.cache_data(ttl=300)
def generate_arima_forecast(_model, steps=6):
    """Generate forecast using the ARIMA model."""
    forecast_result = _model.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean
    forecast_df = forecast_result.summary_frame()

    forecast_df = forecast_df.rename(columns={
        'mean': 'mean',
        'mean_ci_lower': 'mean_ci_lower',
        'mean_ci_upper': 'mean_ci_upper'
    })
    
    return forecast, forecast_df

# Generate ARIMAX forecast
@st.cache_data(ttl=300)
def generate_arimax_forecast(_model, exog, steps=6):
    """Generate forecast using the ARIMAX model with exogenous variables."""
    forecast_result = _model.get_forecast(steps=steps, exog=exog)
    forecast = forecast_result.predicted_mean
    forecast_df = forecast_result.summary_frame()
    
    return forecast, forecast_df


# Generate VAR forecast
@st.cache_data(ttl=300)
def generate_var_forecast(_model, last_obs, steps=6):
    """Generate forecast using the VAR model (multivariate)."""

    forecast = _model.forecast(y=last_obs, steps=steps)
    col_names = _model.names
    pm25_idx = col_names.index('pm2_5') if 'pm2_5' in col_names else 2
    pm25_forecast = forecast[:, pm25_idx]
    
    forecast_df = pd.DataFrame({
        'mean': pm25_forecast,
    })
    
    return pm25_forecast, forecast_df


# Create visualization
def create_forecast_plot(historical_data, forecast_data, forecast_hours):
    """Create interactive plot with historical and forecasted PM2.5."""
    
    # Get last 48 hours of historical data for context
    hist_slice = historical_data.tail(48)
    
    # Create forecast timestamps
    last_timestamp = historical_data.index[-1]
    forecast_timestamps = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=forecast_hours,
        freq='h'
    )
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_slice.index,
        y=hist_slice['pm2_5'],
        name='Historical PM2.5',
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Time:</b> %{x}<br><b>PM2.5:</b> %{y:.2f} μg/m³<extra></extra>'
    ))
    
    # Forecasted data
    fig.add_trace(go.Scatter(
        x=forecast_timestamps,
        y=forecast_data['mean'][:forecast_hours],
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8),
        hovertemplate='<b>Time:</b> %{x}<br><b>Forecast:</b> %{y:.2f} μg/m³<extra></extra>'
    ))
    
    # Confidence interval
    ci_lower_col = 'mean_ci_lower' if 'mean_ci_lower' in forecast_data.columns else 'pi_lower'
    ci_upper_col = 'mean_ci_upper' if 'mean_ci_upper' in forecast_data.columns else 'pi_upper'
    
    if ci_lower_col in forecast_data.columns and ci_upper_col in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=list(forecast_timestamps) + list(forecast_timestamps[::-1]),
            y=list(forecast_data[ci_lower_col][:forecast_hours]) + list(forecast_data[ci_upper_col][:forecast_hours][::-1]),
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Add AQI threshold lines
    fig.add_hline(y=25, line_dash="dot", line_color="green", 
                  annotation_text="Good/Moderate", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color="yellow", 
                  annotation_text="Moderate/USG", annotation_position="right")
    fig.add_hline(y=90, line_dash="dot", line_color="orange", 
                  annotation_text="USG/Unhealthy", annotation_position="right")
    fig.add_hline(y=150, line_dash="dot", line_color="red", 
                  annotation_text="Unhealthy/Very Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text='<b>PM2.5 Nowcast: Historical + Forecast</b>',
            font=dict(size=20)
        ),
        xaxis_title='Time',
        yaxis_title='PM2.5 Concentration (μg/m³)',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_map_visualization(lat, lon, aqi_info):
    """Create a map with the sensor location colored by AQI."""
    fig = go.Figure(go.Scattermap(
        lat=[lat],
        lon=[lon],
        mode='markers+text',
        marker=dict(
            size=30,
            color=aqi_info['color'],
            opacity=0.8
        ),
        text=[f"HUST Station<br>{aqi_info['category']}"],
        textposition="top center",
        textfont=dict(size=12, color='black'),
        hovertemplate="<b>HUST Air Quality Station</b><br>" +
                     f"PM2.5: {aqi_info['pm25']:.1f} μg/m³<br>" +
                     f"AQI: {aqi_info['aqi']}<br>" +
                     f"Category: {aqi_info['category']}<extra></extra>"
    ))
    
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=11
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">Hanoi Air Quality Nowcasting & Health Advisory</div>', 
                unsafe_allow_html=True)
    st.markdown("**Hanoi University of Science and Technology (HUST) Station**")
    st.markdown("---")
    
    # Load data
    location_info = load_location_info()
    training_air, training_weather = load_cleaned_data()
    
    # Load real-time data
    realtime_air, realtime_weather, rt_last_update, rt_error = load_realtime_data(
        location_info['lat'], 
        location_info['lon']
    )
    
    # Combine training + real-time data in memory only
    air_df, weather_df = get_combined_data(training_air, training_weather, realtime_air, realtime_weather)
    
    models = load_models()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    available_models = list(models.keys())
    if not available_models:
        st.error("No models available. Please train models first.")
        return
    
    selected_model_name = st.sidebar.selectbox(
        "Select Forecasting Model",
        available_models,
        help="ETS: Exponential Smoothing (univariate)\nARIMA: AutoRegressive Integrated Moving Average\nARIMAX: ARIMA with weather features"
    )
    
    # Forecast horizon
    forecast_hours = st.sidebar.slider(
        "Forecast Horizon (hours)",
        min_value=1,
        max_value=6,
        value=6,
        help="Select how many hours ahead to forecast"
    )
    
    # Current conditions
    st.sidebar.markdown("---")
    st.sidebar.header("Current Conditions")
    current_pm25 = air_df['pm2_5'].iloc[-1]
    current_temp = weather_df['temperature_2m'].iloc[-1]
    current_humidity = weather_df['relative_humidity_2m'].iloc[-1]
    current_wind = weather_df['wind_speed_10m'].iloc[-1]
    
    st.sidebar.metric("PM2.5", f"{current_pm25:.1f} μg/m³")
    st.sidebar.metric("Temperature", f"{current_temp:.1f} °C")
    st.sidebar.metric("Humidity", f"{current_humidity:.0f}%")
    st.sidebar.metric("Wind Speed", f"{current_wind:.1f} m/s")
    
    # Last updated
    last_update = air_df.index[-1]
    update_source = "Real-time" if rt_last_update and rt_last_update == last_update else "Training data"
    st.sidebar.markdown(f"**Last Updated:** {last_update.strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.caption(f"Source: {update_source}")

    # Refresh data button
    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Latest Data", help="Fetch the most recent air quality and weather data (won't modify training data)"):
        # Clear cache to force reload
        st.cache_data.clear()
        st.sidebar.success("Refreshing...")
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("PM2.5 Forecast")
        
        forecast_data = air_df
        
        # Handle different model types
        if selected_model_name == "ETS":
            # Refit ETS model with recent data
            with st.spinner("Refitting model with recent data..."):
                refitted_model, updated_air_df, refit_error = refit_ets_model(
                    training_air, 
                    location_info['lat'], 
                    location_info['lon']
                )
            
            if refit_error:
                st.warning(f"Could not refit model: {refit_error}. Using pre-trained model.")
                selected_model = models[selected_model_name]
            else:
                selected_model = refitted_model
                forecast_data = updated_air_df
                st.caption(f"Model refitted with data up to {updated_air_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
            
            with st.spinner("Generating forecast..."):
                forecast_values, forecast_df = generate_ets_forecast(selected_model, steps=forecast_hours)
        
        elif selected_model_name == "ARIMA":
            # Refit ARIMA model at forecast origin with trend='c'
            with st.spinner("Refitting ARIMA model with recent data..."):
                refitted_model, updated_air_df, refit_error = refit_arima_model(
                    training_air,
                    location_info['lat'],
                    location_info['lon']
                )
            
            if refit_error:
                st.warning(f"Could not refit model: {refit_error}. Using pre-trained model.")
                selected_model = models[selected_model_name]
            else:
                selected_model = refitted_model
                forecast_data = updated_air_df
                st.caption(f"Model refitted with data up to {updated_air_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
            
            with st.spinner("Generating forecast..."):
                forecast_values, forecast_df = generate_arima_forecast(selected_model, steps=forecast_hours)
        
        elif selected_model_name == "ARIMAX":
            from utils import get_weather_forecast
            
            # Refit ARIMAX model at forecast origin with trend='c'
            with st.spinner("Refitting ARIMAX model with recent data..."):
                refitted_model, updated_air_df, refit_error = refit_arimax_model(
                    training_air,
                    training_weather,
                    location_info['lat'],
                    location_info['lon']
                )
            
            if refit_error:
                st.warning(f"Could not refit model: {refit_error}. Using pre-trained model.")
                selected_model = models[selected_model_name]
            else:
                selected_model = refitted_model
                forecast_data = updated_air_df
                st.caption(f"Model refitted with data up to {updated_air_df.index[-1].strftime('%Y-%m-%d %H:%M')}")
            
            # Get weather forecast for exog
            with st.spinner("Fetching weather forecast..."):
                try:
                    weather_forecast = get_weather_forecast(
                        location_info['lat'], 
                        location_info['lon'], 
                        hours=forecast_hours
                    )
                    
                    if weather_forecast.empty or len(weather_forecast) < forecast_hours:
                        st.warning("Insufficient weather forecast. Falling back to ARIMA.")
                        forecast_values, forecast_df = generate_arima_forecast(
                            models.get("ARIMA", selected_model), steps=forecast_hours
                        )
                    else:
                        # Ensure column order matches training data
                        weather_forecast = weather_forecast[training_weather.columns]
                        with st.spinner("Generating forecast..."):
                            forecast_values, forecast_df = generate_arimax_forecast(
                                selected_model, exog=weather_forecast.values, steps=forecast_hours
                            )
                except Exception as e:
                    st.warning(f"Weather forecast error: {e}. Falling back to ARIMA.")
                    forecast_values, forecast_df = generate_arima_forecast(
                        models.get("ARIMA", selected_model), steps=forecast_hours
                    )
        
        elif selected_model_name == "VAR (Air+Weather)":
            selected_model = models[selected_model_name]
            st.caption("Using VAR model (Air + Weather) - Best for ozone forecasting")
            
            # Use combined_df (Training + Real-time) for forecasting
            combined_df = pd.concat([air_df, weather_df], axis=1)
            
            # VAR needs lagged observations
            if combined_df.isnull().any().any():
                combined_df = combined_df.ffill().bfill()
            
            forecast_data = combined_df
            
            with st.spinner("Generating forecast..."):
                lag_order = selected_model.k_ar
                
                if len(combined_df) < lag_order:
                    st.error(f"Not enough data for VAR forecast. Need at least {lag_order} hours.")
                else:
                    last_obs = combined_df.values[-lag_order:]
                    forecast_values, forecast_df = generate_var_forecast(
                        selected_model, last_obs=last_obs, steps=forecast_hours
                    )
        
        elif selected_model_name == "VAR (Air-Only)":
            selected_model = models[selected_model_name]
            st.caption("Using VAR model (Air Only) - Best for PM10, PM2.5 forecasting")
            
            # Use air_df only for forecasting
            air_only_df = air_df.copy()
            
            if air_only_df.isnull().any().any():
                air_only_df = air_only_df.ffill().bfill()
            
            forecast_data = air_only_df
            
            with st.spinner("Generating forecast..."):
                lag_order = selected_model.k_ar
                
                if len(air_only_df) < lag_order:
                    st.error(f"Not enough data for VAR forecast. Need at least {lag_order} hours.")
                else:
                    last_obs = air_only_df.values[-lag_order:]
                    forecast_values, forecast_df = generate_var_forecast(
                        selected_model, last_obs=last_obs, steps=forecast_hours
                    )
        
        else:
            st.error(f"Unknown model type: {selected_model_name}")
            return
        
        # Create and display plot
        fig = create_forecast_plot(forecast_data, forecast_df, forecast_hours)
        st.plotly_chart(fig, width='stretch')
        
        # Forecast table
        with st.expander("View Detailed Forecast Table"):
            last_timestamp = forecast_data.index[-1]
            forecast_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=forecast_hours,
                freq='h'
            )
            
            # Handle different confidence interval column names
            ci_lower_col = 'mean_ci_lower' if 'mean_ci_lower' in forecast_df.columns else 'pi_lower'
            ci_upper_col = 'mean_ci_upper' if 'mean_ci_upper' in forecast_df.columns else 'pi_upper'
            
            forecast_table_data = {
                'Time': forecast_timestamps.strftime('%Y-%m-%d %H:%M'),
                'PM2.5 (μg/m³)': forecast_df['mean'][:forecast_hours].round(2),
            }
            
            if ci_lower_col in forecast_df.columns:
                forecast_table_data['Lower Bound'] = forecast_df[ci_lower_col][:forecast_hours].round(2)
            if ci_upper_col in forecast_df.columns:
                forecast_table_data['Upper Bound'] = forecast_df[ci_upper_col][:forecast_hours].round(2)
            
            forecast_table = pd.DataFrame(forecast_table_data)
            
            # Add AQI categories
            forecast_table['AQI Category'] = forecast_table['PM2.5 (μg/m³)'].apply(
                lambda x: pm25_to_vn_aqi(x)['category']
            )
            
            st.dataframe(forecast_table, width='stretch')
            
            # Download button
            csv = forecast_table.to_csv(index=False)
            st.download_button(
                label="Download Forecast as CSV",
                data=csv,
                file_name=f"hanoi_pm25_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("Air Quality Index")
        
        # Current AQI
        current_aqi_info = pm25_to_vn_aqi(current_pm25)
        
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; border: 2px solid {current_aqi_info['color']};">
            <h2 style="color: {current_aqi_info['color']};">{current_aqi_info['icon']} {current_aqi_info['category']}</h2>
            <h1 style="color: {current_aqi_info['color']};">AQI: {current_aqi_info['aqi']}</h1>
            <h3>PM2.5: {current_aqi_info['pm25']:.1f} μg/m³</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Forecast AQI for next hour
        next_hour_pm25 = forecast_df['mean'].iloc[0]
        next_hour_aqi_info = pm25_to_vn_aqi(next_hour_pm25)
        
        st.markdown("### Next Hour Forecast")
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; border: 2px solid {next_hour_aqi_info['color']};">
            <h3 style="color: {next_hour_aqi_info['color']};">{next_hour_aqi_info['icon']} {next_hour_aqi_info['category']}</h3>
            <h2 style="color: {next_hour_aqi_info['color']};">AQI: {next_hour_aqi_info['aqi']}</h2>
            <p>PM2.5: {next_hour_aqi_info['pm25']:.1f} μg/m³</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health advisory
        st.markdown("### Health Advisory")
        health_rec = get_health_recommendations(current_aqi_info['category'])
        
        st.info(f"**General Population:**\n{health_rec['general']}")
        st.warning(f"**Sensitive Groups:**\n{health_rec['sensitive']}")
        
        with st.expander("Recommended Actions"):
            st.markdown(health_rec['activities'])
        
        # Alert system
        st.markdown("### Alerts")
        
        # Check if any forecast exceeds unhealthy threshold
        unhealthy_forecasts = forecast_df['mean'][:forecast_hours][forecast_df['mean'][:forecast_hours] > 90]
        
        if len(unhealthy_forecasts) > 0:
            first_unhealthy_hour = unhealthy_forecasts.index[0] + 1
            max_pm25 = unhealthy_forecasts.max()
            max_aqi = pm25_to_vn_aqi(max_pm25)
            
            st.error(f"""
            ⚠️ **Air Quality Alert!**
            
            Unhealthy conditions expected in {first_unhealthy_hour} hour(s).
            
            Peak forecast: {max_pm25:.1f} μg/m³ ({max_aqi['category']})
            
            **Recommendation:** Plan indoor activities.
            """)
        elif next_hour_pm25 > 50:
            st.warning("""
            ⚠️ **Advisory Notice**
            
            Air quality may affect sensitive groups.
            
            Monitor conditions if you're in a sensitive group.
            """)
        else:
            st.success("✅ No air quality alerts. Conditions are favorable!")
    
    # Map section
    st.markdown("---")
    st.subheader("Hanoi Air Quality Map")
    
    map_fig = create_map_visualization(
        location_info['lat'],
        location_info['lon'],
        current_aqi_info
    )
    st.plotly_chart(map_fig, width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><b>Data Sources:</b> Open-Meteo Air Quality API | <b>Model:</b> {model_name} | <b>Update Frequency:</b> Hourly</p>
        <p>Business Analytics Project - Air Quality Nowcasting & Health Advisory System</p>
        <p>This is a forecasting system. For official air quality information, consult local authorities.</p>
    </div>
    """.format(model_name=selected_model_name), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
