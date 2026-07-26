import streamlit as st
import pandas as pd
import requests
import sqlite3
import numpy as np
import pydeck as pdk
from pathlib import Path
from datetime import date, timedelta, datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from geopy.geocoders import Nominatim
from zoneinfo import ZoneInfo

# =====================================================
# CONFIGURATION
# =====================================================

DOWNLOADS_FOLDER = Path.cwd()
DB_PATH = DOWNLOADS_FOLDER / "temperature_data.db"

st.set_page_config(
    page_title="Phase 3 - Temperature BI Dashboard with LSTM Forecasting",
    layout="wide"
)

st.title("Temperature BI Dashboard with LSTM Forecasting")

# -----------------------------------------------------
# 1) TIMEZONE DROPDOWN
# -----------------------------------------------------
tz_options = [
    "auto",
    "GMT",
    "America/Anchorage",
    "America/Los_Angeles",
    "America/Denver",
    "America/Chicago",
    "America/New_York",
    "America/Sao_Paolo",
    "Europe/London",
    "Europe/Berlin",
    "Europe/Moscow",
    "Africa/Cairo",
    "Asia/Bangkok",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Australia/Sydney",
    "Pacific/Auckland"
]

selected_tz = st.selectbox(
    "Select Time Zone ('auto' is based on your operating system's set time zone)",
    options=tz_options,
    index=1
)

# -----------------------------------------------------
# 2) LOCATION INPUTS (Manual vs Geopy)
# -----------------------------------------------------
if 'lat' not in st.session_state:
    st.session_state.lat = 54.5695
if 'lon' not in st.session_state:
    st.session_state.lon = -1.2355

st.subheader("Location Selection")
loc_method = st.radio(
    "Choose input method:",
    ["Enter Coordinates", "Search by Address/Landmark"],
    horizontal=True
)

if loc_method == "Enter Coordinates":
    st.session_state.lat = st.number_input("LATITUDE", value=st.session_state.lat, format="%.4f")
    st.session_state.lon = st.number_input("LONGITUDE", value=st.session_state.lon, format="%.4f")
else:
    address_input = st.text_input("Enter Location, Address, or Landmark  (ex. Berlin, Germany / 10 Downing Street / Eiffel Tower / Teesside University)")
    if st.button("Get Coordinates"):
        if address_input:
            try:
                geolocator = Nominatim(user_agent="weather_bi_dashboard_streamlit")
                location = geolocator.geocode(address_input)
                if location:
                    st.session_state.lat = location.latitude
                    st.session_state.lon = location.longitude
                    st.success(f"Found: {location.address}")
                else:
                    st.error("Location not found. Please try a different search term.")
            except Exception as e:
                st.error(f"Geocoding error: {e}")
    
    st.info(f"**Active Coordinates:** Lat: {st.session_state.lat:.4f}, Lon: {st.session_state.lon:.4f}")

# -----------------------------------------------------
# PYDECK MAP (Uses session state to update dynamically)
# -----------------------------------------------------
data_location = pd.DataFrame({
    'lat': [st.session_state.lat],
    'lng': [st.session_state.lon]
})

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data_location,
    get_position=["lng", "lat"],
    get_color=[255, 100, 100, 160],
    get_radius=200,
    pickable=True,
)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=st.session_state.lat,
        longitude=st.session_state.lon,
        zoom=13,
        pitch=50,
    ),
    layers=[layer]
))

# -----------------------------------------------------
# 3) USER INPUTS (Timezone aware End Date)
# -----------------------------------------------------
if selected_tz == "auto":
    # datetime.now() without arguments automatically uses the 
    # operating system's local timezone settings.
    today_tz = datetime.now().date()
else:
    # Use the specifically selected IANA timezone
    tz_obj = ZoneInfo(selected_tz)
    today_tz = datetime.now(tz_obj).date()


st.write("")
st.write("Enter the start and end dates of the historical temperature to be retrieve. The maximum value of the end date is set based on the time zone selected.")


col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=today_tz - timedelta(days=30),
        max_value=today_tz  # Optional: Also prevents start date from exceeding today
    )
with col2:
    # ADDED max_value=today_tz to prevent selecting a date higher than today's date in the selected TZ
    end_date = st.date_input(
        "End Date",
        value=today_tz,
        max_value=today_tz  
    )

forecast_hours = st.number_input(
    "Forecast Horizon (Hours)",
    min_value=24,
    max_value=240,
    value=48,
    step=24
)

if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()


# =====================================================
# DATABASE FUNCTIONS
# =====================================================

def create_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observation_time TEXT NOT NULL,
            temperature_2m REAL,
            load_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_time TEXT,
            forecast_temperature REAL,
            model_name TEXT,
            created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_to_database(df):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM weather_hourly")
    df_store = df.rename(columns={"time": "observation_time"})
    df_store.to_sql("weather_hourly", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def load_from_database():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT observation_time AS time, temperature_2m
        FROM weather_hourly
        ORDER BY observation_time
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def save_forecast_to_database(forecast_times, forecast_values):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM weather_forecast")
    forecast_df = pd.DataFrame({
        "forecast_time": forecast_times,
        "forecast_temperature": forecast_values,
        "model_name": "LSTM"
    })
    forecast_df.to_sql("weather_forecast", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def load_forecast_from_database():
    conn = sqlite3.connect(DB_PATH)
    forecast_df = pd.read_sql_query(
        """
        SELECT forecast_time, forecast_temperature
        FROM weather_forecast
        ORDER BY forecast_time
        """, conn
    )
    conn.close()
    if not forecast_df.empty:
        forecast_df["forecast_time"] = pd.to_datetime(forecast_df["forecast_time"])
    return forecast_df


# =====================================================
# OPEN-METEO FUNCTIONS
# =====================================================

@st.cache_data
def get_weather_data(url, params):
    # Requests automatically URL-encodes the params dictionary
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "time": hourly.get("time", []),
        "temperature_2m": hourly.get("temperature_2m", [])
    })
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


# =====================================================
# LSTM FUNCTIONS
# =====================================================

def create_sequences(data, look_back=24):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def train_lstm_and_forecast(df, forecast_hours):
    temperatures = df["temperature_2m"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(temperatures)
    
    look_back = 24
    X, y = create_sequences(scaled_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    
    last_window = scaled_data[-look_back:]
    predictions_scaled = []
    
    for _ in range(forecast_hours):
        prediction = model.predict(last_window.reshape(1, look_back, 1), verbose=0)
        predictions_scaled.append(prediction[0, 0])
        last_window = np.vstack([last_window[1:], prediction])
        
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    forecast_values = scaler.inverse_transform(predictions_scaled).flatten()
    
    return forecast_values


# =====================================================
# RETRIEVE DATA & DISPLAY
# =====================================================

if st.button("Retrieve Weather Data"):
    try:
        create_database()

        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Build parameters dictionary
        params = {
            "latitude": st.session_state.lat,
            "longitude": st.session_state.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m"
        }
        
        if selected_tz != "auto":
            # Pass the explicitly chosen IANA timezone string to the API
            params["timezone"] = selected_tz
        # If "auto" is selected, we omit the parameter. Open-Meteo natively defaults 
        # to "auto" and resolves the timezone based on the Lat/Lon coordinates provided.

        with st.spinner("Downloading data..."):
            api_df = get_weather_data(base_url, params)

        if api_df.empty:
            st.warning("No weather data returned.")
            st.stop()

        # ---------------------------------
        # Store Historical Data
        # ---------------------------------
        save_to_database(api_df)
        df = load_from_database()

        st.success(f"{len(df):,} records loaded into SQLite database.")
        st.info(f"Database location:\n{DB_PATH}")

        # ---------------------------------
        # Forecast
        # ---------------------------------
        with st.spinner("Training LSTM model..."):
            forecast_values = train_lstm_and_forecast(df, forecast_hours)

        last_time = df["time"].max()
        forecast_times = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq="h"
        )

        save_forecast_to_database(forecast_times, forecast_values)
        forecast_df = load_forecast_from_database()

        # ---------------------------------
        # Historical Data
        # ---------------------------------
        st.subheader("Historical Temperature Data")
        st.dataframe(df, use_container_width=True, height=400, hide_index=True)

        # ---------------------------------
        # Statistics
        # ---------------------------------
        st.subheader("Summary Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum Temperature", f"{df['temperature_2m'].min():.1f} °C")
        c2.metric("Maximum Temperature", f"{df['temperature_2m'].max():.1f} °C")
        c3.metric("Average Temperature", f"{df['temperature_2m'].mean():.1f} °C")

        # ---------------------------------
        # Forecast Table
        # ---------------------------------
        st.subheader("LSTM Forecast Results")
        st.dataframe(forecast_df, hide_index=True, use_container_width=True)

        # ---------------------------------
        # Combined Chart
        # ---------------------------------
        st.subheader("Historical vs Forecast Temperature")
        historical = df.copy()
        historical["Series"] = "Historical"
        
        forecast = forecast_df.rename(columns={
            "forecast_time": "time",
            "forecast_temperature": "temperature_2m"
        })
        forecast["Series"] = "Forecast"

        combined = pd.concat([
            historical[["time", "temperature_2m", "Series"]],
            forecast[["time", "temperature_2m", "Series"]]
        ])

        chart_data = combined.pivot(index="time", columns="Series", values="temperature_2m")
        st.line_chart(chart_data, use_container_width=True)

    except Exception as e:
        st.error(f"Application Error: {e}")


# =====================================================
# 4) RESET SECTION
# =====================================================

st.divider()


if st.button("Reset Database & Cache"):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM weather_hourly")
        conn.execute("DELETE FROM weather_forecast")
        conn.commit()
        conn.close()
        
        # Clear Streamlit cache for the weather API calls
        st.cache_data.clear()
        
        st.success("Database and cache cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Reset Error: {e}")
