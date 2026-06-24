import streamlit as st
import pandas as pd
import requests
import sqlite3
import numpy as np

from pathlib import Path
from datetime import date, timedelta
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =====================================================
# CONFIGURATION
# =====================================================

#BERLIN
#LATITUDE = 52.52
#LONGITUDE = 13.41

#MANILA
LATITUDE = 14.60
LONGITUDE = 120.9872



#DOWNLOADS_FOLDER = Path.home() / "Downloads"
#DOWNLOADS_FOLDER.mkdir(exist_ok=True)
#DB_PATH = DOWNLOADS_FOLDER / "weather_data.db"

DOWNLOADS_FOLDER =  Path.cwd()
DB_PATH = DOWNLOADS_FOLDER / "weather_data.db"


st.set_page_config(
    page_title="Phase 3 - Weather BI Dashboard with LSTM Forecasting",
    layout="wide"
)

st.title("Weather BI Dashboard with LSTM Forecasting - Berlin")

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

    df_store = df.rename(
        columns={
            "time": "observation_time"
        }
    )

    df_store.to_sql(
        "weather_hourly",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()


def load_from_database():

    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            observation_time AS time,
            temperature_2m
        FROM weather_hourly
        ORDER BY observation_time
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])

    return df


def save_forecast_to_database(
        forecast_times,
        forecast_values):

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        "DELETE FROM weather_forecast"
    )

    forecast_df = pd.DataFrame({
        "forecast_time": forecast_times,
        "forecast_temperature": forecast_values,
        "model_name": "LSTM"
    })

    forecast_df.to_sql(
        "weather_forecast",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()


def load_forecast_from_database():

    conn = sqlite3.connect(DB_PATH)

    forecast_df = pd.read_sql_query(
        """
        SELECT
            forecast_time,
            forecast_temperature
        FROM weather_forecast
        ORDER BY forecast_time
        """,
        conn
    )

    conn.close()

    if not forecast_df.empty:
        forecast_df["forecast_time"] = pd.to_datetime(
            forecast_df["forecast_time"]
        )

    return forecast_df

# =====================================================
# OPEN-METEO FUNCTIONS
# =====================================================

def build_api_url(start_date, end_date):

    return (
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m"
    )


@st.cache_data
def get_weather_data(url):

    response = requests.get(
        url,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    hourly = data.get("hourly", {})

    df = pd.DataFrame({
        "time": hourly.get("time", []),
        "temperature_2m": hourly.get(
            "temperature_2m",
            []
        )
    })

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])

    return df

# =====================================================
# LSTM FUNCTIONS
# =====================================================

def create_sequences(data, look_back=24):

    X = []
    y = []

    for i in range(
        len(data) - look_back
    ):
        X.append(
            data[i:i + look_back]
        )

        y.append(
            data[i + look_back]
        )

    return np.array(X), np.array(y)


def train_lstm_and_forecast(
        df,
        forecast_hours):

    temperatures = (
        df["temperature_2m"]
        .values
        .reshape(-1, 1)
    )

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        temperatures
    )

    look_back = 24

    X, y = create_sequences(
        scaled_data,
        look_back
    )

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    model = Sequential()

    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(look_back, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=16,
        verbose=0
    )

    last_window = scaled_data[-look_back:]

    predictions_scaled = []

    for _ in range(forecast_hours):

        prediction = model.predict(
            last_window.reshape(
                1,
                look_back,
                1
            ),
            verbose=0
        )

        predictions_scaled.append(
            prediction[0, 0]
        )

        last_window = np.vstack([
            last_window[1:],
            prediction
        ])

    predictions_scaled = np.array(
        predictions_scaled
    ).reshape(-1, 1)

    forecast_values = (
        scaler.inverse_transform(
            predictions_scaled
        )
        .flatten()
    )

    return forecast_values

# =====================================================
# USER INPUTS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    start_date = st.date_input(
        "Start Date",
        value=date.today() -
        timedelta(days=30)
    )

with col2:

    end_date = st.date_input(
        "End Date",
        value=date.today()
    )

forecast_hours = st.number_input(
    "Forecast Horizon (Hours)",
    min_value=24,
    max_value=240,
    value=48,
    step=24
)

if start_date > end_date:

    st.error(
        "Start date must be before end date."
    )

    st.stop()

# =====================================================
# RETRIEVE DATA
# =====================================================

if st.button("Retrieve Weather Data"):

    try:

        create_database()

        api_url = build_api_url(
            start_date,
            end_date
        )

        with st.spinner(
                "Downloading data..."):

            api_df = get_weather_data(
                api_url
            )

        if api_df.empty:

            st.warning(
                "No weather data returned."
            )

            st.stop()

        # ---------------------------------
        # Store Historical Data
        # ---------------------------------

        save_to_database(api_df)

        df = load_from_database()

        st.success(
            f"{len(df):,} records loaded "
            f"into SQLite database."
        )

        st.info(
            f"Database location:\n{DB_PATH}"
        )

        # ---------------------------------
        # Forecast
        # ---------------------------------

        with st.spinner(
                "Training LSTM model..."):

            forecast_values = (
                train_lstm_and_forecast(
                    df,
                    forecast_hours
                )
            )

        last_time = df["time"].max()

        forecast_times = pd.date_range(
            start=last_time +
            pd.Timedelta(hours=1),
            periods=forecast_hours,
            freq="h"
        )

        save_forecast_to_database(
            forecast_times,
            forecast_values
        )

        forecast_df = (
            load_forecast_from_database()
        )

        # ---------------------------------
        # Historical Data
        # ---------------------------------

        st.subheader(
            "Historical Weather Data"
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # ---------------------------------
        # Statistics
        # ---------------------------------

        st.subheader(
            "Summary Statistics"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Minimum Temperature",
            f"{df['temperature_2m'].min():.1f} °C"
        )

        c2.metric(
            "Maximum Temperature",
            f"{df['temperature_2m'].max():.1f} °C"
        )

        c3.metric(
            "Average Temperature",
            f"{df['temperature_2m'].mean():.1f} °C"
        )

        # ---------------------------------
        # Forecast Table
        # ---------------------------------

        st.subheader(
            "LSTM Forecast Results"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # ---------------------------------
        # Combined Chart
        # ---------------------------------

        st.subheader(
            "Historical vs Forecast Temperature"
        )

        historical = df.copy()

        historical["Series"] = (
            "Historical"
        )

        forecast = forecast_df.rename(
            columns={
                "forecast_time": "time",
                "forecast_temperature":
                    "temperature_2m"
            }
        )

        forecast["Series"] = (
            "Forecast"
        )

        combined = pd.concat(
            [
                historical[
                    [
                        "time",
                        "temperature_2m",
                        "Series"
                    ]
                ],
                forecast[
                    [
                        "time",
                        "temperature_2m",
                        "Series"
                    ]
                ]
            ]
        )

        chart_data = combined.pivot(
            index="time",
            columns="Series",
            values="temperature_2m"
        )

        st.line_chart(
            chart_data,
            use_container_width=True
        )

        # ---------------------------------
        # CSV Export
        # ---------------------------------

        if st.button(
                "Export Historical Data to CSV"):

            csv_file = (
                DOWNLOADS_FOLDER /
                f"weather_"
                f"{start_date}_"
                f"{end_date}.csv"
            )

            df.to_csv(
                csv_file,
                index=False
            )

            st.success(
                f"CSV saved:\n{csv_file}"
            )

    except Exception as e:

        st.error(
            f"Application Error: {e}"
        )