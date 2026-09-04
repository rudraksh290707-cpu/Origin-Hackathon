import os
import sys
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = "data/demand_weather.csv"
MODEL_PATH = "models/demand_xgb.json"


FEATURES = [
    "temperature",
    "humidity",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
    "cooling_degree",
    "heating_degree",
]


def add_features_for_forecast(df):
    """Create model features without using future actual demand."""

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calendar
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Demand lags
    df["lag_1"] = df["demand_mw"].shift(1)
    df["lag_24"] = df["demand_mw"].shift(24)
    df["lag_168"] = df["demand_mw"].shift(168)

    # Historical rolling features
    df["rolling_mean_24"] = (
        df["demand_mw"]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["rolling_mean_168"] = (
        df["demand_mw"]
        .shift(1)
        .rolling(168)
        .mean()
    )

    df["rolling_std_24"] = (
        df["demand_mw"]
        .shift(1)
        .rolling(24)
        .std()
    )

    # Weather
    df["cooling_degree"] = (
        df["temperature"] - 24
    ).clip(lower=0)

    df["heating_degree"] = (
        18 - df["temperature"]
    ).clip(lower=0)

    return df


def get_open_meteo_weather(start_time, horizon_hours):
    """Fetch continuous hourly weather from Open-Meteo."""

    end_time = (
        start_time
        + pd.Timedelta(hours=horizon_hours - 1)
    )

    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Asia/Kolkata",
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
    }

    print("\nFetching weather from Open-Meteo...")
    print(
        f"Requested: {start_time} → {end_time}"
    )

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            data.get("reason", "Open-Meteo error")
        )

    hourly = data.get("hourly")

    if hourly is None:
        raise RuntimeError(
            "Open-Meteo returned no hourly data."
        )

    weather = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
    })

    weather["timestamp"] = (
        weather["timestamp"]
        .dt.tz_localize(None)
    )

    weather = weather[
        (weather["timestamp"] >= start_time)
        & (weather["timestamp"] <= end_time)
    ].copy()

    weather = (
        weather
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =========================================================
    # STRICT VALIDATION
    # =========================================================

    expected = pd.date_range(
        start=start_time,
        periods=horizon_hours,
        freq="h",
    )

    if len(weather) != horizon_hours:
        raise RuntimeError(
            f"Expected {horizon_hours} weather records, "
            f"received {len(weather)}."
        )

    if weather["timestamp"].duplicated().any():
        raise RuntimeError(
            "Duplicate weather timestamps detected."
        )

    if not weather["timestamp"].equals(
        pd.Series(expected, name="timestamp")
    ):
        raise RuntimeError(
            "Weather timestamps are not continuous."
        )

    if weather[
        ["temperature", "humidity"]
    ].isna().any().any():
        raise RuntimeError(
            "Missing weather values detected."
        )

    print(
        f"Weather records received: {len(weather)}"
    )

    print("Weather validation: PASSED")

    return weather


def forecast_demand(df, model, future_weather):
    """Recursively forecast demand for every future hour."""

    working = df[
        [
            "timestamp",
            "demand_mw",
            "temperature",
            "humidity",
        ]
    ].copy()

    predictions = []

    for _, future_row in future_weather.iterrows():

        new_row = pd.DataFrame([{
            "timestamp": future_row["timestamp"],
            "demand_mw": np.nan,
            "temperature": future_row["temperature"],
            "humidity": future_row["humidity"],
        }])

        working = pd.concat(
            [working, new_row],
            ignore_index=True,
        )

        featured = add_features_for_forecast(
            working
        )

        current = featured.iloc[-1]

        missing = [
            f
            for f in FEATURES
            if pd.isna(current[f])
        ]

        if missing:
            raise RuntimeError(
                f"Missing features at "
                f"{future_row['timestamp']}: {missing}"
            )

        X = pd.DataFrame(
            [[current[f] for f in FEATURES]],
            columns=FEATURES,
        )

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(prediction, 0)

        # Feed prediction back into future history.
        working.loc[
            working.index[-1],
            "demand_mw"
        ] = prediction

        predictions.append(prediction)

    return predictions


def build_daily_summary(forecast):
    """Create daily planning-level summary."""

    daily = (
        forecast
        .set_index("timestamp")
        .resample("D")
        .agg(
            peak_demand_mw=(
                "predicted_demand_mw",
                "max",
            ),
            average_demand_mw=(
                "predicted_demand_mw",
                "mean",
            ),
            min_demand_mw=(
                "predicted_demand_mw",
                "min",
            ),
            peak_temperature=(
                "temperature",
                "max",
            ),
        )
        .reset_index()
    )

    return daily


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        choices=[24, 168],
        help="Forecast horizon: 24 hours or 168 hours",
    )

    args = parser.parse_args()

    horizon = args.hours

    print("=" * 60)
    print(
        f"=== {horizon}-HOUR DEMAND FORECAST ==="
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # Load historical data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.strip()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(
        f"Historical rows: {len(df):,}"
    )

    last_timestamp = df["timestamp"].iloc[-1]

    print(
        f"Last known timestamp: "
        f"{last_timestamp}"
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = xgb.XGBRegressor()

    model.load_model(MODEL_PATH)

    print("Model loaded.")

    # ---------------------------------------------------------
    # Forecast start
    # ---------------------------------------------------------

    forecast_start = (
        last_timestamp
        + pd.Timedelta(hours=1)
    )

    # ---------------------------------------------------------
    # Future weather
    # ---------------------------------------------------------

    future_weather = get_open_meteo_weather(
        forecast_start,
        horizon,
    )

    # ---------------------------------------------------------
    # Demand forecast
    # ---------------------------------------------------------

    print(
        "\nGenerating recursive demand forecast..."
    )

    predictions = forecast_demand(
        df,
        model,
        future_weather,
    )

    forecast = future_weather.copy()

    forecast["predicted_demand_mw"] = predictions

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if len(forecast) != horizon:
        raise RuntimeError(
            f"Expected {horizon} predictions, "
            f"got {len(forecast)}."
        )

    if forecast.isna().any().any():
        raise RuntimeError(
            "Final forecast contains missing values."
        )

    expected = pd.date_range(
        start=forecast_start,
        periods=horizon,
        freq="h",
    )

    if not forecast["timestamp"].equals(
        pd.Series(expected, name="timestamp")
    ):
        raise RuntimeError(
            "Final forecast timestamps are not continuous."
        )

    # ---------------------------------------------------------
    # Peak
    # ---------------------------------------------------------

    peak_idx = forecast[
        "predicted_demand_mw"
    ].idxmax()

    peak = forecast.loc[peak_idx]

    # ---------------------------------------------------------
    # Save hourly forecast
    # ---------------------------------------------------------

    output_path = (
        f"data/forecast_{horizon}h.csv"
    )

    forecast.to_csv(
        output_path,
        index=False,
    )

    # Also maintain the standard 24h file.
    if horizon == 24:
        forecast.to_csv(
            "data/forecast_24h.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Daily summary
    # ---------------------------------------------------------

    if horizon >= 168:

        daily = build_daily_summary(
            forecast
        )

        daily_path = (
            "data/forecast_7d_daily.csv"
        )

        daily.to_csv(
            daily_path,
            index=False,
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n=== FORECAST RESULTS ===")

    print(
        f"Forecast period: "
        f"{forecast['timestamp'].iloc[0]} → "
        f"{forecast['timestamp'].iloc[-1]}"
    )

    print(
        f"Peak demand: "
        f"{peak['predicted_demand_mw']:,.2f} MW"
    )

    print(
        f"Peak time: "
        f"{peak['timestamp']}"
    )

    print(
        f"Average demand: "
        f"{forecast['predicted_demand_mw'].mean():,.2f} MW"
    )

    print(
        f"Minimum demand: "
        f"{forecast['predicted_demand_mw'].min():,.2f} MW"
    )

    print(
        "\nWeather source: Open-Meteo"
    )

    print(
        "Forecast validation: PASSED"
    )

    print(
        f"\nHourly forecast saved to: "
        f"{output_path}"
    )

    if horizon >= 168:

        print(
            f"Daily summary saved to: "
            f"{daily_path}"
        )

        print("\n=== 7-DAY DAILY SUMMARY ===")

        print(
            daily.to_string(index=False)
        )


if __name__ == "__main__":
    main()