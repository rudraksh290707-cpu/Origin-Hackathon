import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import create_features


DATA_PATH = "data/demand_weather.csv"
TARGET = "demand_mw"


# --------------------------------------------------
# Load + feature engineering
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = create_features(df)


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

HISTORICAL_FEATURES = [
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
]

WEATHER_FEATURES = [
    "temperature",
    "humidity",
    "cooling_degree",
    "heating_degree",
]

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

FULL_FEATURES = (
    HISTORICAL_FEATURES
    + WEATHER_FEATURES
    + CALENDAR_FEATURES
)


# --------------------------------------------------
# Time-based split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:].copy()


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(name, features):

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    model = XGBRegressor(
        n_estimators=700,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mape = np.mean(
        np.abs(
            (y_test - predictions)
            / y_test
        )
    ) * 100

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Features: {len(features)}")
    print(f"MAE:      {mae:.2f} MW")
    print(f"RMSE:     {rmse:.2f} MW")
    print(f"MAPE:     {mape:.2f}%")

    return mae, rmse, mape


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

print("=== ABLATION STUDY ===")
print(f"Training rows: {len(train):,}")
print(f"Test rows:     {len(test):,}")


historical_results = evaluate_model(
    "MODEL A — HISTORICAL + CALENDAR",
    HISTORICAL_FEATURES + CALENDAR_FEATURES
)


weather_results = evaluate_model(
    "MODEL B — WEATHER + CALENDAR",
    WEATHER_FEATURES + CALENDAR_FEATURES
)


full_results = evaluate_model(
    "MODEL C — FULL MODEL",
    FULL_FEATURES
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

results = pd.DataFrame(
    {
        "model": [
            "Historical + Calendar",
            "Weather + Calendar",
            "Full Model",
        ],
        "MAE_MW": [
            historical_results[0],
            weather_results[0],
            full_results[0],
        ],
        "RMSE_MW": [
            historical_results[1],
            weather_results[1],
            full_results[1],
        ],
        "MAPE_percent": [
            historical_results[2],
            weather_results[2],
            full_results[2],
        ],
    }
)

print("\n=== SUMMARY ===")
print(results.to_string(index=False))