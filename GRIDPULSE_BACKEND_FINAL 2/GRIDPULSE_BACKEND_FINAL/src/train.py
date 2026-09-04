import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import create_features


DATA_PATH = "data/demand_weather.csv"
MODEL_PATH = "models/demand_xgb.json"

TARGET = "demand_mw"


# --------------------------------------------------
# Load + feature engineering
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = create_features(df)


# --------------------------------------------------
# Feature list
# --------------------------------------------------

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


X = df[FEATURES]
y = df[TARGET]


# --------------------------------------------------
# Chronological train/test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_timestamps = df["timestamp"].iloc[split_index:]


print("=== XGBOOST TRAINING ===")

print(f"Training rows: {len(X_train):,}")
print(f"Test rows:     {len(X_test):,}")

print(
    f"Test period: "
    f"{test_timestamps.iloc[0]} → "
    f"{test_timestamps.iloc[-1]}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

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


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)


# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

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


baseline_mae = 247.54

improvement = (
    (baseline_mae - mae)
    / baseline_mae
) * 100


print("\n=== RESULTS ===")

print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")

print(
    f"\nBaseline MAE: {baseline_mae:.2f} MW"
)

print(
    f"Improvement over baseline: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = (
    pd.Series(
        model.feature_importances_,
        index=FEATURES
    )
    .sort_values(ascending=False)
)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)


# --------------------------------------------------
# Save model
# --------------------------------------------------

model.save_model(MODEL_PATH)

print(
    f"\nModel saved to: {MODEL_PATH}"
)