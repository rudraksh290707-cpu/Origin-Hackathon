import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(
    "data/demand_weather.csv"
)

df.columns = df.columns.str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------
# Baseline prediction
# --------------------------------------------------

df["baseline_prediction"] = (
    df["demand_mw"].shift(24)
)


# Remove rows where baseline isn't available
df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# Time-based test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

train = df.iloc[:split_index]
test = df.iloc[split_index:].copy()


# --------------------------------------------------
# Evaluate
# --------------------------------------------------

actual = test["demand_mw"]
predicted = test["baseline_prediction"]

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

mape = np.mean(
    np.abs(
        (actual - predicted) / actual
    )
) * 100


# --------------------------------------------------
# Results
# --------------------------------------------------

print("=== BASELINE MODEL ===")

print(f"Training rows: {len(train):,}")
print(f"Test rows:     {len(test):,}")

print("\nMetrics:")
print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")

print("\nTest period:")
print(f"Start: {test['timestamp'].min()}")
print(f"End:   {test['timestamp'].max()}")