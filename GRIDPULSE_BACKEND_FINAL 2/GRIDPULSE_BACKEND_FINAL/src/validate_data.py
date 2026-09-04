import pandas as pd

PATH = "data/demand_weather.csv"
df = pd.read_csv(PATH)

# Normalize column names
df.columns = df.columns.str.strip()

# Parse timestamp after normalization
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("=== DATA VALIDATION ===")

# 1. Duplicate timestamps
duplicates = df["timestamp"].duplicated().sum()
print(f"Duplicate timestamps: {duplicates}")

# 2. Expected hourly frequency
diffs = df["timestamp"].diff().dropna()

gaps = diffs[diffs != pd.Timedelta(hours=1)]

print(f"Non-hourly intervals: {len(gaps)}")

if len(gaps) > 0:
    print("\nProblem intervals:")
    print(gaps.head(20))

# 3. Missing values
print("\nMissing values:")
print(df.isna().sum())

# 4. Expected row count
expected_hours = int(
    (df["timestamp"].max() - df["timestamp"].min())
    / pd.Timedelta(hours=1)
) + 1

print(f"\nExpected rows: {expected_hours:,}")
print(f"Actual rows:   {len(df):,}")

# 5. Coverage
print(f"\nStart: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")

# 6. Range sanity checks
print("\nValue ranges:")
print(df[["demand_mw", "temperature", "humidity"]].agg(["min", "max"]))

# 7. Final verdict
valid = (
    duplicates == 0
    and len(gaps) == 0
    and df.isna().sum().sum() == 0
    and len(df) == expected_hours
)

print("\n" + ("✅ DATASET IS CONTINUOUS" if valid
              else "❌ DATASET HAS ISSUES"))