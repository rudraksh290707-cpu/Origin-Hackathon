import pandas as pd


TARGET = "demand_mw"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Make sure timestamps are clean
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Cyclical time features
    # --------------------------------------------------

    df["hour_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi * df["hour"] / 24
        )
    )

    df["hour_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi * df["hour"] / 24
        )
    )

    df["dow_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi * df["day_of_week"] / 7
        )
    )

    df["dow_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi * df["day_of_week"] / 7
        )
    )

    # --------------------------------------------------
    # Demand lag features
    # --------------------------------------------------

    # Previous hour
    df["lag_1"] = df[TARGET].shift(1)

    # Same hour yesterday
    df["lag_24"] = df[TARGET].shift(24)

    # Same hour last week
    df["lag_168"] = df[TARGET].shift(168)

    # --------------------------------------------------
    # Rolling demand features
    # --------------------------------------------------

    df["rolling_mean_24"] = (
        df[TARGET]
        .shift(1)
        .rolling(24)
        .mean()
    )

    df["rolling_mean_168"] = (
        df[TARGET]
        .shift(1)
        .rolling(168)
        .mean()
    )

    df["rolling_std_24"] = (
        df[TARGET]
        .shift(1)
        .rolling(24)
        .std()
    )

    # --------------------------------------------------
    # Temperature features
    # --------------------------------------------------

    # Cooling stress
    df["cooling_degree"] = (
        df["temperature"] - 24
    ).clip(lower=0)

    # Heating stress
    df["heating_degree"] = (
        18 - df["temperature"]
    ).clip(lower=0)

    # --------------------------------------------------
    # Remove rows where lag/rolling features
    # aren't available
    # --------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    df = pd.read_csv(
        "data/demand_weather.csv"
    )

    df.columns = df.columns.str.strip()

    result = create_features(df)

    print("=== FEATURE ENGINEERING ===")
    print(f"Original rows: {len(df):,}")
    print(f"Feature rows:  {len(result):,}")
    print(f"Rows removed:  {len(df) - len(result):,}")

    print("\nFeatures:")
    print(result.columns.tolist())

    print("\nMissing values:")
    print(result.isna().sum())

    print("\nSample:")
    print(result.head())