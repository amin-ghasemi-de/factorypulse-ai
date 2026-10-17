
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metropt3_processed.parquet"
)

SENSORS = [
    "Motor_current",
    "Oil_temperature",
    "TP2",
    "TP3",
]

FAILURE_PERIODS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]


def prepare_ml_data():

    # Load processed dataset
    df = pd.read_parquet(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Dataset shape:", df.shape)
    print("Number of sensors:", len(SENSORS))
    print("Total segments:", df["segment_id"].nunique())

    # Validate sensor values
    missing_values = df[SENSORS].isna().sum()

    if missing_values.any():
        raise ValueError("Sensor data contains missing values.")

    # Chronological split
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.80)

    train_df = df.iloc[:train_end].copy()

    val_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[validation_end:].copy()

    print("\nTrain shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("Train period:",
          train_df["timestamp"].min(),
          "to",
          train_df["timestamp"].max())

    print("Validation period:",
          val_df["timestamp"].min(),
          "to",
          val_df["timestamp"].max())

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Exclude known failure periods from training
    def exclude_failure_periods(data):

        failure_mask = pd.Series(
            False,
            index=data.index
        )

        for start, end in FAILURE_PERIODS:
            failure_mask |= data["timestamp"].between(
                pd.Timestamp(start),
                pd.Timestamp(end)
            )

        return data.loc[~failure_mask].copy()

    train_clean = exclude_failure_periods(train_df)
    val_clean = exclude_failure_periods(val_df)

    print("\nClean Train rows:", len(train_clean))
    print("Clean Validation rows:", len(val_clean))

    # Select sensor features
    X_train = train_clean[SENSORS]
    X_val = val_clean[SENSORS]
    X_test = test_df[SENSORS]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled Train:", X_train_scaled.shape)
    print("Scaled Validation:", X_val_scaled.shape)
    print("Scaled Test:", X_test_scaled.shape)

    return (
        train_clean,
        val_clean,
        test_df,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )
