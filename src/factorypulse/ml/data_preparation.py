
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
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # Exclude known failure periods from training
    failure_mask = pd.Series(False, index=train_df.index)

    for start, end in FAILURE_PERIODS:
        failure_mask |= train_df["timestamp"].between(
            pd.Timestamp(start),
            pd.Timestamp(end)
        )

    train_clean = train_df.loc[~failure_mask].copy()

    print("\nExcluded failure rows:", failure_mask.sum())
    print("Clean Train rows:", len(train_clean))

    # Select sensor features
    X_train = train_clean[SENSORS]
    X_test = test_df[SENSORS]

    # Fit scaler only on clean training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled Train:", X_train_scaled.shape)
    print("Scaled Test:", X_test_scaled.shape)

    return (
        train_clean,
        test_df,
        X_train_scaled,
        X_test_scaled,
        scaler,
    )
