
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metropt3_processed.parquet"
)


# --------------------------------------------------
# 2. Sensor configuration
# --------------------------------------------------

SENSORS = [
    "Motor_current",
    "Oil_temperature",
    "TP2",
    "TP3",
]


# --------------------------------------------------
# 3. Known failure periods
# --------------------------------------------------

FAILURE_PERIODS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]


# --------------------------------------------------
# 4. Load processed dataset
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Number of sensors:", len(SENSORS))
print("Total segments:", df["segment_id"].nunique())


# --------------------------------------------------
# 5. Check missing sensor values
# --------------------------------------------------

missing_values = df[SENSORS].isna().sum()

print("\nMissing sensor values:")
print(missing_values)

if missing_values.any():
    raise ValueError("Sensor data contains missing values.")


# --------------------------------------------------
# 6. Chronological Train/Test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)

print(
    "Train period:",
    train_df["timestamp"].min(),
    "to",
    train_df["timestamp"].max()
)

print(
    "Test period:",
    test_df["timestamp"].min(),
    "to",
    test_df["timestamp"].max()
)


# --------------------------------------------------
# 7. Exclude known failures from training
# --------------------------------------------------

failure_mask = pd.Series(False, index=train_df.index)

for start, end in FAILURE_PERIODS:
    failure_mask |= train_df["timestamp"].between(
        pd.Timestamp(start),
        pd.Timestamp(end)
    )

train_clean = train_df.loc[~failure_mask].copy()

print("\nOriginal Train rows:", len(train_df))
print("Excluded failure rows:", failure_mask.sum())
print("Clean Train rows:", len(train_clean))


# --------------------------------------------------
# 8. Select sensor data
# --------------------------------------------------

X_train = train_clean[SENSORS]
X_test = test_df[SENSORS]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# --------------------------------------------------
# 9. Normalize sensor values
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Train:", X_train_scaled.shape)
print("Scaled Test:", X_test_scaled.shape)

print("\nData preparation completed successfully!")
