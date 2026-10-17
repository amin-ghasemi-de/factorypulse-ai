from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "metropt3_processed.parquet"

GAP_THRESHOLD_SECONDS = 60

# Load raw dataset
df = pd.read_csv(RAW_PATH)

# Convert timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Calculate sampling intervals
df["time_gap_seconds"] = df["timestamp"].diff().dt.total_seconds()

# Create segment IDs
df["segment_id"] = (
    df["time_gap_seconds"] > GAP_THRESHOLD_SECONDS
).cumsum().astype("int32")

# Save processed dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_parquet(OUTPUT_PATH, index=False)

print("Processed dataset saved successfully!")
print("Shape:", df.shape)
print("Segments:", df["segment_id"].nunique())
print("Output:", OUTPUT_PATH)