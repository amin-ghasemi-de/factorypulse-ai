from pathlib import Path
import pandas as pd

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"

# Read the industrial dataset
df = pd.read_csv(DATA_PATH)

# Basic dataset inspection
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Check missing values
print("\nMissing values per column:")
print(df.isna().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check timestamp range
print("\nFirst timestamp:")
print(df["timestamp"].min())

print("\nLast timestamp:")
print(df["timestamp"].max())

# Check chronological order
print("\nTimestamps sorted:")
print(df["timestamp"].is_monotonic_increasing)

# Check duplicate timestamps
print("\nDuplicate timestamps:")
print(df["timestamp"].duplicated().sum())

# Calculate time differences between consecutive records
time_diff = df["timestamp"].diff()

# Convert time differences to seconds
interval_seconds = time_diff.dt.total_seconds()

print("\nSampling interval statistics (seconds):")
print(interval_seconds.describe())

print("\nMost common sampling intervals:")
print(interval_seconds.value_counts().head(10))

print("\nNumber of gaps greater than 60 seconds:")
print((interval_seconds > 60).sum())

print("\nLargest time gap:")
print(time_diff.max())
# Categorize large time gaps
print("\nTime gap categories:")

print("60 seconds to 5 minutes:",
      ((interval_seconds > 60) & (interval_seconds <= 300)).sum())

print("5 minutes to 1 hour:",
      ((interval_seconds > 300) & (interval_seconds <= 3600)).sum())

print("1 hour to 24 hours:",
      ((interval_seconds > 3600) & (interval_seconds <= 86400)).sum())

print("More than 24 hours:",
      (interval_seconds > 86400).sum())

# Identify the 10 largest gaps
print("\nTop 10 largest gaps:")

largest_gaps = interval_seconds.nlargest(10)

for index, seconds in largest_gaps.items():
    print(
        f"Previous: {df.loc[index - 1, 'timestamp']} | "
        f"Current: {df.loc[index, 'timestamp']} | "
        f"Gap: {seconds / 3600:.2f} hours"
    )
# Define maximum allowed gap between consecutive records
GAP_THRESHOLD_SECONDS = 60

# Identify where a new segment begins
new_segment = interval_seconds.gt(GAP_THRESHOLD_SECONDS)

# Assign a unique segment ID to each continuous section
df["segment_id"] = new_segment.cumsum().astype("int32")

# Inspect segmentation results
print("\nTotal segments:")
print(df["segment_id"].nunique())

print("\nFirst 10 segment sizes:")
print(df.groupby("segment_id").size().head(10))

print("\nSegment size statistics:")
print(df.groupby("segment_id").size().describe())

# Calculate segment sizes
segment_sizes = df.groupby("segment_id").size()

# Candidate rolling window sizes
window_sizes = [30, 60, 120, 360]

print("\nSegment eligibility for rolling windows:")

for window in window_sizes:
    eligible_segments = (segment_sizes >= window).sum()
    eligible_rows = segment_sizes[segment_sizes >= window].sum()

    print(
        f"Window {window}: "
        f"{eligible_segments} eligible segments, "
        f"{eligible_rows} rows in eligible segments"
    )