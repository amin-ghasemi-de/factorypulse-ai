import pandas as pd


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()
def check_timestamp_order(df: pd.DataFrame) -> bool:
    """Check whether timestamps are in chronological order."""

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")

    return (
        timestamps.notna().all()
        and timestamps.is_monotonic_increasing
        and timestamps.is_unique
    )