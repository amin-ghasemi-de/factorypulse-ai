import pandas as pd
from factorypulse.validation.data_validator import (
    check_missing_values,
    check_timestamp_order,
)
def test_check_missing_values():
    # Arrange: Create sample data
    df = pd.DataFrame({
        "temperature": [50.0, None, 55.0],
        "pressure": [9.1, 9.2, None]
    })

    # Act: Run the function
    result = check_missing_values(df)

    # Assert: Verify the result
    assert result["temperature"] == 1
    assert result["pressure"] == 1
def test_check_missing_values_no_missing_data():
    # Arrange: Create data without missing values
    df = pd.DataFrame({
        "temperature": [50.0, 52.0, 55.0],
        "pressure": [9.1, 9.2, 9.3]
    })

    # Act: Run the function
    result = check_missing_values(df)

    # Assert: Both columns should have zero missing values
    assert result["temperature"] == 0
    assert result["pressure"] == 0
def test_check_timestamp_order_valid():
    # Arrange
    df = pd.DataFrame({
        "timestamp": [
            "2020-02-01 10:00:00",
            "2020-02-01 10:00:10",
            "2020-02-01 10:00:20"
        ]
    })

    # Act
    result = check_timestamp_order(df)

    # Assert
    assert result is True

def test_check_timestamp_order_invalid():
    # Arrange: Create a DataFrame with unordered timestamps
    df = pd.DataFrame({
        "timestamp": [
            "2020-02-01 10:00:20",
            "2020-02-01 10:00:10",
            "2020-02-01 10:00:30"
        ]
    })

    # Act: Run the function
    result = check_timestamp_order(df)

    # Assert: The timestamps are not in order, so the result should be False
    assert result is False