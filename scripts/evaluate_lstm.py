
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from factorypulse.ml.sequence_dataset import CompressorSequenceDataset
from factorypulse.ml.lstm_autoencoder import LSTMAutoencoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "metropt3_processed.parquet"
)

MODEL_PATH = PROJECT_ROOT / "models" / "lstm_autoencoder_baseline.pt"
SCALER_PATH = PROJECT_ROOT / "models" / "sensor_scaler.joblib"


def main():

    # 1. Load the trained model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    sensors = checkpoint["sensors"]
    sequence_length = checkpoint["sequence_length"]

    model = LSTMAutoencoder(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Load the fitted training scaler
    scaler = joblib.load(SCALER_PATH)

    # 3. Load test data
    df = pd.read_parquet(
        DATA_PATH,
        columns=["timestamp", "segment_id", *sensors],
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    X_test_scaled = scaler.transform(test_df[sensors])

    # 4. Create valid test sequences
    test_dataset = CompressorSequenceDataset(
        data=X_test_scaled,
        segment_ids=test_df["segment_id"].to_numpy(),
        source_indices=test_df.index.to_numpy(),
        sequence_length=sequence_length,
    )

    print("Valid test sequences:", len(test_dataset))

    # 5. Select 512 sequences across the test period
    sample_size = min(512, len(test_dataset))

    if sample_size == 0:
        raise ValueError("No valid test sequences found.")

    indices = np.linspace(
        0,
        len(test_dataset) - 1,
        sample_size,
        dtype=np.int64,
    )

    test_subset = Subset(test_dataset, indices.tolist())

    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # 6. Calculate reconstruction error
    scores = []

    with torch.inference_mode():
        for batch in test_loader:

            reconstruction = model(batch)

            error = (
                (reconstruction - batch) ** 2
            ).mean(dim=(1, 2))

            scores.extend(error.tolist())

    # 7. Associate scores with timestamps
    timestamps = []

    for idx in indices:
        start = test_dataset.start_positions[idx]
        end = start + sequence_length - 1

        timestamps.append(
            test_df.iloc[end]["timestamp"]
        )

    results = pd.DataFrame({
        "timestamp": timestamps,
        "reconstruction_error": scores,
    })

    print("\nReconstruction Error Summary:")
    print(results["reconstruction_error"].describe())

    print("\nTop 10 Highest Reconstruction Errors:")
    print(
        results.nlargest(
            10,
            "reconstruction_error",
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
