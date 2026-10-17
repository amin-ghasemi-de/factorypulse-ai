
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import joblib
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from factorypulse.ml.data_preparation import (
    prepare_ml_data,
    SENSORS,
)

from factorypulse.ml.sequence_dataset import (
    CompressorSequenceDataset,
)

from factorypulse.ml.lstm_autoencoder import (
    LSTMAutoencoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"

SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
SUBSET_SIZE = 2048
RANDOM_SEED = 42


def main():

    torch.manual_seed(RANDOM_SEED)

    # 1. Prepare data
    (
        train_clean,
        val_clean,
        test_df,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = prepare_ml_data()

    # 2. Create training sequences
    train_dataset = CompressorSequenceDataset(
        data=X_train_scaled,
        segment_ids=train_clean["segment_id"].to_numpy(),
        source_indices=train_clean.index.to_numpy(),
        sequence_length=SEQUENCE_LENGTH,
    )

    print("\nValid training sequences:", len(train_dataset))

    val_dataset = CompressorSequenceDataset(
        data=X_val_scaled,
        segment_ids=val_clean["segment_id"].to_numpy(),
        source_indices=val_clean.index.to_numpy(),
        sequence_length=SEQUENCE_LENGTH,
    )

    print("Valid validation sequences:", len(val_dataset))
    # 3. Select baseline training subset
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:SUBSET_SIZE].tolist()

    training_subset = Subset(train_dataset, indices)

    training_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    first_batch = next(iter(training_loader))

    print("First batch shape:", first_batch.shape)

    # 4. Create model
    model = LSTMAutoencoder(
        input_size=len(SENSORS),
        hidden_size=HIDDEN_SIZE,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    import numpy as np

    val_indices = np.linspace(
        0,
        len(val_dataset) - 1,
        min(512, len(val_dataset)),
        dtype=np.int64,
    ).tolist()

    validation_subset = Subset(
        val_dataset,
        val_indices,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    LOG_DIR = PROJECT_ROOT / "runs" / "lstm_time_validation"

    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # 5. Train model
    for epoch in range(NUM_EPOCHS):

        model.train()
        total_loss = 0.0

        for batch in training_loader:

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(training_loader)

        model.eval()

        validation_loss = 0.0

        with torch.inference_mode():

            for batch in validation_loader:
                reconstruction = model(batch)

                loss = criterion(
                    reconstruction,
                    batch
                )

                validation_loss += (
                        loss.item() * batch.size(0)
                )

        validation_loss /= len(validation_subset)

        writer.add_scalar(
            "Loss/Validation",
            validation_loss,
            epoch + 1
        )
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- Validation Loss: {validation_loss:.6f}"
        )

        writer.add_scalar(
            "Loss/Training",
            average_loss,
            epoch + 1
        )
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- Training Loss: {average_loss:.6f}"
        )
    writer.close()

    print("TensorBoard logs saved to:", LOG_DIR)

    # 6. Save model and scaler
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "lstm_autoencoder_baseline.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(SENSORS),
            "hidden_size": HIDDEN_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "sensors": SENSORS,
            "epochs": NUM_EPOCHS,
        },
        model_path,
    )

    scaler_path = MODEL_DIR / "sensor_scaler.joblib"

    joblib.dump(scaler, scaler_path)

    print("\nModel saved to:", model_path)
    print("Scaler saved to:", scaler_path)


if __name__ == "__main__":
    main()
