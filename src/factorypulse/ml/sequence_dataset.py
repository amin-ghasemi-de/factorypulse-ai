
import numpy as np
import torch
from torch.utils.data import Dataset


class CompressorSequenceDataset(Dataset):

    def __init__(
        self,
        data,
        segment_ids,
        source_indices,
        sequence_length=60
    ):
        self.sequence_length = sequence_length

        self.data = torch.from_numpy(
            np.asarray(data, dtype=np.float32)
        )

        segment_ids = np.asarray(segment_ids)
        source_indices = np.asarray(source_indices)

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        if not (
            len(self.data)
            == len(segment_ids)
            == len(source_indices)
        ):
            raise ValueError("Input lengths must match")

        # Detect breaks in the original time series
        breaks = np.where(
            (segment_ids[1:] != segment_ids[:-1])
            | (np.diff(source_indices) != 1)
        )[0] + 1

        group_starts = np.r_[0, breaks]
        group_ends = np.r_[breaks, len(self.data)]

        # Keep valid sequence starting positions only
        valid_starts = []

        for start, end in zip(group_starts, group_ends):
            if end - start >= sequence_length:
                valid_starts.append(
                    np.arange(
                        start,
                        end - sequence_length + 1
                    )
                )

        self.start_positions = (
            np.concatenate(valid_starts)
            if valid_starts
            else np.array([], dtype=np.int64)
        )

    def __len__(self):
        return len(self.start_positions)

    def __getitem__(self, idx):
        start = self.start_positions[idx]
        end = start + self.sequence_length

        return self.data[start:end]
