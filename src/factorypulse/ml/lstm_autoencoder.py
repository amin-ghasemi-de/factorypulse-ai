

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        sequence_length = x.size(1)

        # Encode the sensor sequence
        _, (hidden, _) = self.encoder(x)

        # Repeat the latent representation
        decoder_input = hidden[-1].unsqueeze(1).repeat(
            1, sequence_length, 1
        )

        # Reconstruct the sequence
        decoded, _ = self.decoder(decoder_input)

        reconstruction = self.output_layer(decoded)

        return reconstruction
