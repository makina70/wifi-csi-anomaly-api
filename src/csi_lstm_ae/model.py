from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        latent_size: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.to_decoder_hidden = nn.Linear(latent_size, hidden_size * num_layers)
        self.to_decoder_cell = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        encoded = hidden[-1]
        latent = self.to_latent(encoded)

        batch_size = x.shape[0]
        decoder_hidden = self.to_decoder_hidden(latent).view(
            self.num_layers, batch_size, self.hidden_size
        )
        decoder_cell = self.to_decoder_cell(latent).view(
            self.num_layers, batch_size, self.hidden_size
        )
        decoder_input = torch.zeros_like(x)
        decoded, _ = self.decoder(decoder_input, (decoder_hidden.contiguous(), decoder_cell.contiguous()))
        return self.output(decoded)


class FeatureAutoencoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
