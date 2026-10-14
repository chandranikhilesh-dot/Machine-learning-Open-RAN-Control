"""Compact educational TFT-style multi-KPI forecaster.
Uses an LSTM encoder, multi-head attention and quantile outputs.
This is a student prototype, not the full production TFT implementation.
"""
import torch
from torch import nn

class CompactTFT(nn.Module):
    def __init__(self, n_features=5, hidden=48, horizon=12, quantiles=3):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.input_projection = nn.Linear(n_features, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, horizon * n_features * quantiles)
        )
        self.n_features = n_features

    def forward(self, x):
        z = self.input_projection(x)
        z, _ = self.encoder(z)
        attended, _ = self.attention(z, z, z, need_weights=False)
        z = self.norm(z + attended)
        last = z[:, -1, :]
        out = self.head(last)
        return out.view(-1, self.horizon, self.n_features, self.quantiles)
