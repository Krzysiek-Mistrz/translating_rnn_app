import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder wraps an embedding layer and a multi‐layer LSTM.
    """
    def __init__(self, input_dim: int, emb_dim: int, hid_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell