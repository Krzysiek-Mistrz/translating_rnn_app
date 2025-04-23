import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder wraps an embedding layer, LSTM, and linear+softmax to predict tokens.
    """
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.softmax(self.fc_out(output.squeeze(0)))
        return prediction, hidden, cell