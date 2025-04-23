import random
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    """
    Combines Encoder and Decoder for end-to-end translation.
    """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 device: torch.device):
        super().__init__()
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions must match!"
        assert encoder.n_layers == decoder.n_layers, \
            "Number of layers must match!"
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs