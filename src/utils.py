import time
import torch
import torch.nn as nn

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def epoch_time(start_time: float, end_time: float):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs