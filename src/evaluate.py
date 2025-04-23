import torch
from tqdm import tqdm
from models.seq2seq import Seq2Seq

def evaluate(model: Seq2Seq, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    loop = tqdm(iterator, desc="Evaluating", leave=False)
    with torch.no_grad():
        for src, trg in loop:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loop.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)