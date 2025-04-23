import random
import math
import torch
import torch.optim as optim
from tqdm import tqdm
from models.seq2seq import Seq2Seq
from utils import epoch_time
import time

def train_epoch(model: Seq2Seq,
                iterator,
                optimizer,
                criterion,
                clip: float,
                device):
    model.train()
    epoch_loss = 0
    loop = tqdm(iterator, desc="Training", leave=False)
    for src, trg in loop:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def train(model, train_loader, valid_loader, criterion,
          optimizer, n_epochs, clip, device, save_path):
    best_valid = float('inf')
    train_losses, valid_losses = [], []
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader,
                                 optimizer, criterion, clip, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        mins, secs = epoch_time(start_time, time.time())
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1} | {mins}m {secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val Loss: {valid_loss:.3f}")
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    return train_losses, valid_losses

from evaluate import evaluate