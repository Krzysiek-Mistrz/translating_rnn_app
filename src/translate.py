import torch
from torch import nn

def generate_translation(model: nn.Module,
                         src_sentence: str,
                         text_transform,
                         src_vocab,
                         trg_vocab,
                         device: torch.device,
                         max_len: int = 50):
    model.eval()
    tokens = text_transform[src_vocab["lang"]](src_sentence)
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    hidden, cell = model.encoder(src_tensor)
    trg_indexes = [trg_vocab.get_stoi()['<bos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.get_stoi()['<eos>']:
            break
    trg_tokens = [trg_vocab.get_itos()[i] for i in trg_indexes]
    if trg_tokens and trg_tokens[0] == '<bos>':
        trg_tokens = trg_tokens[1:]
    if trg_tokens and trg_tokens[-1] == '<eos>':
        trg_tokens = trg_tokens[:-1]
    return " ".join(trg_tokens)