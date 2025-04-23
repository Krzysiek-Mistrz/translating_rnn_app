import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from data.dataloaders import get_loaders
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from utils import init_weights, count_parameters
from train import train
from translate import generate_translation

def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Translator")
    parser.add_argument('--mode', choices=['train', 'translate'], required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-path', type=str, default='model.pt')
    parser.add_argument('--sentence', type=str, help="Sentence to translate")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, valid_loader = get_loaders(batch_size=args.batch_size)
    INPUT_DIM = train_loader.dataset.vocab_size['de']
    OUTPUT_DIM = train_loader.dataset.vocab_size['en']
    HID_DIM, EMB_DIM = 256, 128
    N_LAYERS, DROPOUT = 1, 0.3
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(f"Model has {count_parameters(model):,} parameters")
    optimizer = optim.Adam(model.parameters())
    PAD_IDX = train_loader.dataset.vocab_transform['en'].get_stoi()['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    if args.mode == 'train':
        train(model, train_loader, valid_loader, criterion,
              optimizer, args.epochs, clip=1, device=device,
              save_path=args.model_path)
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        sentence = args.sentence or input("Enter sentence to translate: ")
        translation = generate_translation(
            model, sentence,
            train_loader.dataset.text_transform,
            train_loader.dataset.vocab_transform['de'],
            train_loader.dataset.vocab_transform['en'],
            device
        )
        print("Translation:", translation)

if __name__ == "__main__":
    main()