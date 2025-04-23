# Seq2Seq German–English Translator

A simple neural machine translation project using PyTorch’s encoder–decoder architecture with LSTM and teacher forcing. Trains on the Multi30K dataset (German→English) and supports both training and inference modes via a command‑line interface.

## Features
- **Encoder–Decoder with LSTM**: Multi‑layer LSTM encoder and decoder modules.
- **Teacher Forcing**: Configurable ratio for more stable training.
- **Model Checkpointing**: Automatically saves best validation model.
- **Inference**: Generate translations of new sentences.
- **Custom DataLoader**: Prepares and tokenizes Multi30K dataset.

## Requirements
- Python 3.7+
- PyTorch 1.10+
- torchvision (for dataset utilities)
- tqdm
- wget (to fetch the data loader script)

Install dependencies with:
```bash
pip install torch torchvision tqdm
```
*Also those from requirements!*

## Installation & Setup
1. Clone the repository.
2. Ensure `Multi30K_de_en_dataloader.py` is present (the script auto‑downloads if missing, just run main.py).
3. Install Python dependencies.

## Usage

### Training
```bash
python main.py --mode train \
    --batch-size 32 \
    --epochs 10 \
    --model-path model.pt
```
Trains the model, prints epoch losses, and saves the best model to `model.pt`.

### Translating
```bash
python main.py --mode translate \
    --model-path model.pt \
    --sentence "Ein kleines Mädchen spielt im Park."
```
Loads the saved model and outputs the English translation.

## File Overview
- **Multi30K_de_en_dataloader.py**: Downloads and defines `get_translation_dataloaders` to produce PyTorch DataLoaders for training/validation.
- **data/dataloaders.py** (`get_loaders`): Wrapper that returns train and valid loaders with chosen batch size and flip option.
- **models/encoder.py**: `Encoder` class – embeds and encodes source sequences into hidden states.
- **models/decoder.py**: `Decoder` class – embeds inputs, runs LSTM, projects to vocabulary distribution.
- **models/seq2seq.py**: `Seq2Seq` class – orchestrates encoding and step‑by‑step decoding with teacher forcing.
- **train.py**: `train_epoch` and `train` functions – perform training loops, gradient clipping, validation, and checkpointing.
- **evaluate.py**: `evaluate` function – runs model in eval mode and computes validation loss.
- **translate.py**: `generate_translation` – uses the trained model for greedy decoding of new sentences.
- **utils.py**: Helper functions for weight initialization, parameter counting, and timing epochs.
- **main.py**: Command‑line interface; sets up model, optimizer, criterion, and dispatches to training or translation.

## Hyperparameters
- **Embedding dim**: 128
- **Hidden dim**: 256
- **Layers**: 1
- **Dropout**: 0.3
- **Batch size**: 32
- **Epochs**: configurable (default 10)
- **Learning rate**: Adam default

## Contributing
Feel free to open issues or submit pull requests for improvements (e.g., beam search, attention mechanism).

## License
GNU GPL V3 © 2025 Krzychu

