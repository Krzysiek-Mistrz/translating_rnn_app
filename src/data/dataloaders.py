import os
from subprocess import call
from Multi30K_de_en_dataloader import get_translation_dataloaders

if not os.path.exists("Multi30K_de_en_dataloader.py"):
    call([
        "wget",
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "IBMSkillsNetwork-AI0205EN-SkillsNetwork/Multi30K_de_en_dataloader.py"
    ])

def get_loaders(batch_size: int = 32, flip: bool = False):
    """
    Returns (train_loader, valid_loader) for German→English.
    """
    train_loader, valid_loader = get_translation_dataloaders(
        batch_size=batch_size,
        flip=flip
    )
    return train_loader, valid_loader