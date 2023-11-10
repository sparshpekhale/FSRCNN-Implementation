from torch.utils.data import DataLoader
import lightning as pl
import mlflow

from model import FSRCNN
from dataset import DIV2KDataset

import config

def train():
    model = FSRCNN()

    train_ds = DIV2KDataset(config.train_dir, transform=config.transform)
    valid_ds = DIV2KDataset(config.valid_dir, transform=config.transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=False)

    trainer = pl.Trainer(max_epochs=config.EPOCHS)

    mlflow.pytorch.autolog()

    trainer.fit(model, train_loader, valid_loader, ckpt_path=config.ckpt)

if __name__ == '__main__':
    train()