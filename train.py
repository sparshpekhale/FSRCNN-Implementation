from torch.utils.data import DataLoader, random_split
import lightning as pl
import mlflow

from model import FSRCNN
from dataset import DIV2KDataset

import config


model = FSRCNN()

train_ds = DIV2KDataset(config.hr_dir, transform=config.transform)
valid_ds = DIV2KDataset(config.hr_dir, transform=config.transform)

train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, num_workers=6, shuffle=False)

trainer = pl.Trainer(max_epochs=4)

mlflow.pytorch.autolog()

trainer.fit(model, train_loader, valid_loader)