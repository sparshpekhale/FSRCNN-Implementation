from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import nn
import lightning as pl
from torch.nn import functional as F

import config

class FSRCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.d, self.s, self.m = 56, 12, 4
        
        self.model = nn.Sequential(
            nn.Conv2d(3, self.d, kernel_size=5, padding=5//2),
            nn.ReLU(),
            nn.Conv2d(self.d, self.s, kernel_size=1),
            nn.ReLU()
        )
        
        for _ in range(self.m):
            self.model.append(nn.Conv2d(self.s, self.s, kernel_size=3, padding=3//2)),
            self.model.append(nn.ReLU())
            
        self.model.append(nn.Conv2d(self.s, self.d, kernel_size=1))
        self.model.append(nn.ReLU())
        
        self.model.append(nn.ConvTranspose2d(self.d, 3, kernel_size=9,stride=config.SCALING_FACTOR,
                                          padding=9//2, output_padding=config.SCALING_FACTOR-1))
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        
        self.log('train_mse', loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(), lr=config.LEARNING_RATE)
    
