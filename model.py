import torch
from torch import nn
import lightning as pl
from torch.nn import functional as F

from utils import MSE_PSNR
import config

class FSRCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.d, self.s, self.m = 32, 2, 2
        # self.d, self.s, self.m = 56, 12, 4
        
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
        

        self.mse_psnr = MSE_PSNR()

        # Logging
        self.losses = []
        self.val_losses = []

        self.psnr = []
        self.val_psnr = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        # mse = F.mse_loss(logits, y)
        mse, psnr = self.mse_psnr(logits, y)
        self.losses.append(mse)
        self.psnr.append(psnr)
        return mse
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.losses).mean()
        self.log('train_mse', avg_loss, prog_bar=True)
        self.losses.clear()

        avg_psnr = torch.stack(self.psnr).mean()
        self.log('train_psnr', avg_psnr, prog_bar=True)
        self.psnr.clear()
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        # mse = F.mse_loss(logits, y)
        mse, psnr = self.mse_psnr(logits, y)
        self.val_losses.append(mse)
        self.val_psnr.append(psnr)
        return mse
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('val_mse', avg_loss)
        self.val_losses.clear()

        avg_psnr = torch.stack(self.val_psnr).mean()
        self.log('val_psnr', avg_psnr, prog_bar=True)
        self.val_psnr.clear()
    
    def configure_optimizers(self):
        return torch.optim.RMSprop(self.model.parameters(), lr=config.LEARNING_RATE)
    
