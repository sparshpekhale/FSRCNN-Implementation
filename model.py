from math import sqrt

import torch
from torch import nn
import lightning as pl
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from utils import MSE_PSNR
import config

class FSRCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # self.d, self.s, self.m = 32, 2, 2
        self.d, self.s, self.m = 56, 12, 4
        
        self.model = nn.Sequential(
            nn.Conv2d(3, self.d, kernel_size=5, padding=5//2),
            nn.PReLU(self.d),
            nn.Conv2d(self.d, self.s, kernel_size=1),
            nn.PReLU(self.s)
        )
        
        for _ in range(self.m):
            self.model.append(nn.Conv2d(self.s, self.s, kernel_size=3, padding=3//2)),
            self.model.append(nn.PReLU(self.s))
            
        self.model.append(nn.Conv2d(self.s, self.d, kernel_size=1))
        self.model.append(nn.PReLU(self.d))
        
        self.deconv = nn.ConvTranspose2d(self.d, 3, kernel_size=9,stride=config.SCALING_FACTOR,
                            padding=9//2, output_padding=config.SCALING_FACTOR-1)
        
        # Init weights
        self.init_weights()

        # Metrics
        self.mse_psnr = MSE_PSNR()

        # Logging
        self.losses = []
        self.val_losses = []

        self.psnr = []
        self.val_psnr = []
        
    def forward(self, x):
        x = self.model(x)
        x = self.deconv(x)
        # x = F.sigmoid(x) / 2.
        return x
    
    # INIT WEIGHTS
    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # Conv Transpose -> Gaussian, mean=0, std=0.001
                nn.init.normal_(m.weight.data, mean=0, std=0.001)
                nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.Conv2d):
                # Conv -> Kaiming He, mean=0, std=sqrt(2/num_nodes)
                nn.init.normal_(m.weight.data, mean=0, std=sqrt(2/m.weight.data.numel()))
                nn.init.zeros_(m.bias.data)

    # TRAINING
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
        
    # VALIDATION
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
        self.log('val_mse', avg_loss, prog_bar=True)
        self.val_losses.clear()

        avg_psnr = torch.stack(self.val_psnr).mean()
        self.log('val_psnr', avg_psnr, prog_bar=True)
        self.val_psnr.clear()

    # TODO: TESTING
    
    # UTILS
    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            [
                {"params": self.model.parameters(), "lr": config.LEARNING_RATE},
                {"params": self.deconv.parameters(), "lr": config.DECONV_LEARNING_RATE}
            ],
            lr=config.LEARNING_RATE
        )
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=20, min_lr=10e-6, verbose=True)

        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_mse'
            }
        }

    def configure_callbacks(self):
        early_stopping =  EarlyStopping(
            monitor='val_mse',
            min_delta=10e-6,
            patience=40,
            verbose=True
        )

        checkpoints = ModelCheckpoint(
            './checkpoints',
            every_n_epochs=3,
            filename='{epoch}-{val_psnr:.2f}',
            monitor='val_psnr',
            mode='max'
        )

        return [early_stopping, checkpoints]
    

if __name__ == '__main__':
    x = torch.randn(4, 3, config.CROP_DIM//config.SCALING_FACTOR, config.CROP_DIM//config.SCALING_FACTOR)
    
    model = FSRCNN()
    logits = model.forward(x).detach()

    print('x:', x.shape, 'logits:', logits.shape)