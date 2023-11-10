from torch.utils.data import DataLoader
import torch
import lightning as pl

from model import FSRCNN
from dataset import DIV2KDataset

import config

import argparse

import matplotlib.pyplot as plt

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', help='Checkpoint path', required=True)
    args = vars(parser.parse_args())
    
    test_ds = DIV2KDataset(config.valid_dir, transform=config.transform)
    test_loader = DataLoader(test_ds, batch_size=2)

    # model = FSRCNN()
    # trainer = pl.Trainer(inference_mode=True)
    # trainer.test(model, test_loader, ckpt_path=args['ckpt'])

    model = FSRCNN()
    checkpoint = torch.load(args['ckpt'])
    model.load_state_dict(checkpoint['state_dict'])
    
    for lr, hr in test_loader:
        sr = model(lr).detach()

        lr += 0.5
        hr += 0.5

        lo, hi = sr.min(), sr.max()
        sr = (sr-lo) / (hi-lo)
        
        plt.imshow(hr[0].permute(1, 2, 0).numpy())
        plt.show()
        
        plt.imshow(sr[0].permute(1, 2, 0).numpy())
        plt.show()

        break


if __name__ == '__main__':
    test()