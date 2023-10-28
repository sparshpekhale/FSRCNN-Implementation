import os
import time

# import torch
# from torch import nn

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import config

class DIV2KDataset(Dataset):
    def __init__(self, path, transform=config.transform):
        self.hr_dir = path
        self.img_list = os.listdir(path)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        hr = Image.open(self.hr_dir+self.img_list[idx])
        
        # Randomly crop image
        w, h = hr.size
        rand_w = np.random.randint(0, w-config.CROP_DIM)
        rand_h = np.random.randint(0, h-config.CROP_DIM)
        
        hr = hr.crop((rand_w, rand_h, rand_w+config.CROP_DIM, rand_h+config.CROP_DIM))
        
        # Transforms
        if self.transform:
            hr = self.transform(hr)
        
        # Resize to create LR image
        lr = hr.resize((config.CROP_DIM//config.SCALING_FACTOR, config.CROP_DIM//config.SCALING_FACTOR), Image.BICUBIC)
#         display(hr)
#         display(lr)
        
        # Images to tensors
        hr = transforms.ToTensor()(hr) - 0.5
        lr = transforms.ToTensor()(lr) - 0.5
        
        return lr, hr


if __name__ == '__main__':

    ds = DIV2KDataset(config.train_dir)
    for lr, hr in ds:
        plt.imshow(hr.permute(1, 2, 0).detach().numpy()+0.5)
        plt.show()
        plt.imshow(lr.permute(1, 2, 0).detach().numpy()+0.5)
        plt.show()
        break