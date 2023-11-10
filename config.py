import torch
from torchvision import transforms

train_dir = 'data/DIV2K_train_HR/'
valid_dir = 'data/DIV2K_valid_HR/'

SCALING_FACTOR = 4
CROP_DIM = 648

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.6, 0.1)
])

BATCH_SIZE = 16
EPOCHS = 500
LEARNING_RATE = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckpt = None