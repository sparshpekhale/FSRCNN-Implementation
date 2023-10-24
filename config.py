import torch
from torchvision import transforms

hr_dir = 'DIV2K_valid_HR/'

SCALING_FACTOR = 4
CROP_DIM = 648

transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.6, 0.1)
])

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
