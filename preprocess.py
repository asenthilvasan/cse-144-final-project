# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

class UnlabelledDataset(torch.utils.data.Dataset):
    EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    def __init__(self, folder, transform=None):
        self.paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(self.EXTENSIONS)
        )
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]

    def __len__(self):
        return len(self.paths)


def preprocess_data():
    cudnn.benchmark = True
    plt.ion()   # interactive mode


    # Data augmentation and normalization for training
    # Just normalization for validation

    # SWAG ViT-B/16 expects 384px input with BICUBIC interpolation.
    # Normalization is the standard ImageNet mean/std (same as before).
    bicubic = transforms.InterpolationMode.BICUBIC

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.7, 1.0), interpolation=bicubic),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=bicubic),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'data'

    # Two copies of train folder: one with augmentations, one without
    # We split indices and apply each subset to the appropriate copy
    train_full_aug = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    train_full_eval = datasets.ImageFolder(os.path.join(data_dir, 'train'), eval_transform)

    val_fraction = 0.2
    n_total = len(train_full_aug)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    train_indices, val_indices = torch.utils.data.random_split(
        range(n_total), [n_train, n_val]
    )

    image_datasets = {
        'train': torch.utils.data.Subset(train_full_aug, train_indices.indices),
        'val':   torch.utils.data.Subset(train_full_eval, val_indices.indices),
        'test':  UnlabelledDataset(os.path.join(data_dir, 'test'), eval_transform),
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                                shuffle=(x == 'train'), num_workers=4)
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = train_full_aug.classes
    
    return dataloaders, dataset_sizes