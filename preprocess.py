# Structure adapted from the PyTorch transfer-learning tutorial
# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os
from PIL import Image
import torch
from torchvision import datasets, transforms


class UnlabelledDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith('.jpg')
        )
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.paths[idx]

    def __len__(self):
        return len(self.paths)


def preprocess_data(val_fraction=0.0):
    # Augmentation + normalization for train but just resize/normalize for val/test.
    # SWAG ViT-B/16 expects 384px input with BICUBIC interpolation.
    # Normalization is the standard ImageNet mean/std
    bicubic = transforms.InterpolationMode.BICUBIC

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.7, 1.0), interpolation=bicubic),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        # Squash the whole image to 384x384
        transforms.Resize((384, 384), interpolation=bicubic),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'data'

    # Two copies of train folder: one with augmentations, one without
    # We split indices and apply each subset to the correct copy
    train_full_aug = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    train_full_eval = datasets.ImageFolder(os.path.join(data_dir, 'train'), eval_transform)

    image_datasets = {
        'test': UnlabelledDataset(os.path.join(data_dir, 'test'), eval_transform),
    }

    if val_fraction > 0.0:
        n_total = len(train_full_aug)
        n_val = int(n_total * val_fraction)
        n_train = n_total - n_val
        train_indices, val_indices = torch.utils.data.random_split(
            range(n_total), [n_train, n_val]
        )

        image_datasets['train'] = torch.utils.data.Subset(train_full_aug, train_indices.indices)
        image_datasets['val'] = torch.utils.data.Subset(train_full_eval, val_indices.indices)
    else:
        image_datasets['train'] = train_full_aug

    dataloaders = {
        split: torch.utils.data.DataLoader(
            image_datasets[split],
            batch_size=16,
            shuffle=(split == 'train'),
            num_workers=4,
        )
        for split in image_datasets
    }
    dataset_sizes = {split: len(image_datasets[split]) for split in image_datasets}

    return dataloaders, dataset_sizes
