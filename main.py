import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import v2

from model import build_model, freeze_backbone
from train import train_model
from preprocess import preprocess_data


def set_seed(seed: int = 42):
    """Make results as reproducible as possible across runs."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic flags (safe on CPU; on GPU some ops may error if non-deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("Warning: could not enable full deterministic algorithms:", e)


def run_model(unfrozen_layers, val_fraction=0.0):
    dataloaders, dataset_sizes = preprocess_data(val_fraction=val_fraction)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # unfrozen_layers = head only (0) or head + last N of the 12 encoder blocks.
    model = build_model(num_classes=100, pretrained=True)
    freeze_backbone(model, unfrozen_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW decouples weight decay from the gradient update
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=0.05,
    )

    num_epochs = 30
    # Cosine anneals the LR smoothly to ~0 over the whole run.
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # from https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_cutmix_mixup.html
    batch_mix = v2.RandomChoice([
        v2.MixUp(alpha=0.2, num_classes=100),
        v2.CutMix(alpha=1.0, num_classes=100),
    ])

    model = train_model(model, criterion, optimizer, scheduler,
                        num_epochs=num_epochs, batch_mix=batch_mix,
                        dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    # Keep the best validation checkpoint when validation is enabled,
    # otherwise keep the final epoch weights so predict.py can load them.
    torch.save(model.state_dict(), 'model.pth')
    print('Saved trained weights to model.pth')


if __name__ == "__main__":
    set_seed(42)
    run_model(1, val_fraction=0.0)
