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
from train import train_model
from preprocess import preprocess_data

def set_seed(seed: int = 42):
    """Make results as reproducible as possible across runs."""
    import os, random
    import numpy as np
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # If you later switch to CUDA and want maximal determinism:
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic flags (safe on CPU; on GPU some ops may error if non-deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("Warning: could not enable full deterministic algorithms:", e)


def run_model(unfrozen_layers):
    dataloaders, dataset_sizes = preprocess_data()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")  

    # How many of the trailing transformer encoder blocks to fine-tune.
    # 0 = classification head only (most regularized; good for tiny data).
    # N = head + last N of the 12 encoder blocks. ViT-B/16 has 12 blocks total.
    UNFROZEN_BACKBONE_LAYERS = unfrozen_layers

    model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 100)
    )

    # Freeze everything, then re-enable the head + last N encoder blocks.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True
    if UNFROZEN_BACKBONE_LAYERS > 0:
        for block in model.encoder.layers[-UNFROZEN_BACKBONE_LAYERS:]:
            for param in block.parameters():
                param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer_conv = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=15, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)


if __name__ == "__main__":
    set_seed(42)
    for i in range(3):
        run_model(i)