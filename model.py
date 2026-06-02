import torch.nn as nn
import torchvision


def build_model(num_classes=100, pretrained=True):
    """Build the SWAG ViT-B/16 with our classification head.
    """
    if pretrained:
        model = torchvision.models.vit_b_16(weights="IMAGENET1K_SWAG_E2E_V1")
    else:
        model = torchvision.models.vit_b_16(weights=None, image_size=384)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model, unfrozen_layers):
    """Freeze everything, then re-enable the head + last unfrozen_layers

    ViT-B/16 has 12 encoder blocks. unfrozen_layers=0 trains the head only.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True
    if unfrozen_layers > 0:
        for block in model.encoder.layers[-unfrozen_layers:]:
            for param in block.parameters():
                param.requires_grad = True
    return model
