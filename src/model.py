"""Transfer learning model factory."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("resnet18", "resnet50", "efficientnet_b0", "densenet121")


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Return an ImageNet backbone with a new classification head."""
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes))
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED_MODELS}")
    return model
