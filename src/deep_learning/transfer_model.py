import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


def build_transfer_model(
    model_name: str = 'resnet18',
    num_classes: int = 4,
    pretrained: bool = True
) -> nn.Module:
    """
    Build transfer learning model with modified classification head.

    Args:
        model_name: Backbone architecture ('resnet18').
        num_classes: Number of target classes.
        pretrained: Whether to load ImageNet pre-trained weights.
    """
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model
