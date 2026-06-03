from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from histological_cancer_analysis.constants import (
    BINARY_OUTPUT_SIZE,
    IMAGE_CHANNELS,
    ModelName,
)


class SimpleCNN(nn.Module):
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, BINARY_OUTPUT_SIZE),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_resnet18(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, BINARY_OUTPUT_SIZE)
    return model


def build_model(name: str, dropout: float = 0.25, pretrained: bool = False) -> nn.Module:
    if name == ModelName.SIMPLE_CNN:
        return SimpleCNN(dropout=dropout)
    if name == ModelName.RESNET18:
        return build_resnet18(pretrained=pretrained)

    msg = f"Unsupported model name: {name}"
    raise ValueError(msg)
