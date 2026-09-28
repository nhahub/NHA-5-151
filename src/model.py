import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, num_classes):
    if name == "cnn":
        return SmallCNN(num_classes)
    if name == "resnet18":
        resnet = getattr(models, "ResNet18_Weights", None)
        try:
            weights = resnet.IMAGENET1K_V1 if resnet is not None else None
            m = models.resnet18(weights=weights)
        except Exception:
            m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown model: {name}")
