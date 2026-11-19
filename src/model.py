from dataclasses import dataclass

import torch.nn as nn
from torchvision import models
from torchvision.models import vgg as torchvision_vgg


@dataclass(frozen=True)
class VGG16Config:
    classifier_hidden: int = 512
    dropout: float = 0.5
    pool_size: int = 1

    def __post_init__(self):
        if self.classifier_hidden < 1:
            raise ValueError("classifier_hidden must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.pool_size < 1:
            raise ValueError("pool_size must be positive.")


class SmallCNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class VGG16Classifier(nn.Module):
    """Torchvision VGG16 feature stack with a configurable classification head."""

    def __init__(self, num_classes, config):
        super().__init__()
        self.features = torchvision_vgg.make_layers(torchvision_vgg.cfgs["D"])
        self.avgpool = nn.AdaptiveAvgPool2d((config.pool_size, config.pool_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                512 * config.pool_size * config.pool_size,
                config.classifier_hidden,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))


def _load_vgg16_feature_weights(model):
    weights_enum = getattr(models, "VGG16_Weights", None)
    if weights_enum is None:
        raise RuntimeError(
            "Pretrained VGG16 requires torchvision's VGG16_Weights API. "
            "Upgrade torchvision to use ImageNet weights."
        )

    state_dict = weights_enum.DEFAULT.get_state_dict(progress=True, check_hash=True)
    feature_state = {
        key.removeprefix("features."): value
        for key, value in state_dict.items()
        if key.startswith("features.")
    }
    model.features.load_state_dict(feature_state)
    del feature_state, state_dict


def _freeze_feature_extractor(model, head_prefix):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(head_prefix)
    model._feature_extractor_frozen = True


def build_model(
    name,
    num_classes,
    in_channels=3,
    *,
    pretrained=None,
    freeze_features=False,
    vgg_config=None,
):
    """Build a classifier while preserving the RGB and 13-band CNN contract.

    ``pretrained=None`` retains the historical ResNet-18 behavior of attempting
    ImageNet initialization and falling back to scratch weights. Pass an
    explicit boolean to select scratch or transfer learning without fallback.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    if pretrained not in (None, True, False):
        raise TypeError("pretrained must be True, False, or None.")
    if freeze_features and pretrained is not True:
        raise ValueError("freeze_features requires pretrained=True.")

    name = name.lower()
    if name == "cnn":
        if pretrained is True or freeze_features:
            raise ValueError("The custom CNN does not support pretrained weights.")
        return SmallCNN(num_classes, in_channels=in_channels)

    if name == "resnet18":
        if in_channels != 3:
            raise ValueError(
                "resnet18 currently supports 3-channel input only; "
                "the 13-band first-conv adaptation is planned in M3-T03."
            )
        weights_enum = getattr(models, "ResNet18_Weights", None)
        if pretrained is None:
            try:
                weights = (
                    weights_enum.IMAGENET1K_V1 if weights_enum is not None else None
                )
                model = models.resnet18(weights=weights)
            except Exception:
                model = models.resnet18(weights=None)
        elif weights_enum is None:
            model = models.resnet18(pretrained=pretrained)
        else:
            model = models.resnet18(
                weights=weights_enum.DEFAULT if pretrained else None
            )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_features:
            _freeze_feature_extractor(model, "fc.")
        return model

    if name == "vgg16":
        if in_channels != 3:
            raise ValueError("vgg16 supports 3-channel input only.")
        if vgg_config is None:
            vgg_config = VGG16Config(pool_size=7 if pretrained else 1)
        if not isinstance(vgg_config, VGG16Config):
            raise TypeError("vgg_config must be a VGG16Config instance.")
        model = VGG16Classifier(num_classes, vgg_config)
        if pretrained:
            _load_vgg16_feature_weights(model)
        if freeze_features:
            _freeze_feature_extractor(model, "classifier.")
        return model

    raise ValueError(f"unknown model: {name}")
