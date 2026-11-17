"""Model-contract wiring tests (SmallCNN adoption support).

These tests pin the SmallCNN/ResNet channel contract: the shared
build_model() helper must honor in_channels, the 13-band multispectral path
must run a real forward pass, ResNet18 stays available but 3-channel until
the planned M3-T03 adaptation, and evaluation's validate_checkpoint helper
must reject checkpoints from a different model/channel/class contract.

The ResNet18 tests run fully offline: the torchvision constructor is
monkeypatched so no pretrained-weight download can mask behavior.
"""

import pytest
import torch
import torch.nn as nn

import src.model as model_module
from src.evaluate import validate_checkpoint
from src.model import SmallCNN, build_model

NUM_CLASSES = 10
CLASSES = [f"c{i}" for i in range(NUM_CLASSES)]


def _checkpoint(model, num_channels=13, classes=None):
    return {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "classes": classes if classes is not None else list(CLASSES),
        "num_channels": num_channels,
        "model_name": "SmallCNN",
    }


def test_build_model_cnn_honors_in_channels():
    model = build_model("cnn", NUM_CLASSES, in_channels=13)
    assert model.features[0][0].in_channels == 13
    out = model(torch.rand(2, 13, 64, 64))
    assert out.shape == (2, NUM_CLASSES)


def test_build_model_default_stays_three_channel_rgb():
    # Backward compatibility for the RGB entry point (root train.py).
    model = build_model("cnn", NUM_CLASSES)
    assert model.features[0][0].in_channels == 3
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, NUM_CLASSES)


def test_smallcnn_forward_thirteen_band():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    assert sum(p.numel() for p in model.parameters()) == 292074
    out = model(torch.rand(4, 13, 64, 64))
    assert out.shape == (4, NUM_CLASSES)


class _FakeResNet(nn.Module):
    """Minimal stand-in with the attributes build_model touches."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
        self.fc = nn.Linear(512, 1000)


def test_resnet18_preserved_offline(monkeypatch):
    requested = {}

    def fake_resnet18(weights=None):
        requested["weights"] = weights
        return _FakeResNet()

    monkeypatch.setattr(model_module.models, "resnet18", fake_resnet18)
    model = build_model("resnet18", NUM_CLASSES)
    assert model.conv1.in_channels == 3
    assert model.fc.out_features == NUM_CLASSES
    # ImageNet weights were requested from the (faked) constructor —
    # no network download happened in this test.
    assert requested["weights"] is not None
    assert "IMAGENET1K" in str(requested["weights"])


def test_resnet18_rejects_thirteen_channels_offline():
    # Raises before any constructor/download attempt.
    with pytest.raises(ValueError, match="3-channel"):
        build_model("resnet18", NUM_CLASSES, in_channels=13)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("vgg", NUM_CLASSES)


def test_validate_checkpoint_accepts_matching_checkpoint():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    checkpoint = _checkpoint(model)
    # Must not raise.
    validate_checkpoint(
        checkpoint, model, num_channels=13, model_name="SmallCNN",
        classes=CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 13, 64, 64))
    assert out.shape == (1, NUM_CLASSES)


def test_validate_checkpoint_rejects_wrong_channel_count():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    checkpoint = _checkpoint(model, num_channels=3)
    with pytest.raises(RuntimeError, match="num_channels"):
        validate_checkpoint(
            checkpoint, model, num_channels=13, model_name="SmallCNN",
            classes=CLASSES,
        )


def test_validate_checkpoint_rejects_wrong_model_name():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    checkpoint = _checkpoint(model)
    checkpoint["model_name"] = "ResNet18"
    with pytest.raises(RuntimeError, match="model_name"):
        validate_checkpoint(
            checkpoint, model, num_channels=13, model_name="SmallCNN",
            classes=CLASSES,
        )


def test_validate_checkpoint_rejects_shape_mismatch():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    rgb_model = SmallCNN(NUM_CLASSES, in_channels=3)
    checkpoint = _checkpoint(rgb_model, num_channels=None)
    checkpoint.pop("num_channels")  # force fallback to shape checking
    with pytest.raises(RuntimeError, match="shape"):
        validate_checkpoint(
            checkpoint, model, num_channels=13, model_name="SmallCNN",
            classes=CLASSES,
        )


def test_validate_checkpoint_rejects_wrong_class_order():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    shuffled = list(reversed(CLASSES))
    checkpoint = _checkpoint(model, classes=shuffled)
    with pytest.raises(RuntimeError, match="class list"):
        validate_checkpoint(
            checkpoint, model, num_channels=13, model_name="SmallCNN",
            classes=CLASSES,
        )


def test_validate_checkpoint_rejects_wrong_class_count():
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    checkpoint = _checkpoint(model, classes=CLASSES[:-1])
    with pytest.raises(RuntimeError, match="class list"):
        validate_checkpoint(
            checkpoint, model, num_channels=13, model_name="SmallCNN",
            classes=CLASSES,
        )


def test_train_checkpoint_roundtrip_through_real_guard(tmp_path):
    # Mirror src/train.py's save and src/evaluate.py's actual load path.
    model = SmallCNN(NUM_CLASSES, in_channels=13)
    path = tmp_path / "best_model.pth"
    torch.save(_checkpoint(model), path)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = SmallCNN(
        len(loaded["classes"]), in_channels=loaded["num_channels"]
    )
    validate_checkpoint(
        loaded, rebuilt, num_channels=13, model_name="SmallCNN",
        classes=CLASSES,
    )
    rebuilt.load_state_dict(loaded["model_state_dict"])
    rebuilt.eval()
    with torch.no_grad():
        out = rebuilt(torch.rand(1, 13, 64, 64))
    assert out.shape == (1, NUM_CLASSES)
