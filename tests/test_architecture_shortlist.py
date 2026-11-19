import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

import src.architecture_experiments as experiments
import src.model as model_module
from src.dataset import (
    get_loaders,
    load_rgb_split_manifest,
    make_stratified_splits,
)
from src.model import VGG16Config, build_model


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


@pytest.mark.parametrize("name", ["cnn", "resnet18", "vgg16"])
def test_scratch_architectures_train_one_batch_and_return_ten_logits(name):
    model = build_model(
        name,
        num_classes=10,
        vgg_config=VGG16Config(classifier_hidden=32, dropout=0.1, pool_size=1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 9], dtype=torch.long)

    logits = model(images)
    assert logits.shape == (2, 10)
    loss = nn.CrossEntropyLoss()(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def test_pretrained_vgg_freezes_only_feature_extractor(monkeypatch):
    monkeypatch.setattr(
        model_module, "_load_vgg16_feature_weights", lambda _model: None
    )
    model = build_model(
        "vgg16",
        num_classes=10,
        pretrained=True,
        freeze_features=True,
        vgg_config=VGG16Config(classifier_hidden=16, pool_size=1),
    )

    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_pretrained_resnet_uses_requested_weights_and_freezes_features(monkeypatch):
    weights_enum = getattr(model_module.models, "ResNet18_Weights", None)
    if weights_enum is None:
        pytest.skip("This torchvision version does not expose weights enums.")
    original_resnet18 = model_module.models.resnet18
    requested = {}

    def fake_resnet18(*, weights):
        requested["weights"] = weights
        return original_resnet18(weights=None)

    monkeypatch.setattr(model_module.models, "resnet18", fake_resnet18)
    model = build_model(
        "resnet18",
        num_classes=10,
        pretrained=True,
        freeze_features=True,
    )

    assert requested["weights"] is weights_enum.DEFAULT
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_pretrained_loader_preserves_split_and_uses_imagenet_input_size(tmp_path):
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for sample_index in range(10):
            image = np.full(
                (16, 16, 3),
                fill_value=(class_index * 10 + sample_index) % 255,
                dtype=np.uint8,
            )
            Image.fromarray(image).save(class_dir / f"{sample_index}.jpg")

    dataset = datasets.ImageFolder(str(tmp_path))
    splits = make_stratified_splits(dataset.targets, seed=42)
    train_loader, val_loader, test_loader, classes = get_loaders(
        tmp_path,
        batch_size=10,
        seed=42,
        num_workers=0,
        split_indices=splits,
        pretrained=True,
    )

    assert classes == CLASS_NAMES
    assert [len(loader.dataset) for loader in (train_loader, val_loader, test_loader)] == [
        len(indices) for indices in splits
    ]
    images, _ = next(iter(val_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert train_loader.dataset.indices == [int(index) for index in splits[0]]


def test_rgb_split_manifests_are_loaded_and_validated(tmp_path):
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for sample_index in range(10):
            image = np.full((8, 8, 3), class_index, dtype=np.uint8)
            Image.fromarray(image).save(class_dir / f"{sample_index}.jpg")

    dataset = datasets.ImageFolder(str(tmp_path))
    expected_splits = make_stratified_splits(dataset.targets, seed=42)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for split_name, indices in zip(
        ("train", "validation", "test"), expected_splits
    ):
        with (manifest_dir / f"{split_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=("image_path", "class_name", "label_id")
            )
            writer.writeheader()
            for index in indices:
                path, label = dataset.samples[index]
                writer.writerow(
                    {
                        "image_path": Path(path).relative_to(tmp_path).as_posix(),
                        "class_name": dataset.classes[label],
                        "label_id": label,
                    }
                )

    actual_splits = load_rgb_split_manifest(
        tmp_path, dataset, seed=42, manifest_dir=manifest_dir
    )
    assert actual_splits == expected_splits


def test_evaluation_metrics_match_expected_confusion_and_macro_scores():
    report, matrix = experiments.calculate_evaluation_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        ["class-a", "class-b"],
    )

    assert report["accuracy"] == pytest.approx(0.75)
    assert report["macro avg"]["precision"] == pytest.approx(5 / 6)
    assert report["macro avg"]["recall"] == pytest.approx(0.75)
    assert report["macro avg"]["f1-score"] == pytest.approx(11 / 15)
    np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])


def test_train_architecture_writes_metrics_checkpoint_and_logs_run(
    tmp_path, monkeypatch
):
    samples = TensorDataset(
        torch.randn(10, 3),
        torch.arange(10, dtype=torch.long),
    )
    loader = DataLoader(samples, batch_size=10)
    monkeypatch.setitem(experiments.MODEL_PRESETS, "test-cnn", ("cnn", False, False))
    monkeypatch.setattr(
        experiments,
        "get_loaders",
        lambda **_kwargs: (loader, loader, loader, CLASS_NAMES),
    )
    monkeypatch.setattr(
        experiments,
        "build_model",
        lambda *_args, **_kwargs: nn.Linear(3, 10),
    )
    logged = {}

    def fake_log_experiment(**kwargs):
        logged.update(kwargs)
        assert all(Path(path).is_file() for path in kwargs["artifact_paths"])

    monkeypatch.setattr(experiments, "log_experiment", fake_log_experiment)

    result = experiments.train_architecture(
        "test-cnn",
        tmp_path,
        (list(range(10)), list(range(10)), list(range(10))),
        CLASS_NAMES,
        tmp_path,
        epochs=1,
        batch_size=10,
    )

    assert Path(result["checkpoint"]).is_file()
    assert Path(result["history"]).is_file()
    assert Path(result["confusion_matrix"]).is_file()
    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["f1_macro"] <= 1
    assert result["total_parameters"] == 40
    assert result["trainable_parameters"] == 40
    assert result["model_size_mb"] > 0
    assert logged["run_name"] == "test-cnn"
    assert logged["history"][0]["epoch"] == 1
