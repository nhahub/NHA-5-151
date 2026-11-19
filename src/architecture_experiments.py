"""Train, evaluate, and compare the EuroSAT RGB architecture shortlist."""

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.datasets import ImageFolder

from src.dataset import (
    DEFAULT_DATA,
    EUROSAT_CLASS_NAMES,
    _resolve_data_dir,
    get_loaders,
    load_rgb_split_manifest,
    make_stratified_splits,
)
from src.model import VGG16Config, build_model
from src.tracking import log_experiment

ROOT = Path(__file__).resolve().parents[1]
MODEL_PRESETS = {
    "cnn": ("cnn", False, False),
    "resnet18": ("resnet18", False, False),
    "resnet18-frozen": ("resnet18", True, True),
    "resnet18-finetune": ("resnet18", True, False),
    "vgg16": ("vgg16", False, False),
    "vgg16-frozen": ("vgg16", True, True),
    "vgg16-finetune": ("vgg16", True, False),
}
COMPARISON_FIELDS = (
    "architecture",
    "pretrained",
    "features_frozen",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "training_seconds",
    "total_parameters",
    "trainable_parameters",
    "model_size_mb",
    "checkpoint",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None, collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "_feature_extractor_frozen", False):
        model.eval()
        head = getattr(model, "fc", None)
        if head is None:
            head = getattr(model, "classifier", None)
        if head is None:
            raise RuntimeError("Frozen model does not expose a classifier head.")
        head.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    actual = []
    predicted = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            guesses = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (guesses == labels).sum().item()
            total_samples += batch_size
            if collect_predictions:
                actual.extend(labels.detach().cpu().tolist())
                predicted.extend(guesses.detach().cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("The data loader returned no samples.")

    result = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
    if collect_predictions:
        result["actual"] = actual
        result["predicted"] = predicted
    return result


def calculate_evaluation_metrics(actual, predicted, class_names):
    actual = list(actual)
    predicted = list(predicted)
    if not actual or len(actual) != len(predicted):
        raise ValueError("Evaluation requires equally sized, non-empty label lists.")
    if any(label < 0 or label >= len(class_names) for label in actual + predicted):
        raise ValueError("Evaluation labels must match the class mapping.")
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(class_names))),
    )
    return report, matrix


def _allocate_run_directory(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"run_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _save_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _save_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def _save_confusion_matrix(path, matrix, class_names):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])


def train_architecture(
    preset_name,
    data_dir,
    split_indices,
    class_names,
    output_dir,
    *,
    epochs=5,
    batch_size=64,
    learning_rate=None,
    seed=42,
    num_workers=0,
    vgg_hidden_units=512,
    vgg_dropout=0.5,
):
    if preset_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown architecture preset: {preset_name}")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if learning_rate is not None and (
        not math.isfinite(learning_rate) or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a finite positive number.")

    model_name, pretrained, freeze_features = MODEL_PRESETS[preset_name]
    if learning_rate is None:
        learning_rate = 0.0001 if pretrained else 0.001

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    train_loader, val_loader, test_loader, loader_classes = get_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
        num_workers=num_workers,
        split_indices=split_indices,
        pretrained=pretrained,
    )
    if loader_classes != class_names:
        raise RuntimeError("Dataset class mapping changed between architecture runs.")
    if len(class_names) != 10:
        raise ValueError(
            f"EuroSAT_RGB must contain 10 classes; found {len(class_names)}."
        )

    model = build_model(
        model_name,
        num_classes=len(class_names),
        pretrained=pretrained,
        freeze_features=freeze_features,
        vgg_config=VGG16Config(
            classifier_hidden=vgg_hidden_units,
            dropout=vgg_dropout,
            pool_size=7 if pretrained else 1,
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError(f"{preset_name} has no trainable parameters.")

    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    checkpoint_path = output_dir / f"{preset_name}.pth"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"{preset_name} epoch {epoch}/{epochs}: "
            f"train loss={row['train_loss']:.4f}, "
            f"val loss={row['val_loss']:.4f}, "
            f"val accuracy={row['val_accuracy']:.4f}",
            flush=True,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "architecture": preset_name,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "pretrained": pretrained,
                    "features_frozen": freeze_features,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

    training_seconds = time.perf_counter() - start
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        collect_predictions=True,
    )
    report, matrix = calculate_evaluation_metrics(
        test_metrics["actual"],
        test_metrics["predicted"],
        class_names,
    )

    history_path = output_dir / f"{preset_name}_history.csv"
    confusion_path = output_dir / f"{preset_name}_confusion_matrix.csv"
    metrics_path = output_dir / f"{preset_name}_metrics.json"
    _save_history(history_path, history)
    _save_confusion_matrix(confusion_path, matrix, class_names)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    metrics = {
        "architecture": preset_name,
        "model_name": model_name,
        "pretrained": pretrained,
        "features_frozen": freeze_features,
        "accuracy": test_metrics["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "test_loss": test_metrics["loss"],
        "training_seconds": training_seconds,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_count,
        "model_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "classes": class_names,
        "per_class_metrics": {
            name: report[name] for name in class_names
        },
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
        "confusion_matrix": str(confusion_path),
    }
    _save_json(metrics_path, metrics)

    effective_learning_rate = learning_rate
    log_experiment(
        params={
            "architecture": preset_name,
            "model_name": model_name,
            "pretrained": pretrained,
            "features_frozen": freeze_features,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "learning_rate": effective_learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "num_classes": len(class_names),
            "train_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "vgg_hidden_units": vgg_hidden_units if model_name == "vgg16" else 0,
            "vgg_dropout": vgg_dropout if model_name == "vgg16" else 0.0,
        },
        metrics={
            "train_loss": history[-1]["train_loss"],
            "train_accuracy": history[-1]["train_accuracy"],
            "validation_loss": history[-1]["val_loss"],
            "validation_accuracy": history[-1]["val_accuracy"],
            "test_loss": metrics["test_loss"],
            "test_accuracy": metrics["accuracy"],
            "test_precision_macro": metrics["precision_macro"],
            "test_recall_macro": metrics["recall_macro"],
            "test_f1_macro": metrics["f1_macro"],
            "test_precision_weighted": metrics["precision_weighted"],
            "test_recall_weighted": metrics["recall_weighted"],
            "test_f1_weighted": metrics["f1_weighted"],
            "training_seconds": training_seconds,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_count,
            "model_size_mb": metrics["model_size_mb"],
            "best_validation_loss": best_val_loss,
            "best_epoch": best_epoch,
        },
        artifact_paths=[
            checkpoint_path,
            history_path,
            confusion_path,
            metrics_path,
        ],
        history=history,
        run_name=preset_name,
    )
    return metrics


def _write_comparison(path, results):
    rows = sorted(results, key=lambda result: result["accuracy"], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                key: result[key]
                for key in COMPARISON_FIELDS
            }
            for result in rows
        )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and compare EuroSAT_RGB model architectures."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PRESETS),
        default=["cnn", "resnet18", "vgg16"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override the default: 0.001 from scratch, 0.0001 pretrained.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing RGB train.csv, validation.csv, and test.csv "
            "manifests. Defaults to data/processed/splits/seed_<seed> when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "architecture_comparison",
    )
    parser.add_argument("--vgg-hidden-units", type=int, default=512)
    parser.add_argument("--vgg-dropout", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.vgg_hidden_units < 1:
        parser.error("--vgg-hidden-units must be positive.")
    if not 0 <= args.vgg_dropout < 1:
        parser.error("--vgg-dropout must be in [0, 1).")
    if len(args.models) != len(set(args.models)):
        parser.error("--models cannot contain duplicate architectures.")
    return args


def main(argv=None):
    args = parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir).resolve()

    dataset = ImageFolder(str(data_dir))
    if tuple(dataset.classes) != EUROSAT_CLASS_NAMES:
        raise ValueError(
            "EuroSAT_RGB class folders do not match the project's expected "
            f"10-class ImageFolder mapping: {dataset.classes}"
        )
    split_indices = load_rgb_split_manifest(
        data_dir,
        dataset,
        seed=args.seed,
        manifest_dir=args.split_manifest_dir,
    )
    split_source = "saved RGB manifests"
    if split_indices is None:
        split_indices = make_stratified_splits(dataset.targets, args.seed)
        split_source = "deterministic stratified seed split"
    output_dir = _allocate_run_directory(args.output_dir.resolve())
    _save_json(
        output_dir / "classes.json",
        {str(index): name for index, name in enumerate(dataset.classes)},
    )
    split_sizes = [len(indices) for indices in split_indices]
    print(
        f"Dataset: {data_dir}\n"
        f"Classes ({len(dataset.classes)}): {dataset.classes}\n"
        f"Split sizes (train/validation/test): {split_sizes}\n"
        f"Split source: {split_source}\n"
        f"Seed: {args.seed}; outputs: {output_dir}",
        flush=True,
    )

    results = []
    for preset_name in args.models:
        results.append(
            train_architecture(
                preset_name,
                data_dir,
                split_indices,
                dataset.classes,
                output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                num_workers=args.num_workers,
                vgg_hidden_units=args.vgg_hidden_units,
                vgg_dropout=args.vgg_dropout,
            )
        )

    sorted_results = _write_comparison(
        output_dir / "comparison.csv",
        results,
    )
    _save_json(output_dir / "comparison.json", sorted_results)
    print("\nTest-set comparison (sorted by accuracy):", flush=True)
    for result in sorted_results:
        print(
            f"{result['architecture']}: accuracy={result['accuracy']:.4f}, "
            f"macro F1={result['f1_macro']:.4f}, "
            f"parameters={result['total_parameters']:,}, "
            f"training={result['training_seconds']:.1f}s",
            flush=True,
        )
    print(f"Comparison table: {output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
