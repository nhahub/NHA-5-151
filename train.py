"""Train and evaluate a EuroSAT land-cover classifier."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from src.dataset import get_loaders
from src.model import build_model
from src.tracking import log_experiment

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "raw" / "EuroSAT_RGB" / "EuroSAT_RGB"
DEFAULT_MODEL_PATH = ROOT / "models" / "land_type_classifier.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a EuroSAT RGB classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", choices=("cnn", "resnet18"), default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def run_epoch(model, loader, loss_function, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    sample_count = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()

            predictions = model(images)
            loss = loss_function(predictions, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError(
            "epochs and batch size must be positive; "
            "learning rate must be greater than zero"
        )
    if not args.data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {args.data_dir}. "
            "Download EuroSAT with 'python src/download_eurosat.py'."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = build_model(args.model, num_classes=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, "
            f"validation_loss: {val_loss:.4f}, validation_accuracy: {val_accuracy:.4f}"
        )

    test_loss, test_accuracy = run_epoch(model, test_loader, loss_function, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "class_names": class_names,
        },
        args.output,
    )
    print(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
    print(f"Saved model: {args.output}")

    log_experiment(
        params={
            "model": args.model,
            "data_dir": str(args.data_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "seed": args.seed,
            "num_classes": len(class_names),
            "device": str(device),
        },
        metrics={
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        artifact_paths=[args.output],
    )


if __name__ == "__main__":
    main()
