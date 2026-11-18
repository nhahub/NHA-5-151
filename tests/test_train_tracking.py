"""Tests for MLflow tracking in the multispectral training pipeline.

Synthetic data only; MLflow runs against a real local file store in a temp
directory, so the real mlruns/ folder is never touched. No real EuroSAT
data and no claim about real model accuracy.
"""

import sys

import mlflow
import numpy as np
import pytest

import src.tracking as tracking
from src import train

CLASSES = ["AnnualCrop", "Forest", "SeaLake"]


@pytest.fixture()
def tiny_dataset(tmp_path):
    root = tmp_path / "EuroSAT_MS"
    rng = np.random.default_rng(0)
    for name in CLASSES:
        (root / name).mkdir(parents=True)
        for i in range(10):
            image = rng.uniform(0, 1, size=(13, 8, 8)).astype(np.float32)
            np.save(root / name / f"{name}_{i}.npy", image)
    return root


@pytest.fixture()
def isolated_tracking(tmp_path, monkeypatch):
    store = tmp_path / "mlruns"
    monkeypatch.setattr(tracking, "TRACKING_DIR", store)
    return store


def _run_train(monkeypatch, tmp_path, data_dir, *extra):
    argv = [
        "train.py",
        "--data-dir", str(data_dir),
        "--output-dir", str(tmp_path / "out"),
        "--splits-path", str(tmp_path / "splits.csv"),
        "--epochs", "2",
        "--batch-size", "4",
        "--device", "cpu",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()


def _runs(store):
    mlflow.set_tracking_uri(store.as_uri())
    experiment = mlflow.get_experiment_by_name(tracking.EXPERIMENT_NAME)
    if experiment is None:
        return []
    return mlflow.search_runs([experiment.experiment_id], output_format="list")


def test_ms_training_logs_one_mlflow_run(
    monkeypatch, tmp_path, tiny_dataset, isolated_tracking
):
    _run_train(monkeypatch, tmp_path, tiny_dataset)

    runs = _runs(isolated_tracking)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["pipeline"] == "multispectral"
    assert run.data.params["model"] == "SmallCNN"
    assert run.data.params["num_channels"] == "13"
    assert run.data.params["seed"] == "42"
    for key in ("train_loss", "validation_loss", "test_loss", "test_accuracy",
                "best_epoch"):
        assert key in run.data.metrics

    # Logged values must equal what training wrote to metrics.json.
    import json

    summary = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert run.data.metrics["test_accuracy"] == pytest.approx(
        summary["test_accuracy"]
    )
    assert run.data.metrics["validation_loss"] == pytest.approx(
        summary["best_validation_loss"]
    )

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    assert {"best_model.pth", "metrics.json", "metrics.csv", "classes.json"} <= artifacts


def test_no_tracking_flag_skips_mlflow(
    monkeypatch, tmp_path, tiny_dataset, isolated_tracking
):
    _run_train(monkeypatch, tmp_path, tiny_dataset, "--no-tracking")

    assert (tmp_path / "out" / "metrics.json").is_file()
    assert not isolated_tracking.exists()


def test_tracking_failure_does_not_lose_training(
    monkeypatch, tmp_path, tiny_dataset, isolated_tracking, capsys
):
    def broken(*args, **kwargs):
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(tracking, "log_experiment", broken)
    _run_train(monkeypatch, tmp_path, tiny_dataset)

    output = capsys.readouterr().out
    assert "WARNING: MLflow logging failed" in output
    assert (tmp_path / "out" / "best_model.pth").is_file()
    assert (tmp_path / "out" / "metrics.json").is_file()
