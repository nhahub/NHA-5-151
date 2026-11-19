from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import src.tracking as tracking


def test_log_experiment_logs_parameters_metrics_history_and_artifact(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "checkpoint.pth"
    artifact.write_bytes(b"checkpoint")
    calls = {"metrics": [], "artifacts": []}

    @contextmanager
    def start_run(run_name=None):
        calls["run_name"] = run_name
        yield

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _uri: calls.update(tracking_uri=_uri),
        set_experiment=lambda _name: calls.update(experiment=_name),
        start_run=start_run,
        log_params=lambda params: calls.update(params=params),
        log_metrics=lambda metrics, step=None: calls["metrics"].append((metrics, step)),
        log_artifact=lambda path: calls["artifacts"].append(path),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")

    tracking.log_experiment(
        {"architecture": "cnn", "seed": 42},
        {"test_accuracy": 0.8},
        artifact_paths=[artifact],
        history=[
            {
                "epoch": 1,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "val_loss": 1.2,
                "val_accuracy": 0.4,
            }
        ],
        run_name="cnn",
    )

    assert calls["experiment"] == "eurosat-land-cover-classification"
    assert calls["run_name"] == "cnn"
    assert calls["params"]["architecture"] == "cnn"
    assert calls["metrics"][0] == ({"test_accuracy": 0.8}, None)
    assert calls["metrics"][1][1] == 1
    assert calls["artifacts"] == [str(artifact)]
    assert tracking.TRACKING_DIR.is_dir()


def test_log_experiment_reports_missing_artifacts(tmp_path, monkeypatch):
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _uri: None,
        set_experiment=lambda _name: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")

    with pytest.raises(FileNotFoundError, match="MLflow artifact"):
        tracking.log_experiment(
            {},
            {},
            artifact_paths=[Path(tmp_path / "missing.pth")],
        )


def test_log_experiment_preserves_directory_artifacts(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    calls = []

    @contextmanager
    def start_run(run_name=None):
        yield

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _uri: None,
        set_experiment=lambda _name: None,
        start_run=start_run,
        log_params=lambda _params: None,
        log_metrics=lambda _metrics, step=None: None,
        log_artifacts=lambda path: calls.append(path),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setattr(tracking, "TRACKING_DIR", tmp_path / "mlruns")

    tracking.log_experiment({}, {}, artifact_paths=[artifact_dir])

    assert calls == [str(artifact_dir)]
