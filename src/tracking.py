"""Local MLflow tracking for model training runs."""

import logging
import os
from pathlib import Path

import mlflow

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "eurosat-land-cover-classification"
LOGGER = logging.getLogger(__name__)


def log_experiment(
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    artifact_paths: list[str | Path] | None = None,
) -> None:
    """Log one completed training run to the local MLflow tracking directory."""
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for artifact_path in artifact_paths or []:
            path = Path(artifact_path)
            if path.is_file():
                mlflow.log_artifact(str(path))
            elif path.is_dir():
                mlflow.log_artifacts(str(path))
            else:
                LOGGER.warning("Skipping missing MLflow artifact: %s", path)
