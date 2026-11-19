"""Local MLflow tracking for EuroSAT RGB and multispectral experiments."""

import logging
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).resolve().parents[1]
TRACKING_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "eurosat-land-cover-classification"
LOGGER = logging.getLogger(__name__)


def log_experiment(
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    artifact_paths: list[str | Path] | None = None,
    history: list[dict[str, int | float]] | None = None,
    run_name: str | None = None,
) -> None:
    """Log a run while preserving existing file and directory artifact support."""
    paths = [Path(artifact_path) for artifact_path in artifact_paths or []]
    for path in paths:
        if not path.exists():
            LOGGER.error("Cannot log missing MLflow artifact: %s", path)
            raise FileNotFoundError(f"MLflow artifact does not exist: {path}")

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow logging requires the project dependencies. "
            "Install them with `python -m pip install -r requirements.txt`."
        ) from exc

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        for row in history or []:
            step = int(row["epoch"])
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key != "epoch"
                },
                step=step,
            )

        for path in paths:
            if path.is_file():
                mlflow.log_artifact(str(path))
            elif path.is_dir():
                mlflow.log_artifacts(str(path))
            else:
                LOGGER.error("Cannot log missing MLflow artifact: %s", path)
                raise FileNotFoundError(f"MLflow artifact does not exist: {path}")
