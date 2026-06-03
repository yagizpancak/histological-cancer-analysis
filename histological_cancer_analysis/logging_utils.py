from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning.loggers import CSVLogger, Logger


def build_logger(logging_config: Any) -> Logger | bool:
    logger_name = str(logging_config.logger).lower()
    if logger_name == "none":
        return False
    if logger_name == "csv":
        return CSVLogger(
            save_dir=str(logging_config.save_dir),
            name=str(logging_config.experiment_name),
        )
    if logger_name == "mlflow":
        from pytorch_lightning.loggers import MLFlowLogger

        return MLFlowLogger(
            experiment_name=str(logging_config.experiment_name),
            run_name=logging_config.run_name,
            tracking_uri=str(logging_config.mlflow_tracking_uri),
        )

    msg = f"Unsupported logger: {logging_config.logger}"
    raise ValueError(msg)


def ensure_output_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
