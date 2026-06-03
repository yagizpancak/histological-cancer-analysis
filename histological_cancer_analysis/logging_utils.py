from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pytorch_lightning.loggers import CSVLogger, Logger

from histological_cancer_analysis.constants import LoggerName


def build_logger(logging_config: Any) -> Logger | bool:
    logger_name = str(logging_config.logger).lower()
    if logger_name == LoggerName.NONE:
        return False
    if logger_name == LoggerName.CSV:
        return CSVLogger(
            save_dir=str(logging_config.save_dir),
            name=str(logging_config.experiment_name),
        )
    if logger_name == LoggerName.MLFLOW:
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


def log_artifacts(logger: Any, artifact_paths: Sequence[Path]) -> None:
    if logger is False or not artifact_paths:
        return

    if isinstance(logger, Sequence) and not isinstance(logger, str):
        for single_logger in logger:
            log_artifacts(single_logger, artifact_paths)
        return

    from pytorch_lightning.loggers import MLFlowLogger

    if not isinstance(logger, MLFlowLogger):
        return

    for artifact_path in artifact_paths:
        if artifact_path.exists():
            logger.experiment.log_artifact(logger.run_id, str(artifact_path))


def log_metrics(logger: Any, metrics: Mapping[str, float]) -> None:
    if logger is False or not metrics:
        return

    if isinstance(logger, Sequence) and not isinstance(logger, str):
        for single_logger in logger:
            log_metrics(single_logger, metrics)
        return

    from pytorch_lightning.loggers import MLFlowLogger

    if not isinstance(logger, MLFlowLogger):
        return

    logger.experiment.log_metrics(
        logger.run_id,
        {metric_name: float(metric_value) for metric_name, metric_value in metrics.items()},
    )
