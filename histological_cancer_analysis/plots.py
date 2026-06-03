from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import RocCurveDisplay

matplotlib.use("Agg")

from matplotlib import pyplot as plt

TRAIN_LOSS_COLUMNS = ("train/loss_epoch", "train/loss")
VALIDATION_LOSS_COLUMNS = ("val/loss",)


def save_loss_history_plots(
    training_loss: Sequence[tuple[int, float]],
    validation_loss: Sequence[tuple[int, float]],
    validation_roc_auc: Sequence[tuple[int, float]],
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    training_loss_path = _save_history_curve(
        points=training_loss,
        output_path=plots_dir / "training_loss.png",
        title="Training loss",
        ylabel="Loss",
    )
    validation_loss_path = _save_history_curve(
        points=validation_loss,
        output_path=plots_dir / "validation_loss.png",
        title="Validation loss",
        ylabel="Loss",
    )
    validation_roc_auc_path = _save_history_curve(
        points=validation_roc_auc,
        output_path=plots_dir / "validation_roc_auc.png",
        title="Validation ROC-AUC",
        ylabel="ROC-AUC",
    )

    for plot_path in (training_loss_path, validation_loss_path, validation_roc_auc_path):
        if plot_path is not None:
            plot_paths.append(plot_path)

    return plot_paths


def save_training_loss_plots(metrics_csv: Path, plots_dir: Path) -> list[Path]:
    if not metrics_csv.exists():
        return []

    metrics = pd.read_csv(metrics_csv)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    train_loss_path = _save_metric_curve(
        metrics=metrics,
        metric_columns=TRAIN_LOSS_COLUMNS,
        output_path=plots_dir / "training_loss.png",
        title="Training loss",
        ylabel="Loss",
    )
    validation_loss_path = _save_metric_curve(
        metrics=metrics,
        metric_columns=VALIDATION_LOSS_COLUMNS,
        output_path=plots_dir / "validation_loss.png",
        title="Validation loss",
        ylabel="Loss",
    )

    for plot_path in (train_loss_path, validation_loss_path):
        if plot_path is not None:
            plot_paths.append(plot_path)

    return plot_paths


def save_roc_curve_plot(
    labels: list[int],
    probabilities: list[float],
    plots_dir: Path,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "roc_curve.png"
    _, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axis)
    axis.set_title("ROC curve")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _save_metric_curve(
    metrics: pd.DataFrame,
    metric_columns: tuple[str, ...],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path | None:
    metric_column = _find_metric_column(metrics, metric_columns)
    if metric_column is None:
        return None

    curve = metrics.dropna(subset=[metric_column])
    if curve.empty:
        return None

    x_column = "epoch" if "epoch" in curve.columns else "step"
    _, axis = plt.subplots(figsize=(7, 4))
    axis.plot(curve[x_column], curve[metric_column], marker="o", linewidth=1.5)
    axis.set_xlabel(x_column.title())
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _save_history_curve(
    points: Sequence[tuple[int, float]],
    output_path: Path,
    title: str,
    ylabel: str,
) -> Path | None:
    if not points:
        return None

    epochs, values = zip(*points, strict=True)
    _, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, values, marker="o", linewidth=1.5)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _find_metric_column(
    metrics: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for column in candidates:
        if column in metrics.columns and not metrics[column].dropna().empty:
            return column
    return None
