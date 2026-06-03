from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from histological_cancer_analysis.artifacts import save_json_artifact
from histological_cancer_analysis.constants import DatasetSplit
from histological_cancer_analysis.data import HistologyPatchDataset, create_stratified_splits
from histological_cancer_analysis.dataloaders import create_data_loader
from histological_cancer_analysis.lightning_module import CancerClassifier
from histological_cancer_analysis.logging_utils import (
    build_logger,
    log_artifacts,
    log_metrics,
)
from histological_cancer_analysis.models import build_model
from histological_cancer_analysis.plots import save_roc_curve_plot
from histological_cancer_analysis.train import build_split_config
from histological_cancer_analysis.transforms import build_image_transforms


def evaluate(config: DictConfig) -> dict[str, Any]:
    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path is None:
        msg = "Set checkpoint_path=/path/to/model.ckpt before running evaluation."
        raise ValueError(msg)

    create_stratified_splits(build_split_config(config.data))
    test_loader = build_test_dataloader(config)
    lightning_module = load_classifier_from_checkpoint(config, Path(checkpoint_path))
    probabilities, labels = predict_probabilities(lightning_module, test_loader)
    metrics = compute_metrics(labels, probabilities, threshold=config.threshold)
    roc_curve_path = save_roc_curve_plot(
        labels=labels,
        probabilities=probabilities,
        plots_dir=Path(config.logging.plots_dir),
    )
    metrics["roc_curve_path"] = str(roc_curve_path)
    confusion_matrix_path = save_json_artifact(
        metrics["confusion_matrix"],
        Path(config.logging.save_dir) / "evaluation" / "confusion_matrix.json",
    )
    metrics["confusion_matrix_path"] = str(confusion_matrix_path)
    log_evaluation_results(config, metrics, [roc_curve_path, confusion_matrix_path])
    return metrics


def build_test_dataloader(config: DictConfig) -> DataLoader:
    splits_dir = Path(config.data.splits_dir)
    image_dir = Path(config.data.image_dir)
    test_dataset = HistologyPatchDataset(
        split_csv=splits_dir / f"{DatasetSplit.TEST}.csv",
        image_dir=image_dir,
        image_extension=config.data.image_extension,
        transform=build_image_transforms(config.data.image_size, augment=False),
    )
    return create_data_loader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )


def load_classifier_from_checkpoint(
    config: DictConfig,
    checkpoint_path: Path,
) -> CancerClassifier:
    model = build_model(
        name=config.model.name,
        dropout=config.model.get("dropout", 0.25),
        pretrained=False,
    )
    lightning_module = CancerClassifier.load_from_checkpoint(
        checkpoint_path,
        model=model,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
        map_location="cpu",
    )
    lightning_module.eval()
    return lightning_module


def predict_probabilities(
    lightning_module: CancerClassifier,
    data_loader: DataLoader,
) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for images, batch_labels in data_loader:
            logits = lightning_module(images)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.int().flatten().cpu().tolist())

    return probabilities, labels


def compute_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    return {
        "roc_auc": roc_auc_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def log_evaluation_results(
    config: DictConfig,
    metrics: dict[str, Any],
    artifact_paths: list[Path],
) -> None:
    logger = build_logger(config.logging)
    log_metrics(
        logger,
        {
            "test/roc_auc": metrics["roc_auc"],
            "test/accuracy": metrics["accuracy"],
            "test/precision": metrics["precision"],
            "test/recall": metrics["recall"],
            "test/f1": metrics["f1"],
        },
    )
    log_artifacts(logger, artifact_paths)
