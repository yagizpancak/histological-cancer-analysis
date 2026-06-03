from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from dvc.exceptions import DvcException
from dvc.repo import Repo
from omegaconf import DictConfig
from PIL import Image

from histological_cancer_analysis.constants import RGB_MODE
from histological_cancer_analysis.evaluate import load_classifier_from_checkpoint
from histological_cancer_analysis.transforms import build_image_transforms


def infer(config: DictConfig) -> dict[str, Any]:
    checkpoint_path = config.get("checkpoint_path")
    image_path = config.get("image_path")
    if checkpoint_path is None:
        msg = "Set checkpoint_path=/path/to/model.ckpt before running inference."
        raise ValueError(msg)
    if image_path is None:
        msg = "Set image_path=/path/to/image.tif before running inference."
        raise ValueError(msg)

    if config.dvc.enabled:
        data_targets = [target for target in config.dvc.data_targets if Path(target).exists()]
        if data_targets:
            with suppress(DvcException):
                Repo(Path.cwd()).pull(targets=data_targets, remote=config.dvc.data_remote)

        model_targets = [
            target for target in config.dvc.model_targets if Path(target).exists()
        ]
        if model_targets:
            with suppress(DvcException):
                Repo(Path.cwd()).pull(targets=model_targets, remote=config.dvc.model_remote)

    probability = predict_probability(config, Path(checkpoint_path), Path(image_path))
    return {
        "probability": probability,
        "label": int(probability >= config.threshold),
    }


def predict_probability(config: DictConfig, checkpoint_path: Path, image_path: Path) -> float:
    lightning_module = load_classifier_from_checkpoint(config, checkpoint_path)
    transform = build_image_transforms(config.data.image_size, augment=False)

    with Image.open(image_path) as image_file:
        image = image_file.convert(RGB_MODE)
    batch = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = lightning_module(batch)
        return float(torch.sigmoid(logit).item())
