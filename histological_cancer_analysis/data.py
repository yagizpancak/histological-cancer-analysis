from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor

from histological_cancer_analysis.constants import (
    ID_COLUMN,
    LABEL_COLUMN,
    RGB_MODE,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)

ImageOutput = Image.Image | Tensor
ImageTransform = Callable[[Image.Image], ImageOutput]


@dataclass(frozen=True)
class SplitConfig:
    labels_csv: Path
    splits_dir: Path
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42


class HistologyPatchDataset:
    def __init__(
        self,
        split_csv: Path,
        image_dir: Path,
        image_extension: str = "tif",
        transform: ImageTransform | None = None,
    ) -> None:
        self.samples = _read_labels(split_csv)
        self.image_dir = image_dir
        self.image_extension = image_extension.lstrip(".")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[ImageOutput, int]:
        row = self.samples.iloc[index]
        image_path = self.image_dir / f"{row[ID_COLUMN]}.{self.image_extension}"
        with Image.open(image_path) as image_file:
            image = image_file.convert(RGB_MODE)
        label = int(row[LABEL_COLUMN])

        if self.transform is not None:
            return self.transform(image), label

        return image, label


def create_stratified_splits(config: SplitConfig) -> None:
    train_csv = config.splits_dir / f"{TRAIN_SPLIT}.csv"
    validation_csv = config.splits_dir / f"{VALIDATION_SPLIT}.csv"
    test_csv = config.splits_dir / f"{TEST_SPLIT}.csv"
    if train_csv.exists() and validation_csv.exists() and test_csv.exists():
        return

    labels = _read_labels(config.labels_csv)
    _validate_split_fractions(config)

    train_data, holdout_data = train_test_split(
        labels,
        train_size=config.train_fraction,
        random_state=config.random_state,
        stratify=labels[LABEL_COLUMN],
    )
    validation_relative_fraction = config.validation_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation_data, test_data = train_test_split(
        holdout_data,
        train_size=validation_relative_fraction,
        random_state=config.random_state,
        stratify=holdout_data[LABEL_COLUMN],
    )

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    train_data.sort_values(ID_COLUMN).to_csv(train_csv, index=False)
    validation_data.sort_values(ID_COLUMN).to_csv(validation_csv, index=False)
    test_data.sort_values(ID_COLUMN).to_csv(test_csv, index=False)


def _read_labels(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        msg = f"Labels file not found: {labels_csv}"
        raise FileNotFoundError(msg)

    labels = pd.read_csv(labels_csv)
    required_columns = {ID_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(labels.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"{labels_csv} is missing required columns: {missing}"
        raise ValueError(msg)

    return labels


def _validate_split_fractions(config: SplitConfig) -> None:
    total_fraction = config.train_fraction + config.validation_fraction + config.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        msg = "Train, validation, and test fractions must sum to 1.0."
        raise ValueError(msg)
