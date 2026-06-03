from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor

from histological_cancer_analysis.constants import (
    ID_COLUMN,
    LABEL_COLUMN,
    RGB_MODE,
    DatasetSplit,
)

ImageOutput = Image.Image | Tensor
ImageTransform = Callable[[Image.Image], ImageOutput]
PLACEHOLDER_SOURCE_URL = "replace-with-google-drive-file-id"


@dataclass(frozen=True)
class SplitConfig:
    labels_csv: Path
    splits_dir: Path
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    random_state: int


class HistologyPatchDataset:
    def __init__(
        self,
        split_csv: Path,
        image_dir: Path,
        image_extension: str,
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
    train_csv = config.splits_dir / f"{DatasetSplit.TRAIN}.csv"
    validation_csv = config.splits_dir / f"{DatasetSplit.VALIDATION}.csv"
    test_csv = config.splits_dir / f"{DatasetSplit.TEST}.csv"
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


def download_data(data_config: Any, dvc_config: Any) -> None:
    labels_csv = Path(data_config.labels_csv)
    image_dir = Path(data_config.image_dir)
    image_extension = str(data_config.image_extension).lstrip(".")
    if _raw_data_exists(labels_csv, image_dir, image_extension):
        return

    source_url = str(dvc_config.source_url)
    if PLACEHOLDER_SOURCE_URL in source_url:
        msg = (
            "Raw data is missing and dvc.source_url still uses the placeholder "
            "Google Drive link."
        )
        raise FileNotFoundError(msg)

    archive_path = Path(dvc_config.archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        import gdown

        downloaded_path = gdown.download(
            url=source_url,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )
        if downloaded_path is None:
            msg = f"Failed to download data archive from {source_url}"
            raise RuntimeError(msg)

    _extract_dataset_archive(
        archive_path=archive_path,
        output_dir=archive_path.parent,
        image_dir=image_dir,
        image_extension=image_extension,
    )
    if not _raw_data_exists(labels_csv, image_dir, image_extension):
        msg = (
            "Data archive was downloaded, but the expected files were not found: "
            f"{labels_csv} and {image_dir}/*.{image_extension}"
        )
        raise FileNotFoundError(msg)


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


def _raw_data_exists(labels_csv: Path, image_dir: Path, image_extension: str) -> bool:
    if not labels_csv.exists() or not image_dir.exists():
        return False
    return any(image_dir.glob(f"*.{image_extension}"))


def _extract_dataset_archive(
    archive_path: Path,
    output_dir: Path,
    image_dir: Path,
    image_extension: str,
) -> None:
    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)

    train_archive_path = output_dir / "train.zip"
    if train_archive_path.exists() and not any(image_dir.glob(f"*.{image_extension}")):
        image_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(train_archive_path) as archive:
            archive.extractall(image_dir)


def _validate_split_fractions(config: SplitConfig) -> None:
    total_fraction = config.train_fraction + config.validation_fraction + config.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        msg = "Train, validation, and test fractions must sum to 1.0."
        raise ValueError(msg)
