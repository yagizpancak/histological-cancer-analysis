from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    labels_csv: Path
    splits_dir: Path
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42


def create_stratified_splits(config: SplitConfig) -> None:
    train_csv = config.splits_dir / "train.csv"
    validation_csv = config.splits_dir / "validation.csv"
    test_csv = config.splits_dir / "test.csv"
    if train_csv.exists() and validation_csv.exists() and test_csv.exists():
        return

    labels = _read_labels(config.labels_csv)
    _validate_split_fractions(config)

    train_data, holdout_data = train_test_split(
        labels,
        train_size=config.train_fraction,
        random_state=config.random_state,
        stratify=labels["label"],
    )
    validation_relative_fraction = config.validation_fraction / (
        config.validation_fraction + config.test_fraction
    )
    validation_data, test_data = train_test_split(
        holdout_data,
        train_size=validation_relative_fraction,
        random_state=config.random_state,
        stratify=holdout_data["label"],
    )

    config.splits_dir.mkdir(parents=True, exist_ok=True)
    train_data.sort_values("id").to_csv(train_csv, index=False)
    validation_data.sort_values("id").to_csv(validation_csv, index=False)
    test_data.sort_values("id").to_csv(test_csv, index=False)


def _read_labels(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        msg = f"Labels file not found: {labels_csv}"
        raise FileNotFoundError(msg)

    labels = pd.read_csv(labels_csv)
    required_columns = {"id", "label"}
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
