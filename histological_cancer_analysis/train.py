from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from histological_cancer_analysis.callbacks import LossHistoryCallback
from histological_cancer_analysis.constants import DatasetSplit, LoggerName
from histological_cancer_analysis.data import (
    HistologyPatchDataset,
    SplitConfig,
    create_stratified_splits,
)
from histological_cancer_analysis.dataloaders import create_data_loader
from histological_cancer_analysis.lightning_module import CancerClassifier
from histological_cancer_analysis.logging_utils import (
    build_logger,
    ensure_output_dirs,
    log_artifacts,
)
from histological_cancer_analysis.models import build_model
from histological_cancer_analysis.plots import save_loss_history_plots
from histological_cancer_analysis.transforms import build_image_transforms


def train(config: DictConfig) -> None:
    pl.seed_everything(config.seed, workers=True)
    ensure_output_dirs(
        config.trainer.checkpoints_dir,
        config.logging.save_dir,
        config.logging.plots_dir,
    )
    create_stratified_splits(build_split_config(config.data))
    train_loader, validation_loader = build_train_dataloaders(config)
    lightning_module, trainer, loss_history = build_training_components(config)
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
    )
    plot_paths = save_loss_history_plots(
        training_loss=loss_history.training_loss,
        validation_loss=loss_history.validation_loss,
        plots_dir=Path(config.logging.plots_dir),
    )
    log_artifacts(trainer.logger, plot_paths)


def build_split_config(data_config: DictConfig) -> SplitConfig:
    return SplitConfig(
        labels_csv=Path(data_config.labels_csv),
        splits_dir=Path(data_config.splits_dir),
        train_fraction=data_config.train_fraction,
        validation_fraction=data_config.validation_fraction,
        test_fraction=data_config.test_fraction,
        random_state=data_config.random_state,
    )


def build_train_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    splits_dir = Path(config.data.splits_dir)
    image_dir = Path(config.data.image_dir)
    train_dataset = HistologyPatchDataset(
        split_csv=splits_dir / f"{DatasetSplit.TRAIN}.csv",
        image_dir=image_dir,
        image_extension=config.data.image_extension,
        transform=build_image_transforms(config.data.image_size, augment=True),
    )
    validation_dataset = HistologyPatchDataset(
        split_csv=splits_dir / f"{DatasetSplit.VALIDATION}.csv",
        image_dir=image_dir,
        image_extension=config.data.image_extension,
        transform=build_image_transforms(config.data.image_size, augment=False),
    )
    train_loader = create_data_loader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    validation_loader = create_data_loader(
        dataset=validation_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    return train_loader, validation_loader


def build_training_components(
    config: DictConfig,
) -> tuple[CancerClassifier, pl.Trainer, LossHistoryCallback]:
    model = build_model(
        name=config.model.name,
        dropout=config.model.get("dropout", 0.25),
        pretrained=config.model.get("pretrained", False),
    )
    lightning_module = CancerClassifier(
        model=model,
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    loss_history = LossHistoryCallback()
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        log_every_n_steps=config.trainer.log_every_n_steps,
        callbacks=build_callbacks(config, loss_history),
        logger=build_logger(config.logging),
        fast_dev_run=config.trainer.fast_dev_run,
    )
    return lightning_module, trainer, loss_history


def build_callbacks(
    config: DictConfig,
    loss_history: LossHistoryCallback,
) -> list[pl.Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.trainer.checkpoints_dir),
        filename="{epoch:02d}",
        monitor=config.trainer.monitor,
        mode=config.trainer.monitor_mode,
        save_top_k=1,
    )
    callbacks: list[pl.Callback] = [checkpoint_callback, loss_history]
    if config.logging.logger != LoggerName.NONE:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks
