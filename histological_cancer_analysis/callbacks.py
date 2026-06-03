from __future__ import annotations

import pytorch_lightning as pl
import torch


class LossHistoryCallback(pl.Callback):
    def __init__(self) -> None:
        self.training_loss: list[tuple[int, float]] = []
        self.validation_loss: list[tuple[int, float]] = []
        self.validation_roc_auc: list[tuple[int, float]] = []

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        _append_metric(
            history=self.training_loss,
            trainer=trainer,
            metric_name="train/loss_epoch",
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        _append_metric(
            history=self.validation_loss,
            trainer=trainer,
            metric_name="val/loss",
        )
        _append_metric(
            history=self.validation_roc_auc,
            trainer=trainer,
            metric_name="val/roc_auc",
        )


def _append_metric(
    history: list[tuple[int, float]],
    trainer: pl.Trainer,
    metric_name: str,
) -> None:
    metric = trainer.callback_metrics.get(metric_name)
    if metric is None:
        return

    if isinstance(metric, torch.Tensor):
        metric = metric.detach().cpu().item()

    history.append((trainer.current_epoch, float(metric)))
