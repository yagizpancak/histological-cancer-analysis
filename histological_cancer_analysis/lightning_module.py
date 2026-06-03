from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class CancerClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = nn.BCEWithLogitsLoss()

        self.validation_accuracy = BinaryAccuracy()
        self.validation_precision = BinaryPrecision()
        self.validation_recall = BinaryRecall()
        self.validation_f1 = BinaryF1Score()
        self.validation_roc_auc = BinaryAUROC()

        self.test_accuracy = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_roc_auc = BinaryAUROC()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).flatten()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, labels = batch
        labels = labels.float().flatten()
        logits = self(images)
        loss = self.loss(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        images, labels = batch
        labels = labels.int().flatten()
        logits = self(images)
        probabilities = torch.sigmoid(logits)
        loss = self.loss(logits, labels.float())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log(
            "val/accuracy", self.validation_accuracy(probabilities, labels), on_epoch=True
        )
        self.log(
            "val/precision", self.validation_precision(probabilities, labels), on_epoch=True
        )
        self.log("val/recall", self.validation_recall(probabilities, labels), on_epoch=True)
        self.log("val/f1", self.validation_f1(probabilities, labels), on_epoch=True)
        self.log("val/roc_auc", self.validation_roc_auc(probabilities, labels), on_epoch=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        images, labels = batch
        labels = labels.int().flatten()
        logits = self(images)
        probabilities = torch.sigmoid(logits)
        loss = self.loss(logits, labels.float())

        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", self.test_accuracy(probabilities, labels), on_epoch=True)
        self.log("test/precision", self.test_precision(probabilities, labels), on_epoch=True)
        self.log("test/recall", self.test_recall(probabilities, labels), on_epoch=True)
        self.log("test/f1", self.test_f1(probabilities, labels), on_epoch=True)
        self.log("test/roc_auc", self.test_roc_auc(probabilities, labels), on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {"optimizer": optimizer}
