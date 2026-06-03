from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn


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
        self.save_hyperparameters(ignore=["model"])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).flatten()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {"optimizer": optimizer}
