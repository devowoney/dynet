from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
from torch import nn


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    lr: float = 1e-3


class NetCDFZarrForecastModule(L.LightningModule):
    def __init__(self, input_dim: int, input_steps: int, pred_steps: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.lr = cfg.lr
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * input_steps, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, input_dim * pred_steps),
        )
        self.loss_fn = nn.MSELoss()
        self.input_dim = input_dim
        self.pred_steps = pred_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y.view(x.shape[0], self.pred_steps, self.input_dim)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step(batch, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

