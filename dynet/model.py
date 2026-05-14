"""Lightning model definitions."""

from __future__ import annotations

from typing import Any

try:
    import pytorch_lightning as pl
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pl = None
    torch = None
    nn = None

    class _LightningModule:
        def save_hyperparameters(self, *args, **kwargs):
            return None
else:
    _LightningModule = pl.LightningModule


class MLPRegressor(_LightningModule):
    """A tiny MLP regressor for synthetic dynamics."""

    def __init__(self, hidden_dim: int = 64, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        if nn is None:
            self.net = None
            self.loss_fn = None
            return

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: Any) -> Any:
        if self.net is None:
            raise ModuleNotFoundError("torch and pytorch-lightning are required")
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> Any:
        if torch is None:
            raise ModuleNotFoundError("torch and pytorch-lightning are required")
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
