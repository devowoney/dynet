"""Data modules for data-driven model development."""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import pytorch_lightning as pl
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    torch = None
    DataLoader = None
    TensorDataset = None

    class _LightningDataModule:
        pass
else:
    _LightningDataModule = pl.LightningDataModule


class SineWaveDataModule(_LightningDataModule):
    """Simple synthetic regression datamodule."""

    def __init__(
        self,
        batch_size: int = 32,
        train_size: int = 512,
        val_size: int = 128,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.noise_std = noise_std
        self._train_dataset: Optional["TensorDataset"] = None
        self._val_dataset: Optional["TensorDataset"] = None

    def _build_dataset(self, size: int) -> "TensorDataset":
        if torch is None or TensorDataset is None:
            raise ModuleNotFoundError(
                "torch and pytorch-lightning are required to use SineWaveDataModule"
            )

        x = torch.linspace(-math.pi, math.pi, size).unsqueeze(1)
        noise = torch.randn_like(x) * self.noise_std
        y = torch.sin(x) + noise
        return TensorDataset(x, y)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = self._build_dataset(self.train_size)
            self._val_dataset = self._build_dataset(self.val_size)

    def train_dataloader(self):
        if DataLoader is None or self._train_dataset is None:
            raise RuntimeError("Call setup('fit') after installing torch/lightning")
        return DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if DataLoader is None or self._val_dataset is None:
            raise RuntimeError("Call setup('fit') after installing torch/lightning")
        return DataLoader(self._val_dataset, batch_size=self.batch_size)
