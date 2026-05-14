from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, input_steps: int, pred_steps: int) -> None:
        if data.ndim < 2:
            raise ValueError("data must have at least [time, feature] dimensions")
        if input_steps < 1 or pred_steps < 1:
            raise ValueError("input_steps and pred_steps must be >= 1")

        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.num_samples = self.data.shape[0] - input_steps - pred_steps + 1
        if self.num_samples <= 0:
            raise ValueError("not enough timesteps for requested input/pred windows")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.input_steps]
        y = self.data[idx + self.input_steps : idx + self.input_steps + self.pred_steps]
        return x, y


@dataclass
class DataConfig:
    path: str = ""
    variable: str = "state"
    store_format: str = "zarr"  # zarr | netcdf
    input_steps: int = 12
    pred_steps: int = 1
    batch_size: int = 32
    num_workers: int = 0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    time_dim: str = "time"
    split_seed: int = 42


class ManualDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None

    def _load_array(self) -> np.ndarray:
        if not self.cfg.path:
            raise ValueError("data.path must be set to a .zarr store or .nc file")
        if xr is None:
            raise ImportError("xarray is required to load netcdf/zarr data")

        path = Path(self.cfg.path)
        if self.cfg.store_format == "zarr":
            ds = xr.open_zarr(path.as_posix(), consolidated=False)
        elif self.cfg.store_format == "netcdf":
            ds = xr.open_dataset(path.as_posix(), engine="netcdf4")
        else:
            raise ValueError("data.store_format must be one of: zarr, netcdf")

        if self.cfg.variable not in ds:
            raise KeyError(f"Variable '{self.cfg.variable}' not found in dataset")

        arr = ds[self.cfg.variable]
        if self.cfg.time_dim not in arr.dims:
            raise KeyError(f"time_dim '{self.cfg.time_dim}' not found in variable dims")

        arr = arr.transpose(self.cfg.time_dim, ...)
        data = arr.values
        return data.reshape(data.shape[0], -1)

    def setup(self, stage: Optional[str] = None) -> None:
        full_ds = SequenceDataset(
            data=self._load_array(),
            input_steps=self.cfg.input_steps,
            pred_steps=self.cfg.pred_steps,
        )
        n_total = len(full_ds)
        n_val = int(n_total * self.cfg.val_ratio)
        n_test = int(n_total * self.cfg.test_ratio)
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError("Split sizes produced empty train set; adjust ratios")

        self._train_ds, self._val_ds, self._test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.cfg.split_seed),
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("DataModule not set up. Call setup() before train_dataloader().")
        return DataLoader(
            self._train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("DataModule not set up. Call setup() before val_dataloader().")
        return DataLoader(
            self._val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("DataModule not set up. Call setup() before test_dataloader().")
        return DataLoader(
            self._test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
