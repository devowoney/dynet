from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from dynet.data.manual_datamodule import DataConfig, ManualDataModule
from dynet.models.netcdf_zarr_module import ModelConfig, NetCDFZarrForecastModule

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    log_every_n_steps: int = 10


def infer_input_dim(data_cfg: DataConfig) -> int:
    if not data_cfg.path:
        raise ValueError("data.path is required")
    if xr is None:
        raise ImportError("xarray is required to infer input dimensions")

    p = Path(to_absolute_path(data_cfg.path))
    if data_cfg.store_format == "zarr":
        ds = xr.open_zarr(p.as_posix(), consolidated=False)
    elif data_cfg.store_format == "netcdf":
        ds = xr.open_dataset(p.as_posix(), engine="netcdf4")
    else:
        raise ValueError("data.store_format must be one of: zarr, netcdf")

    var = ds[data_cfg.variable].transpose(data_cfg.time_dim, ...)
    return int(torch.tensor(var.values.shape[1:]).prod().item())


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg) -> None:
    L.seed_everything(int(cfg.seed), workers=True)
    data_cfg = DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    model_cfg = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    trainer_cfg = TrainerConfig(**OmegaConf.to_container(cfg.trainer, resolve=True))

    data_cfg.split_seed = int(cfg.seed)
    data_cfg.path = to_absolute_path(data_cfg.path)
    datamodule = ManualDataModule(data_cfg)
    model = NetCDFZarrForecastModule(
        input_dim=infer_input_dim(data_cfg),
        input_steps=data_cfg.input_steps,
        pred_steps=data_cfg.pred_steps,
        cfg=model_cfg,
    )

    trainer = L.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
