# dynet
Dev repository: ML-based dynamical system emulator

## Default ML architecture scaffold

This repository now includes a default structure for:

1. PyTorch Lightning + Hydra configuration
2. End-to-end NetCDF/Zarr forecasting model wiring
3. Manual LightningDataModule and DataLoader construction

### Layout

- `configs/`: Hydra configs (`data`, `model`, `trainer`, and root defaults)
- `src/dynet/data/manual_datamodule.py`: manual data module and sequence dataset
- `src/dynet/models/netcdf_zarr_module.py`: Lightning forecasting module
- `src/dynet/train.py`: Hydra-driven training entrypoint

### Run

Install dependencies and run:

```bash
pip install -e .
python -m dynet.train data.path=/absolute/path/to/data.zarr data.store_format=zarr
```

For NetCDF:

```bash
python -m dynet.train data.path=/absolute/path/to/data.nc data.store_format=netcdf
```
