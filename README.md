# dynet
Dev repository: ML-based dynamical system emulator

## Data-driven model development scaffold

This repository now includes a minimal Hydra + PyTorch Lightning setup:

- Hydra config tree at `dynet/configs`
- Synthetic data module: `dynet.data.SineWaveDataModule`
- Lightning model: `dynet.model.MLPRegressor`
- Training entrypoint: `python -m dynet.train`

Install dependencies first:

```bash
pip install -r requirements.txt
```
