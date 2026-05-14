import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestScaffoldFiles(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        expected = [
            ROOT / "configs" / "config.yaml",
            ROOT / "configs" / "data" / "netcdf_zarr.yaml",
            ROOT / "configs" / "model" / "netcdf_zarr.yaml",
            ROOT / "configs" / "trainer" / "default.yaml",
            ROOT / "src" / "dynet" / "train.py",
            ROOT / "src" / "dynet" / "data" / "manual_datamodule.py",
            ROOT / "src" / "dynet" / "models" / "netcdf_zarr_module.py",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing expected file: {path}")


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("lightning")
    and importlib.util.find_spec("numpy"),
    "torch/lightning/numpy not installed",
)
class TestManualDataModuleCore(unittest.TestCase):
    def test_sequence_dataset_windows(self) -> None:
        import numpy as np

        from dynet.data.manual_datamodule import SequenceDataset

        data = np.arange(20, dtype=np.float32).reshape(10, 2)
        ds = SequenceDataset(data=data, input_steps=3, pred_steps=2)
        self.assertEqual(len(ds), 6)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (3, 2))
        self.assertEqual(tuple(y.shape), (2, 2))

