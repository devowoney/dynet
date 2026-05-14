import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dynet import train


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "dynet" / "configs"


class HydraLightningSetupTests(unittest.TestCase):
    def test_component_configs_include_targets(self):
        component_files = [
            CONFIG_ROOT / "data" / "sine.yaml",
            CONFIG_ROOT / "model" / "mlp.yaml",
            CONFIG_ROOT / "trainer" / "default.yaml",
        ]

        for config_file in component_files:
            with config_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            self.assertIn("_target_", data, msg=f"missing _target_ in {config_file}")

    def test_root_config_declares_hydra_defaults(self):
        with (CONFIG_ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        self.assertEqual(data["defaults"][:3], [{"data": "sine"}, {"model": "mlp"}, {"trainer": "default"}])

    def test_build_components_uses_hydra_instantiate_order(self):
        cfg = {
            "data": {"_target_": "dynet.data.SineWaveDataModule"},
            "model": {"_target_": "dynet.model.MLPRegressor"},
            "trainer": {"_target_": "pytorch_lightning.Trainer"},
        }

        with patch("dynet.train.instantiate", side_effect=["dm", "m", "t"]) as mocked:
            trainer, model, datamodule = train.build_components(cfg)

        self.assertEqual((trainer, model, datamodule), ("t", "m", "dm"))
        self.assertEqual(mocked.call_count, 3)


if __name__ == "__main__":
    unittest.main()
