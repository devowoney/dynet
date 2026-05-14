"""Hydra + Lightning training entrypoint."""

from __future__ import annotations

from typing import Any, Mapping

try:
    import hydra
    from hydra.utils import instantiate
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    hydra = None
    instantiate = None


def build_components(cfg: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    if instantiate is None:
        raise ModuleNotFoundError("hydra-core is required to build components")

    datamodule = instantiate(cfg["data"])
    model = instantiate(cfg["model"])
    trainer = instantiate(cfg["trainer"])
    return trainer, model, datamodule


if hydra is not None:

    @hydra.main(version_base=None, config_path="configs", config_name="config")
    def main(cfg: Mapping[str, Any]) -> None:
        trainer, model, datamodule = build_components(cfg)
        trainer.fit(model=model, datamodule=datamodule)

else:

    def main(cfg: Any = None) -> None:
        raise ModuleNotFoundError(
            "hydra-core is required to run dynet.train.main; install requirements first"
        )


if __name__ == "__main__":
    main()
