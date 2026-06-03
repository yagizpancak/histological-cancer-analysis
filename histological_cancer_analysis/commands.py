from __future__ import annotations

from collections.abc import Sequence

import fire
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from histological_cancer_analysis.train import train as run_train

CONFIG_MODULE = "configs"
CONFIG_NAME = "config"


def compose_config(overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_module(version_base=None, config_module=CONFIG_MODULE):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def train(*overrides: str) -> None:
    config = compose_config(overrides)
    run_train(config)


def main() -> None:
    fire.Fire({"train": train})


__all__ = ["compose_config", "main", "train"]
