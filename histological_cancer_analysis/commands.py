from __future__ import annotations

import json
from collections.abc import Sequence

import fire
from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from histological_cancer_analysis.evaluate import evaluate as run_evaluate
from histological_cancer_analysis.infer import infer as run_infer
from histological_cancer_analysis.train import train as run_train

CONFIG_MODULE = "configs"
CONFIG_NAME = "config"


def compose_config(overrides: Sequence[str] = ()) -> DictConfig:
    with initialize_config_module(version_base=None, config_module=CONFIG_MODULE):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def train(*overrides: str) -> None:
    config = compose_config(overrides)
    run_train(config)


def evaluate(*overrides: str) -> None:
    config = compose_config(overrides)
    metrics = run_evaluate(config)
    print(json.dumps(metrics, indent=2))


def infer(*overrides: str) -> None:
    config = compose_config(overrides)
    result = run_infer(config)
    print(json.dumps(result, indent=2))


def main() -> None:
    fire.Fire({"evaluate": evaluate, "infer": infer, "train": train})


__all__ = ["compose_config", "evaluate", "infer", "main", "train"]
