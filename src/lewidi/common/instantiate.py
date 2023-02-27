from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import hydra
from loguru import logger as log
from omegaconf import DictConfig
from pytorch_lightning.utilities import rank_zero_only


if TYPE_CHECKING:
    from pytorch_lightning import Callback
    from pytorch_lightning.loggers import Logger


def instantiate_loggers(logger_config: DictConfig) -> list[Logger]:
    """Instantiate the loggers from the config."""
    logger_list: list[Logger] = []

    if not logger_config:
        log.warning("No logger configs found! Skipping...")
        return logger_list

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    # Iterate over all the configs and instantiate each
    for config in logger_config.values():
        if isinstance(config, DictConfig) and "_target_" in config:
            log.info(f"Instantiating logger <{config._target_}>")
            logger_list.append(hydra.utils.instantiate(config))

    return logger_list


def instantiate_callbacks(callbacks_config: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config."""
    callbacks_list: list[Callback] = []

    if not callbacks_config:
        log.warning("No callback configs found! Skipping..")
        return callbacks_list

    if not isinstance(callbacks_config, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for config in callbacks_config.values():
        if isinstance(config, DictConfig) and "_target_" in config:
            log.info(f"Instantiating callback <{config._target_}>")
            callbacks_list.append(hydra.utils.instantiate(config))

    return callbacks_list


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    hparams = {}

    cfg = object_dict["config"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")

    # hparams["extras"] = cfg.get("extras")

    # hparams["task_name"] = cfg.get("task_name")
    # hparams["tags"] = cfg.get("tags")
    # hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""
    log.info("Closing loggers...")

    # Close wandb runs if they exist
    with suppress(ModuleNotFoundError):
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()
