from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from hydra.errors import InstantiationException
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer, seed_everything

from lewidi.common.instantiate import (
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)
from lewidi.data.datamodules import LeWiDiMultiTaskDataModule
from lewidi.data.datasets import DatasetName


if TYPE_CHECKING:
    from lewidi.data import LeWiDiDataModule


CONFIGS_ROOT = Path(__file__).parent.parent.parent.joinpath("configs/")


OmegaConf.register_new_resolver("DatasetName", lambda x: DatasetName[x])
OmegaConf.register_new_resolver("path", Path, replace=True)


def get_all_tasks_for_multitask_model(datamodule: LeWiDiDataModule) -> set[str] | None:
    """Get all the task names for the multitask model."""
    if not isinstance(datamodule, LeWiDiMultiTaskDataModule):
        return None

    datamodule.prepare_data()
    datamodule.setup("fit")
    model_tasks = datamodule.get_all_task_names()

    return model_tasks


@hydra.main(version_base="1.3", config_path=str(CONFIGS_ROOT), config_name="train.yaml")
def main(config: DictConfig) -> None:
    """Load the config and run the experiment."""
    # Set the seed for everything.
    seed = config.get("seed")
    if seed:
        seed_everything(seed, workers=True)

    logger.info("Instantiating datamodule...")
    datamodule: LeWiDiDataModule = hydra.utils.instantiate(config.data)

    # If we should dynamically create the tasks for the model
    model_tasks = get_all_tasks_for_multitask_model(datamodule)

    logger.info("Instantiating model...")
    try:
        model: LightningModule = hydra.utils.instantiate(config.model, tasks=model_tasks)
    except InstantiationException:
        model = hydra.utils.instantiate(config.model)

    logger.info("Instantiating callbacks...")
    experiment_callbacks = instantiate_callbacks(config.get("callbacks"))

    logger.info("Instantiating loggers...")
    experiment_loggers = instantiate_loggers(config.get("loggers"))

    logger.info("Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=experiment_callbacks, logger=experiment_loggers
    )

    if experiment_loggers:
        logger.info("Logging hyperparameters...")
        log_hyperparameters(
            {
                "config": config,
                "datamodule": datamodule,
                "model": model,
                "callbacks": experiment_callbacks,
                "logger": experiment_loggers,
                "trainer": trainer,
            }
        )

    if config.get("train", False):
        logger.info("Starting model training")
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.get("checkpoint_path"))

    if config.get("test", False):
        logger.info("Starting testing")
        ckpt_path = trainer.checkpoint_callback.best_model_path  # pyright: ignore

        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
