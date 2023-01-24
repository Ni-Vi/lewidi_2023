import os
import random
from enum import Enum

import numpy as np
import torch
from pydantic import BaseSettings


class Task(Enum):
    """Types of tasks."""

    multi_label = "multi_label"
    multi_task = "multi_task"
    single = "single"


class Predict(Enum):
    """Types of predictions from the model."""

    label = "label"
    mc_predict = "mc_predict"


class LeWiDiHyperparameters(BaseSettings):
    """Hyperparameters for the models."""

    batch_size: int = 16
    learning_rate: float = 1e-7
    max_length: int = 128
    num_epochs: int = 20
    random_state: int = 9999
    task: Task = Task.multi_label
    predict: Predict = Predict.label
    seed: int = 12345
    # mc_passes = 10
    # ar_dat = 0
    # sort_by = None
    # batch_weight = None

    def set_python_seeds(self) -> None:
        """Set all the seeds."""
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
