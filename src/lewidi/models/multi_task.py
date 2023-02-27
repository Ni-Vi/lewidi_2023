from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch_lightning import LightningModule

from lewidi.models.components import MultiTaskMetric, TaskMetric


if TYPE_CHECKING:
    from jaxtyping import Float
    from transformers import PreTrainedModel

    from lewidi.data.model_instance import MultiTaskModelInstance
    from lewidi.models.components import ClassifierHead


class LeWiDiMultiTaskModel(LightningModule):
    """MultiTask model for LeWiDi."""

    def __init__(
        self,
        backbone: PreTrainedModel,
        heads: dict[str, ClassifierHead],
        optimizer: Any,
        loss_weight_per_task: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._backbone = backbone
        self._heads = torch.nn.ModuleDict(heads)

        self._optimizer = optimizer

        # Either use the provided loss weighting per task, or default to just 1.0
        self._loss_weight_per_task = (
            loss_weight_per_task
            if loss_weight_per_task
            else {task_name: 1.0 for task_name in heads}
        )

        self._criterion = torch.nn.CrossEntropyLoss()

        self._train_metrics = MultiTaskMetric(prefix="train")
        self._train_metrics_per_task = {task: TaskMetric(f"train/{task}") for task in self._heads}
        self._val_metrics = MultiTaskMetric(prefix="val")
        self._val_metrics_per_task = {task: TaskMetric(f"val/{task}") for task in self._heads}
        self._test_metrics = MultiTaskMetric(prefix="test")
        self._test_metrics_per_task = {task: TaskMetric(f"test/{task}") for task in self._heads}

    @classmethod
    def from_tasks(
        cls, tasks: set[str], template_head: Any, *args: Any, **kwargs: Any
    ) -> LeWiDiMultiTaskModel:
        """Create heads from tasks."""
        heads = {task: template_head() for task in tasks}
        return cls(*args, heads=heads, **kwargs)

    def training_step(self, batch: MultiTaskModelInstance) -> torch.Tensor:
        """Evaluate batch during training and return the loss."""
        return self._model_step(
            batch, per_task_metrics=self._train_metrics_per_task, step_metric=self._train_metrics
        )

    def validation_step(self, batch: MultiTaskModelInstance) -> torch.Tensor:
        """Evaluate batch during validation and return the loss."""
        return self._model_step(
            batch, per_task_metrics=self._val_metrics_per_task, step_metric=self._val_metrics
        )

    def test_step(self, batch: MultiTaskModelInstance) -> torch.Tensor:
        """Evaluate batch during testing and return the loss."""
        return self._model_step(
            batch, per_task_metrics=self._test_metrics_per_task, step_metric=self._test_metrics
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training."""
        return self._optimizer(params=self.parameters())

    def _model_step(
        self,
        batch: MultiTaskModelInstance,
        *,
        per_task_metrics: dict[str, TaskMetric],
        step_metric: MultiTaskMetric,
    ) -> torch.Tensor:
        """Evaluate batch during training and return the loss."""
        backbone_output: Float[torch.Tensor, "batch_size sequence_length dim"] = self._backbone(
            input_ids=batch["token_ids"],
            attention_mask=batch["token_mask"],
        ).pooler_output

        loss = torch.tensor(0, device=backbone_output.device, dtype=torch.float)

        logits_per_task: dict[str, Float[torch.Tensor, "batch_size labels"]] = {}

        for task in self._heads:
            task_logits: Float[torch.Tensor, "batch_size labels"] = self._heads[task](
                backbone_output
            )
            logits_per_task[task] = task_logits

            if torch.all(batch["target_hard_label_per_task"][task] < 0):
                continue

            task_prob_dist = torch.sigmoid(task_logits)
            task_loss = (
                self._criterion(task_prob_dist, target=batch["target_hard_label_per_task"][task])
                * self._loss_weight_per_task[task]
            )
            loss += task_loss

            per_task_metrics[task].update(
                loss, task_logits, batch["target_hard_label_per_task"][task]
            )

        # Update metrics
        step_metric.update(
            loss,
            logits_per_task,
            batch["target_hard_label_per_task"],
            batch["target_hard_label"],
            batch["target_soft_label"],
        )

        # Log metrics
        self.log_dict(step_metric.compute(), prog_bar=True, logger=True)
        for task_metric in per_task_metrics.values():
            self.log_dict(task_metric.compute(), prog_bar=True, logger=True)

        return loss
