from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from pytorch_lightning import LightningModule

from lewidi.models.components import SingleTaskMetric


if TYPE_CHECKING:
    from jaxtyping import Float
    from transformers import PreTrainedModel

    from lewidi.data import SingleTaskModelInstance
    from lewidi.models.components import ClassifierHead


class LeWiDiSingleTaskModel(LightningModule):
    """Single-task model for LeWiDi."""

    def __init__(
        self,
        backbone: PreTrainedModel,
        head: ClassifierHead,
        # optimizer: torch.optim.Optimizer,
        optimizer: Any,
    ) -> None:
        super().__init__()
        self._backbone = backbone
        self._head = head

        self._optimizer = optimizer

        self._criterion = torch.nn.CrossEntropyLoss()

        self._train_metrics = SingleTaskMetric("train")
        self._val_metrics = SingleTaskMetric("val")
        self._test_metrics = SingleTaskMetric("test")

    def training_step(self, batch: SingleTaskModelInstance) -> torch.Tensor:
        """Evaluate batch during training and return the loss."""
        # Get model output
        logits = self._step(batch)

        prob_per_class = torch.sigmoid(logits)
        loss = self._criterion(prob_per_class, batch["target_hard_label"])

        # Update metrics
        self._train_metrics.update(
            loss, logits, batch["target_hard_label"], batch["target_soft_label"]
        )
        self.log_dict(self._train_metrics.compute(), prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch: SingleTaskModelInstance, _: int) -> torch.Tensor:
        """Evaluate batch during validation and return the loss."""
        # Get model output
        logits = self._step(batch)

        prob_per_class = torch.sigmoid(logits)
        loss = self._criterion(prob_per_class, batch["target_hard_label"])

        # Update metrics
        self._val_metrics.update(
            loss, logits, batch["target_hard_label"], batch["target_soft_label"]
        )
        self.log_dict(self._val_metrics.compute(), prog_bar=True, logger=True)

        return loss

    def test_step(self, batch: SingleTaskModelInstance) -> torch.Tensor:
        """Evaluate batch during validation and return the loss."""
        # Get model output
        logits = self._step(batch)

        prob_per_class = torch.sigmoid(logits)
        loss = self._criterion(prob_per_class, batch["target_hard_label"])

        # Update metrics
        self._test_metrics.update(
            loss, logits, batch["target_hard_label"], batch["target_soft_label"]
        )
        self.log_dict(self._test_metrics.compute(), prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training."""
        return self._optimizer(params=self.parameters())

    def _step(self, batch: SingleTaskModelInstance) -> Float[torch.Tensor, batch_size labels]:
        """Take a step with the model and return the logits."""
        backbone_output: Float[torch.Tensor, batch_size sequence_length dim] = self._backbone(
            input_ids=batch["token_ids"],
            attention_mask=batch["token_mask"],
        ).pooler_output

        logits: Float[torch.Tensor, batch_size labels] = self._head(backbone_output)

        return logits
