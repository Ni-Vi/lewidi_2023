from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


if TYPE_CHECKING:
    from jaxtyping import Float, Int


def compute_cross_entropy(
    predicted_soft_labels: torch.Tensor,
    target_soft_labels: torch.Tensor,
    epsilon: float = 1e-12,
    epsilon2: float = 1e-9,
) -> torch.Tensor:
    """Compute cross entropy as done for the shared task."""
    # Clip all values to (eps, 1-eps)
    predictions = torch.clamp(predicted_soft_labels, epsilon, 1 - epsilon)

    batch_size = predictions.size(0)

    cross_entropy = -torch.sum(target_soft_labels * torch.log(predictions + epsilon2)) / batch_size

    return cross_entropy


def extract_soft_labels_from_task_hard_labels(
    hard_labels_per_task: torch.Tensor, num_classes: int = 2
) -> torch.Tensor:
    """Extract soft labels from hard labels."""
    binned_labels = torch.bincount(hard_labels_per_task, minlength=num_classes)
    soft_label_distribution = binned_labels / binned_labels.sum()
    return soft_label_distribution


def get_relevant_hard_labels_from_predictions(
    logits_per_task: dict[str, Float[torch.Tensor, batch_size labels]],
    target_label_per_task: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    """For all the predicted labels, only keep the hard labels where we have a target."""
    # Ensure the inputs are sorted properly
    logits_per_task = dict(sorted(logits_per_task.items()))
    target_label_per_task = dict(sorted(target_label_per_task.items()))

    # Convert the logits to a single tensor
    logits: Float[torch.Tensor, batch_size tasks labels] = torch.stack(
        list(logits_per_task.values()), dim=1
    )
    # Get the prediction per task from the logits
    prediction_per_task: Int[torch.Tensor, batch_size tasks] = logits.sigmoid().argmax(-1)

    # Convert the target label per task into a single tensor
    targets: Int[torch.Tensor, batch_size tasks] = torch.stack(
        list(target_label_per_task.values()), dim=1
    )
    # Get a mask of all the predictions which should be ignored in the aggregation
    prediction_per_task_mask = targets.ge(0)

    predictions_per_example: list[torch.Tensor] = []

    for pred, mask in zip(
        prediction_per_task.unbind(0), prediction_per_task_mask.unbind(0), strict=True
    ):
        # Aggregate the prediction over the targets that matter for the current example
        predicted_hard_label = pred[mask]
        predictions_per_example.append(predicted_hard_label)

    return predictions_per_example


def aggregate_hard_labels_from_predictions(
    hard_labels_per_example: list[torch.Tensor],
) -> torch.Tensor:
    """Aggregate hard labels per example."""
    hard_labels_list = [example_labels.mode()[0] for example_labels in hard_labels_per_example]

    return torch.stack(hard_labels_list, dim=0)


@dataclass
class TaskMetric:
    """Compute metric for the given task."""

    prefix: str
    loss: MeanMetric = MeanMetric()
    f1_score: MulticlassF1Score = MulticlassF1Score(num_classes=2, ignore_index=-100)

    def update(self, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the metrics."""
        self.loss.update(loss)
        self.f1_score.update(logits, targets)

    def reset(self) -> None:
        """Reset metrics."""
        self.loss.reset()
        self.f1_score.reset()

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics for the logger."""
        return {
            f"{self.prefix}/loss": self.loss.compute(),
            f"{self.prefix}/f1": self.f1_score.compute(),
        }


@dataclass
class SingleTaskMetric(TaskMetric):
    """Compute metrics for the single-task model."""

    cross_entropy: MeanMetric = MeanMetric()

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        target_hard_label: torch.Tensor,
        target_soft_label: torch.Tensor,
    ) -> None:
        """Update the metrics."""
        self.loss.update(loss)
        self.f1_score.update(logits, target_hard_label)

        # Convert logits to a prob dist
        predicted_soft_labels = logits.softmax(-1)
        self.cross_entropy.update(compute_cross_entropy(predicted_soft_labels, target_soft_label))

    def reset(self) -> None:
        """Reset metrics."""
        self.loss.reset()
        self.f1_score.reset()
        self.cross_entropy.reset()

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics for the logger."""
        return {
            f"{self.prefix}/loss": self.loss.compute(),
            f"{self.prefix}/f1": self.f1_score.compute(),
            f"{self.prefix}/cross_entropy": self.cross_entropy.compute(),
        }


@dataclass
class MultiTaskMetric:
    """Compute aggregated metrics for the multi-task model."""

    prefix: str
    loss: MeanMetric = MeanMetric()
    cross_entropy: MeanMetric = MeanMetric()
    f1_score: BinaryF1Score = BinaryF1Score()

    def update(
        self,
        loss: torch.Tensor,
        logits_per_task: dict[str, torch.Tensor],
        target_hard_label_per_task: dict[str, torch.Tensor],
        target_hard_label: torch.Tensor,
        target_soft_label: torch.Tensor,
    ) -> None:
        """Update the metrics."""
        self.loss.update(loss)

        relevant_hard_labels = get_relevant_hard_labels_from_predictions(
            logits_per_task, target_hard_label_per_task
        )
        predicted_hard_labels = aggregate_hard_labels_from_predictions(relevant_hard_labels)
        self.f1_score.update(predicted_hard_labels, target_hard_label)

        predicted_soft_labels = torch.stack(
            [
                extract_soft_labels_from_task_hard_labels(example_labels)
                for example_labels in relevant_hard_labels
            ],
            dim=0,
        )
        self.cross_entropy.update(compute_cross_entropy(predicted_soft_labels, target_soft_label))

    def reset(self) -> None:
        """Reset metrics."""
        self.loss.reset()
        self.f1_score.reset()
        self.cross_entropy.reset()

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics for the logger."""
        return {
            f"{self.prefix}/loss": self.loss.compute(),
            f"{self.prefix}/f1": self.f1_score.compute(),
            f"{self.prefix}/cross_entropy": self.cross_entropy.compute(),
        }
