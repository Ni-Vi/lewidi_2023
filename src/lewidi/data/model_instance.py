from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    import torch


class ModelInstance(TypedDict):
    """Instance provided to the model for modelling.

    This is also the instance used when evaluating the test set.
    """

    token_ids: torch.Tensor
    token_mask: torch.Tensor

    target_hard_label: torch.Tensor
    target_soft_label: torch.Tensor


class SingleTaskModelInstance(ModelInstance):
    """Instance provided to the single-task model for modelling."""


class MultiTaskModelInstance(ModelInstance):
    """Instance provided to the multi-task model."""

    target_hard_label_per_task: dict[str, torch.Tensor]
