from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from allennlp_light.modules import FeedForward
    from jaxtyping import Float


class Head(torch.nn.Module):
    """Base head which is inherited by other Heads."""

    def forward(
        self, encoding: Float[torch.Tensor, "batch ... dim"]
    ) -> Float[torch.Tensor, "batch ... label"]:
        """Project the encoded inputs to a set of logits over the output distribution."""
        raise NotImplementedError


class ClassifierHead(Head):
    """Classifier head that just classifies the input.

    This is inspired by AllenNLP's `ClassifierHead` (https://github.com/allenai/allennlp/blob/main/allennlp/models/heads/classifier_head.py).
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        feedforward: FeedForward | None = None,
    ) -> None:
        super().__init__()
        self._num_labels = num_labels
        self._feedforward = feedforward

        classifier_input_dim = (
            self._feedforward.get_output_dim() if self._feedforward is not None else input_dim
        )
        self._classification_layer = torch.nn.Linear(classifier_input_dim, num_labels)

    def forward(
        self, encoding: Float[torch.Tensor, "batch dim"]
    ) -> Float[torch.Tensor, "batch label"]:
        """Classifier the encoded input into logits over the labels."""
        if self._feedforward is not None:
            encoding = self._feedforward(encoding)

        logits = self._classification_layer(encoding)
        return logits
