from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rich.pretty import pprint as rich_print
from sklearn.metrics import f1_score as compute_f1_score

from lewidi.models.components.metrics import compute_cross_entropy


if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from torchdata.datapipes.map import MapDataPipe

    from lewidi.data.datamodules import LeWiDiDataModule
    from lewidi.data.datasets.constants import DatasetName


class LeWiDiSVMModel:
    """Train an SVM model on the hard labels."""

    def __init__(
        self,
        dataset_name: DatasetName,
        vectorizer: TfidfVectorizer,
        model: LogisticRegression,
        datamodule: LeWiDiDataModule,
    ) -> None:
        self._dataset_name = dataset_name
        self._vectorizer = vectorizer
        self._model = model
        self._datamodule = datamodule

    def train(self) -> None:
        """Train the SVM model."""
        # Extract data from instances
        dataset_instances = self._datamodule.prepare_dataset_instances(
            self._datamodule.raw_data_paths.train
        )
        train_utterances: MapDataPipe[str] = dataset_instances.map(
            lambda instance: instance.utterance
        )
        training_labels: MapDataPipe[int] = dataset_instances.map(
            lambda instance: instance.hard_label
        )

        # Prepare the vectorizer
        train_vectorizer = self._vectorizer.fit_transform(train_utterances)

        # Prepare the training targets
        train_targets = np.asarray(list(training_labels))  # pyright: ignore

        # Train the model
        self._model = self._model.fit(train_vectorizer, train_targets)

    def test(self) -> None:
        """Test the SVM model and get metrics."""
        dataset_instances = self._datamodule.prepare_dataset_instances(
            self._datamodule.raw_data_paths.test
        )

        test_utterances: MapDataPipe[str] = dataset_instances.map(
            lambda instance: instance.utterance
        )
        test_labels: MapDataPipe[int] = dataset_instances.map(lambda instance: instance.hard_label)

        test_vectorizer = self._vectorizer.transform(test_utterances)
        test_targets = np.asarray(list(test_labels))  # pyright: ignore

        predictions = self._model.predict(test_vectorizer)

        f1_score = compute_f1_score(test_targets, predictions, average="micro")
        cross_entropy = compute_cross_entropy(
            torch.from_numpy(test_targets), torch.from_numpy(predictions)
        )
        rich_print(f"[{self._dataset_name.name}] F1: {f1_score} / cross-entropy: {cross_entropy}")
