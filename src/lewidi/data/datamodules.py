from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import torch
from pydantic import BaseModel, FilePath, parse_file_as
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import Concater
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper

from lewidi.data.datasets import DatasetInstance, DatasetName
from lewidi.data.datasets.dataset_instance import parse_instances_from_raw_data
from lewidi.data.model_instance import MultiTaskModelInstance, SingleTaskModelInstance


if TYPE_CHECKING:
    from pathlib import Path

    from lewidi.data.tokenizers import Tokenizer

T = TypeVar("T", SingleTaskModelInstance, MultiTaskModelInstance)


class DatasetSplitPaths(BaseModel):
    """File path for a given dataset split."""

    train: FilePath
    val: FilePath
    test: FilePath


class DataLoaderOptions(BaseModel, extra="allow"):
    """Various options that are passed to the DataLoader.

    Any additional options (i.e. kwargs) are also passed.
    """

    batch_size: int
    shuffle: bool
    num_workers: int = 2
    drop_last: bool = True


class Stage(str, Enum):
    """Stages under which the datamodule can be prepared with.

    Need to subclass str to make this a `StrEnum` because I don't want to start overriding methods
    within the datamodule and this is the cleanest and simplest way to enforce some semblance of
    type.
    """

    fit = "fit"
    test = "test"


class DatasetNotPreparedError(AssertionError):
    """Raise assertion if the dataset is not prepared."""

    def __init__(self, stage: Stage) -> None:
        super().__init__(f"The `setup()` method must be called with `stage='{stage.value}'`")


class LeWiDiDataModule(Generic[T], LightningDataModule):
    """Prepare and load data for the LeWiDi model experiments."""

    def __init__(
        self,
        dataset_name: DatasetName,
        raw_data_paths: DatasetSplitPaths,
        tokenizer: Tokenizer,
        processed_data_root: Path,
        dataloader_options: DataLoaderOptions,
        *,
        force_data_preparation: bool = False,
    ) -> None:
        super().__init__()
        self._dataset_name = dataset_name
        self._tokenizer = tokenizer

        self.raw_data_paths = raw_data_paths

        self._dataloader_options = dataloader_options

        self._processed_data_dir = processed_data_root
        self._force_data_preparation = force_data_preparation

        self._train_dataset: MapDataPipe[T] | None = None
        self._val_dataset: MapDataPipe[T] | None = None
        self._test_dataset: MapDataPipe[T] | None = None

    def prepare_data(self) -> None:
        """Prepare processed data for modelling."""
        # Create the directory for processed data if it doesnt exist
        self._processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset for each data split
        self._prepare_dataset_split(self.raw_data_paths.train)
        self._prepare_dataset_split(self.raw_data_paths.val)
        self._prepare_dataset_split(self.raw_data_paths.test)

    def setup(self, stage: str) -> None:
        """Setup the datamodule."""
        stage = Stage(stage)
        if stage == Stage.fit:
            self._train_dataset = self._prepare_dataset_split(self.raw_data_paths.train)
            self._val_dataset = self._prepare_dataset_split(self.raw_data_paths.val)
            self._test_dataset = self._prepare_dataset_split(self.raw_data_paths.test)

        if stage == Stage.test:
            self._test_dataset = self._prepare_dataset_split(self.raw_data_paths.test)

    def train_dataloader(self) -> DataLoader[T]:
        """Create a dataloader over the training data."""
        if self._train_dataset is None:
            raise DatasetNotPreparedError(Stage.fit)
        return DataLoader(self._train_dataset, **self._dataloader_options.dict())

    def val_dataloader(self) -> DataLoader[T]:
        """Create a dataloader over the validation data."""
        if self._val_dataset is None:
            raise DatasetNotPreparedError(Stage.fit)
        return DataLoader(self._val_dataset, **self._dataloader_options.dict())

    def test_dataloader(self) -> DataLoader[T]:
        """Create a dataloader over the test data."""
        if self._test_dataset is None:
            raise DatasetNotPreparedError(Stage.test)
        return DataLoader(self._test_dataset, **self._dataloader_options.dict())

    def _prepare_dataset_split(self, path: Path) -> MapDataPipe[T]:
        """Prepare the datasets for the dataset split.

        The choice of dataset split results in different preparations of the model instances.
        Training and validation instances have targets, whereas test instances do not.
        """
        dataset_instances = self.prepare_dataset_instances(path)
        return self._prepare_model_instances_with_targets(dataset_instances)

    def prepare_dataset_instances(self, path: Path) -> MapDataPipe[DatasetInstance]:
        """Prepare dataset instances from the given path."""
        processed_data_path = self._processed_data_dir.joinpath(path.name)

        # If the dataset instances do not already exist, or we want to force preparation
        if not processed_data_path.exists() or self._force_data_preparation:
            dataset_instances = parse_instances_from_raw_data(path, self._dataset_name)
            # Cache the dataset instances

        # Otherwise just load the instances from the file
        else:
            dataset_instances = parse_file_as(list[DatasetInstance], path)

        # Convert into a pipe
        dataset_instances_pipe = SequenceWrapper(dataset_instances)
        # Cast for mypy
        return cast(MapDataPipe[DatasetInstance], dataset_instances_pipe)

    def _prepare_model_instances_with_targets(
        self,
        dataset_instances: MapDataPipe[DatasetInstance],
    ) -> MapDataPipe[T]:
        """Prepare model instances with the targets included.

        This is likely used for training/validation instances.
        """
        model_instances_pipe = dataset_instances.map(
            self._convert_dataset_instance_to_model_instance
        )

        # Cast for mypy
        return cast(MapDataPipe[T], model_instances_pipe)

    def _convert_dataset_instance_to_model_instance(self, dataset_instance: DatasetInstance) -> T:
        """Convert dataset instance to a model instance."""
        raise NotImplementedError


class LeWiDiSingleTaskDataModule(LeWiDiDataModule[SingleTaskModelInstance]):
    """DataModule to prepare instances for the single-task model."""

    def _convert_dataset_instance_to_model_instance(
        self, dataset_instance: DatasetInstance
    ) -> SingleTaskModelInstance:
        """Convert dataset instance to single-task model instance."""
        tokens = self._tokenizer(dataset_instance.utterance)
        return SingleTaskModelInstance(
            token_ids=tokens.data["input_ids"],
            token_mask=tokens.data["attention_mask"],
            target_hard_label=torch.tensor(dataset_instance.hard_label, dtype=torch.long),
            target_soft_label=torch.tensor(
                list(dataset_instance.soft_label.values()), dtype=torch.float
            ),
        )


class LeWiDiMultiTaskDataModule(LeWiDiDataModule[MultiTaskModelInstance]):
    """DataModule to prepare instances for the multi-task model."""

    def get_all_task_names(self) -> set[str]:
        """Get all the possible task names from all instances.

        Iterate over all the loaded datasets and just keep the task names from the targets.
        """
        datapipes = []

        if self._train_dataset:
            datapipes.append(self._train_dataset.to_iter_datapipe())
        if self._val_dataset:
            datapipes.append(self._val_dataset.to_iter_datapipe())
        if self._test_dataset:
            datapipes.append(self._test_dataset.to_iter_datapipe())

        if not datapipes:
            raise AssertionError(
                "There are no datasets to iterate over to extract the tasks from.",
            )

        # Merge all the datasets into a single datapipe
        combined_datapipe = Concater(*datapipes)

        # Flatmap over all of them to get the list of tasks
        all_tasks = combined_datapipe.flatmap(
            lambda instance: list(instance["hard_label_per_task"].keys()),
        )
        unique_tasks: set[str] = set(all_tasks)

        return unique_tasks

    def _convert_dataset_instance_to_model_instance(
        self, dataset_instance: DatasetInstance
    ) -> MultiTaskModelInstance:
        """Convert dataset instance to the multi-task model instance."""
        annotation_per_annotators = {
            annotator: annotation if annotation is not None else -100
            for annotator, annotation in dataset_instance.annotation_per_annotators.items()
        }
        annotation_per_annotators = {
            annotator: torch.as_tensor(annotation)
            for annotator, annotation in annotation_per_annotators.items()
        }
        # Sort by keys
        annotation_per_annotators = dict(sorted(annotation_per_annotators.items()))

        tokens = self._tokenizer(dataset_instance.utterance)
        return MultiTaskModelInstance(
            token_ids=tokens.data["input_ids"],
            token_mask=tokens.data["attention_mask"],
            target_hard_label=torch.tensor(dataset_instance.hard_label, dtype=torch.long),
            target_soft_label=torch.tensor(
                list(dataset_instance.soft_label.values()), dtype=torch.float
            ),
            target_hard_label_per_task=annotation_per_annotators,
        )
