from enum import Enum
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe
from transformers import AutoTokenizer

from lewidi.datasets import DatasetName, InstanceLoader, ModelInstance


class Stage(str, Enum):  # noqa: WPS600
    """Stages under which the datamodule can be prepared with.

    Need to subclass str to make this a `StrEnum` because I don't want to start overriding methods
    within the datamodule and this is the cleanest and simplest way to enforce some semblance of
    type.
    """

    train = "train"
    test = "test"


class LeWiDiDataModule(LightningDataModule):
    """Prepare and load data for the LeWiDi model experiments."""

    def __init__(
        self,
        dataset_name: DatasetName,
        train_data_path: Path,
        val_data_path: Path,
        test_data_path: Path,
        batch_size: int,
        shuffle: bool,
        pretrained_model: str,
        processed_data_root: Path,
        num_workers: int = 2,
        drop_last: bool = True,
        force_data_preparation: bool = False,
    ) -> None:
        super().__init__()
        self._dataset_name = dataset_name

        self._pretrained_model = pretrained_model

        self._train_data_path = train_data_path
        self._val_data_path = val_data_path
        self._test_data_path = test_data_path

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._drop_last = drop_last

        self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model)

        self._processed_data_dir = processed_data_root.joinpath(self._dataset_name.value)
        self._force_data_preparation = force_data_preparation

        self._train_dataset: MapDataPipe[ModelInstance] | None = None
        self._val_dataset: MapDataPipe[ModelInstance] | None = None
        self._test_dataset: MapDataPipe[ModelInstance] | None = None

    def prepare_data(self) -> None:
        """Prepare processed data for modelling."""
        # Create the directory for processed data if it doesnt exist
        self._processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset for each data split
        self._prepare_dataset(self._train_data_path)
        self._prepare_dataset(self._val_data_path)

    def setup(self, stage: str) -> None:
        """Setup the datamodule."""
        stage = Stage(stage)
        if stage == Stage.train:
            self._train_dataset = self._prepare_dataset(self._train_data_path)
            self._val_dataset = self._prepare_dataset(self._val_data_path)
            self._test_dataset = self._prepare_dataset(self._test_data_path)

        if stage == Stage.test:
            self._test_dataset = self._prepare_dataset(self._test_data_path)

    def train_dataloader(self) -> DataLoader[ModelInstance]:
        """Create a dataloader over the training data."""
        if self._train_dataset is None:
            raise AssertionError("The `setup()` method must be called with `stage='train'`")
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
        )

    def val_dataloader(self) -> DataLoader[ModelInstance]:
        """Create a dataloader over the validation data."""
        if self._val_dataset is None:
            raise AssertionError("The `setup()` method must be called with `stage='train'`")
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
        )

    def test_dataloader(self) -> DataLoader[ModelInstance]:
        """Create a dataloader over the test data."""
        if self._test_dataset is None:
            raise AssertionError(
                "The `setup()` method must be called with `stage='test'` or `stage='train'`"
            )
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            drop_last=self._drop_last,
        )

    def _prepare_dataset(self, data_path: Path) -> MapDataPipe[ModelInstance]:
        """Prepare the data for the dataset split and cache it.

        It is easier to just use a datapipe instead of creating a boilerplate `Dataset` class, even
        though torchdata is in beta.
        """
        processed_data_path = self._processed_data_dir.joinpath(data_path.name)

        # If the model instances already exist, just load it
        if processed_data_path.exists() or not self._force_data_preparation:
            model_instances = InstanceLoader[ModelInstance].parse_file(processed_data_path)
            return model_instances.create_datapipe()

        # If the model instances don't exist, we need to create them.
        dataset_instances = InstanceLoader.from_raw_data(data_path, self._dataset_name)

        # Convert the dataset instances to model instances
        model_instances = dataset_instances.convert(self._tokenizer)

        # Save the model instances
        model_instances.save(processed_data_path)

        return model_instances.create_datapipe()
