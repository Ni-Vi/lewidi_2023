from pathlib import Path

import pandas as pd
from lightning import LightningDataModule

from lewidi.datasets import DatasetName, create_dataframe_from_instances, load_dataset


class LeWiDiDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: DatasetName,
        train_data_path: Path,
        val_data_path: Path,
        test_data_path: Path,
        batch_size: int,
        pretrained_model: str,
        processed_data_root: Path,
        force_data_preparation: bool = False,
    ) -> None:
        self._dataset_name = dataset_name

        self._pretrained_model = pretrained_model

        self._train_data_path = train_data_path
        self._val_data_path = val_data_path
        self._test_data_path = test_data_path

        self._batch_size = batch_size

        self._processed_data_dir = processed_data_root.joinpath(self._dataset_name.value)
        self._force_data_preparation = force_data_preparation

    def prepare_data(self) -> None:
        """Prepare processed data for modelling."""
        # Create the directory for processed data if it doesnt exist
        self._processed_data_dir.mkdir(parents=True, exist_ok=True)

        # Create datafranes for each data split
        train_dataset = self._prepare_instance_for_dataset_split(self._train_data_path)
        val_dataset = self._prepare_instance_for_dataset_split(self._val_data_path)

        processed_train_data_path = self._processed_data_dir.joinpath(self._train_data_path.name)

        # Process every file if not done already
        train_instances = create_dataframe_from_instances(
            load_dataset(self._train_data_path, self._dataset_name)
        )

    def _prepare_instance_for_dataset_split(self, data_path: Path) -> pd.DataFrame:
        """Prepare the data for the split and cache it."""
        processed_data_path = self._processed_data_dir.joinpath(self._train_data_path.name)

        # If the file exists, just load it
        if processed_data_path.exists() or self._force_data_preparation:
            return pd.read_json(processed_data_path)

        # If the file doesnt exist, create the dataframe and cache it
        instances_dataframe = create_dataframe_from_instances(
            load_dataset(self._train_data_path, self._dataset_name)
        )
        instances_dataframe.to_json(processed_data_path)

        return instances_dataframe
