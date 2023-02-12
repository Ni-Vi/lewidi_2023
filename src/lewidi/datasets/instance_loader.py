import json
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import orjson
from pydantic import validator
from pydantic.generics import GenericModel
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper
from transformers import PreTrainedTokenizerBase

from lewidi.datasets.constants import DatasetName
from lewidi.datasets.instance import (
    DatasetInstance,
    ModelInstance,
    include_all_annotators_in_all_instances,
)


T = TypeVar("T", DatasetInstance, ModelInstance)


def convert_dataset_instance_to_model_instance(
    instance: DatasetInstance, tokenizer: PreTrainedTokenizerBase
) -> ModelInstance:
    """Convert a single dataset instance to a model instance."""
    annotation_per_annotators = {
        annotator: annotation
        for annotator, annotation in instance.annotation_per_annotators.items()
        if annotation is not None
    }
    return ModelInstance(
        tokens=tokenizer(instance.utterance),
        tasks=instance.tasks,
        annotation_per_annotator=annotation_per_annotators,
        soft_label=instance.soft_label,
        hard_label=instance.hard_label,
    )


class InstanceLoader(GenericModel, Generic[T], validate_assignment=True):
    """Generic Instance Loader with methods to help with handling cache."""

    instances: list[T]

    @classmethod
    def from_raw_data(
        cls, file_path: Path, dataset_name: DatasetName
    ) -> "InstanceLoader[DatasetInstance]":
        """Load all the instances from the raw data.

        Load the data using json, then dump and reload with orjson to automatically handle and
        remove all the NaN's since NaN is not actually valid under the [JSON
        schema](https://www.json.org/json-en.html).
        """
        raw_data: dict[int, dict[str, Any]] = orjson.loads(
            orjson.dumps(json.loads(file_path.read_bytes()))
        )

        # Parse and validate all the data
        instances = [
            DatasetInstance.parse_obj({"dataset": dataset_name, **raw_data_item})
            for raw_data_item in raw_data.values()
        ]

        instances = include_all_annotators_in_all_instances(instances)

        return InstanceLoader(instances=instances)

    @validator("instances")
    @classmethod
    def ensure_instances_are_same_type(cls, instances: list[T]) -> list[T]:
        """Ensure all the instances are the same type."""
        instance_type = type(instances[0])
        if not all([isinstance(instance, instance_type) for instance in instances]):
            raise AssertionError(
                "All instances within the loader are not of the type `ModelInstance`. This means that something has likely gone wrong somewhere and the list of instances was modified outside of the class."
            )

        return instances

    def save(self, file_path: Path) -> None:
        """Save the instances to disk."""
        data_as_bytes = orjson.dumps(self.dict())
        file_path.write_bytes(data_as_bytes)

    def convert(self, tokenizer: PreTrainedTokenizerBase) -> "InstanceLoader[ModelInstance]":
        """Convert all the DatasetInstances to ModelInstances."""
        # Use a partial function so the fn can be used with the mapper
        convert_instance_fn = partial(
            convert_dataset_instance_to_model_instance, tokenizer=tokenizer
        )

        # Process instances using a datapipe
        dataset_instance_pipe = IterableWrapper(self.instances)
        model_instance_pipe = dataset_instance_pipe.map(convert_instance_fn)

        # Output and return the new instance loader
        model_instances: list[ModelInstance] = list(model_instance_pipe)
        return InstanceLoader(instances=model_instances)

    def create_datapipe(self) -> MapDataPipe[T]:
        """Create a datapipe of the model instances.

        It is easier to just use a datapipe instead of creating a boilerplate `Dataset` class, even
        though torchdata is in beta.
        """
        data_pipe = SequenceWrapper(self.instances)
        # Cast for mypy
        return cast(MapDataPipe[T], data_pipe)
