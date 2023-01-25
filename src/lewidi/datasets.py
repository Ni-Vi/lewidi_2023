import itertools
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import orjson
import pandas as pd
from pydantic import BaseModel, Field, root_validator


class DatasetLanguage(Enum):
    """Language for the dataset."""

    english = "en"
    arabic = "ar"


class DatasetName(Enum):
    """Domains used for modelling."""

    conv_abuse = "ConvAbuse"
    hs_brexit = "HS-Brexit"
    armis = "ArMIS"
    md_agreement = "MD-Agreement"


def _fix_annotation(annotation: int, dataset_name: DatasetName) -> int:
    """Fix any annotations that are not 0 or 1."""
    if dataset_name != DatasetName.conv_abuse:
        return annotation

    if annotation < 0:
        return 1
    if annotation >= 0:
        return 0

    raise ValueError("Annotation value is not supported??")


class Instance(BaseModel, extra="allow"):
    """Common instance for any dataset."""

    dataset: DatasetName
    text: str
    annotation_task: str = Field(..., alias="annotation task")
    annotation_count: int = Field(..., alias="number of annotations")
    annotation_per_annotators: dict[str, int | None]
    annotations: str | None
    annotators: str
    lang: DatasetLanguage
    hard_label: int | None
    soft_label: dict[int, float | None]
    split: Literal["train", "dev", "test"]

    @root_validator(pre=True)
    @classmethod
    def fix_empty_annotations(cls, raw_values: dict[str, Any]) -> dict[str, Any]:
        """Annotations can be an empty string, so convert those to None."""
        annotations: str | None = raw_values.get("annotations")

        if not annotations:
            raw_values["annotations"] = None

        return raw_values

    @root_validator(pre=True)
    @classmethod
    def create_annotation_per_annotator_dict(cls, raw_values: dict[str, Any]) -> dict[str, Any]:
        """Create a dictionary of each annotators value.

        This validator will automatically map each annotation to the annotator.
        """
        dataset_name: DatasetName | None = raw_values.get("dataset")
        annotations: str | None = raw_values.get("annotations")
        annotators: str | None = raw_values.get("annotators")

        if not dataset_name:
            raise AssertionError("There should be a dataset name for this instance?")

        if not annotators:
            raise AssertionError("There should be annotators for this instance.")

        annotator_list: list[str] = annotators.split(",")

        annotation_list: list[int | None]

        if annotations:
            annotation_list = [
                _fix_annotation(int(annotation), dataset_name)
                for annotation in annotations.split(",")
            ]
        else:
            annotation_list = [None for _ in annotator_list]

        raw_values["annotation_per_annotators"] = {
            annotator: annotation for annotator, annotation in zip(annotator_list, annotation_list)
        }

        return raw_values

    @property
    def utterance(self) -> str:
        """Get the utterance from the text.

        TODO: Add more details about why this is needed.
        """
        if self.dataset == DatasetName.conv_abuse:
            search_term = '"user":'
            start_index = self.text.rfind(search_term)
            return self.text[start_index + len(search_term) :].strip().lstrip('"').rstrip('"}')

        return self.text

    def as_dataframe_dict(self) -> dict[str, Any]:
        """Convert to the form used by the dataframe."""
        return {
            "text": self.utterance,
            "annotations": self.annotations,
            "annotators": self.annotators,
            "soft_label_0": self.soft_label[0],
            "soft_label_1": self.soft_label[1],
            "hard_label": self.hard_label,
        }


def load_dataset(file_path: Path, dataset_name: DatasetName) -> list[Instance]:
    """Load dataset from the file."""
    # Load all the raw data
    raw_data: dict[int, dict[str, Any]] = orjson.loads(
        orjson.dumps(json.loads(file_path.read_bytes()))
    )

    # Parse and validate all the data
    parsed_data = [
        Instance.parse_obj({"dataset": dataset_name, **raw_data_item})
        for raw_data_item in raw_data.values()
    ]

    return parsed_data


def get_all_annotators(dataset_instances: list[Instance]) -> set[str]:
    """Get a unique set of all the annotators from a loaded dataset."""
    all_annotators = (instance.annotation_per_annotators.keys() for instance in dataset_instances)
    unique_annotators = itertools.chain.from_iterable(all_annotators)
    return set(unique_annotators)


def create_annotator_annotation_dataframe(instances: list[Instance]) -> pd.DataFrame:
    """Create dataframe of annoators and their annotations per instance."""
    annotator_dict: dict[str, list[int | None]] = {
        annotator: [] for annotator in get_all_annotators(instances)
    }

    for instance in instances:
        annotation_per_annotators = instance.annotation_per_annotators

        for annotator in annotator_dict.keys():
            annotation = annotation_per_annotators.get(annotator)
            annotator_dict[annotator].append(annotation)

    annotator_annotation_dataframe = pd.DataFrame(annotator_dict)

    # Sort the columns of the dataframe so it's easier to read
    annotator_annotation_dataframe = annotator_annotation_dataframe.reindex(
        sorted(annotator_annotation_dataframe.columns), axis=1
    )

    return annotator_annotation_dataframe


def create_dataframe_from_instances(instances: list[Instance]) -> pd.DataFrame:
    """Create the dataframe of instances for the data.."""
    annotator_annotation_dataframe = create_annotator_annotation_dataframe(instances)
    instance_metadata_dataframe = pd.DataFrame(
        [instance.as_dataframe_dict() for instance in instances]
    )
    merged_dataframe = instance_metadata_dataframe.join(annotator_annotation_dataframe)

    return merged_dataframe


if __name__ == "__main__":
    file_path = Path("data/MD-Agreement_dataset/MD-Agreement_train.json")
    instances = load_dataset(file_path, DatasetName.md_agreement)
    dataset_dataframe = create_dataframe_from_instances(instances)
