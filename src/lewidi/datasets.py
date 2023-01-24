import itertools
import json
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, validator


class DatasetName(Enum):
    """Domains used for modelling."""

    conv_abuse = "ConvAbuse"
    hs_brexit = "HS-Brexit"
    armis = "ArMIS"
    md_agreement = "MD-Agreement"


class Dataset(BaseModel, extra="allow"):
    """Dataset thing."""

    dataset: DatasetName
    text: str
    annotation_task: str = Field(..., alias="annotation task")
    annotation_count: int = Field(..., alias="number of annotations")
    annotations: list[int]
    annotators: set[str]
    lang: Literal["en", "ar"]
    hard_label: int
    soft_label: dict[int, float]
    split: Literal["train", "dev", "test"]

    @validator("annotations", pre=True)
    @classmethod
    def convert_annotations(cls, annotations: str | list[int]) -> list[int]:
        """Convert the annotations if not in the right form."""
        if isinstance(annotations, str):
            return [int(annotation) for annotation in annotations.split(",")]
        return annotations

    @validator("annotators", pre=True)
    @classmethod
    def convert_annotators(cls, annotators: str | Iterable[str]) -> set[str]:
        """Convert the annotations if not in the right form."""
        if isinstance(annotators, str):
            return set(annotators.split(","))
        return set(annotators)

    @property
    def utterance(self) -> str:
        """Get the utterance from the text.

        TODO: Add more details about why this is needed.
        """
        if self.dataset == DatasetName.conv_abuse:
            search_term = '"user:"'
            start_index = self.text.rfind(search_term)
            return self.text[start_index + len(search_term)].strip().lstrip('"').rstrip('"}')

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


def load_dataset(file_path: Path, dataset_name: DatasetName) -> pd.DataFrame:
    """Load dataset from the file."""
    # Load all the raw data
    raw_data: dict[int, dict[str, Any]] = json.loads(file_path.read_bytes())

    # Parse and validate all the data
    parsed_data = (
        Dataset.parse_obj({"dataset_name": dataset_name, **raw_data_item})
        for raw_data_item in raw_data.values()
    )

    # Convert the data to the dataframe
    parsed_data_for_dataframe = {
        idx: instance.as_dataframe_dict() for idx, instance in enumerate(parsed_data)
    }
    return pd.DataFrame.from_dict(parsed_data_for_dataframe)


def get_all_annotators(dataset_list: list[Dataset]) -> set[str]:
    """Get a unique set of all the annotators from a loaded dataset."""
    all_annotators = (instance.annotators for instance in dataset_list)
    unique_annotators = itertools.chain.from_iterable(all_annotators)
    return set(unique_annotators)


def split_dataset_for_annotators(
    instances: list[Dataset],
    dataframe: pd.DataFrame,
    dataset_name: DatasetName,
    is_test_set: bool = False,
) -> pd.DataFrame:
    """Split the dataset across the annotators."""
