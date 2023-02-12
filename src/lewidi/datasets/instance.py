import itertools
from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator
from transformers import BatchEncoding

from lewidi.datasets.constants import DatasetLanguage, DatasetName


def _fix_annotation(annotation: int, dataset_name: DatasetName) -> int:
    """Fix any annotations that are not 0 or 1."""
    if dataset_name != DatasetName.conv_abuse:
        return annotation

    if annotation < 0:
        return 1
    if annotation >= 0:
        return 0

    raise ValueError("Annotation value is not supported??")


class DatasetInstance(BaseModel, extra="allow", allow_population_by_field_name=True):
    """Common instance for any dataset."""

    dataset: DatasetName
    split: Literal["train", "dev", "test"]
    lang: DatasetLanguage

    text: str

    annotation_task: str = Field(..., alias="annotation task")
    annotation_count: int = Field(..., alias="number of annotations")
    annotation_per_annotators: dict[str, int | None]
    annotations: str | None
    annotators: str

    hard_label: int | None
    soft_label: dict[int, float | None]

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

    @property
    def tasks(self) -> list[str]:
        """Get the tasks associated with this instance and the dataset."""
        return self.annotators.split(",")

    def include_all_annotators_in_annotations_dict(self, all_annotators: set[str]) -> None:
        """Use set of all annotators to extend the dict of annotation per annotators.

        This is done because we want to know how every annotator has evaluated this instance, and
        if they haven't, it should be a None.
        """
        # Create dictionary of all annotators that did not annotate this instance
        update_dict = {
            annotator: None
            for annotator in all_annotators
            if annotator not in self.annotation_per_annotators
        }
        self.annotation_per_annotators.update(update_dict)


def get_all_annotators(instances: list[DatasetInstance]) -> set[str]:
    """Get a unique set of all the annotators throughout the instances."""
    all_annotators = (instance.annotation_per_annotators.keys() for instance in instances)
    unique_annotators = itertools.chain.from_iterable(all_annotators)
    return set(unique_annotators)


def include_all_annotators_in_all_instances(
    instances: list[DatasetInstance],
) -> list[DatasetInstance]:
    """Update each instance with all annotators from all instances."""
    all_annotators = get_all_annotators(instances)
    for instance in instances:
        instance.include_all_annotators_in_annotations_dict(all_annotators)

    return instances


class ModelInstance(BaseModel):
    """Instance provided to the model for modelling."""

    tokens: BatchEncoding
    tasks: list[str]
    annotation_per_annotator: dict[str, int]
    soft_label: dict[int, float | None]
    hard_label: int | None
