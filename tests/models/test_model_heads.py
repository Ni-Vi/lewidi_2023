import pytest
import torch

from lewidi.models.components import ClassifierHead, Head


@pytest.mark.xfail
def test_head_module_raises_error() -> None:
    head = Head()

    encoding = torch.randn((2, 500))
    head(encoding)
    head(encoding)


def test_classifier_head_input_and_output_works() -> None:
    batch_size = 2
    num_labels = 2
    dim = 500

    head = ClassifierHead(dim, num_labels)

    encoding = torch.randn((batch_size, dim))
    model_output = head(encoding)

    assert len(model_output.shape) == 2
    assert model_output.size(0) == batch_size
    assert model_output.size(1) == num_labels
