from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field


if TYPE_CHECKING:
    from arabert.preprocess import ArabertPreprocessor
    from transformers import BatchEncoding, PreTrainedTokenizerBase


class Tokenizer(Protocol):
    """Common tokenizer for LeWiDi datamodules."""

    def __call__(self, text: str) -> BatchEncoding:
        """Tokenize input text."""
        ...


@dataclass
class BERTTokenizer(Tokenizer):
    """BERT Tokenizer for LeWiDi datamodules."""

    tokenizer: PreTrainedTokenizerBase
    tokenize_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, text: str) -> BatchEncoding:
        """Tokenize text using BERT."""
        return self.tokenizer(text, **self.tokenize_kwargs).convert_to_tensors(tensor_type="pt")


@dataclass
class AraBERTTokenizer(Tokenizer):
    """AraBERT tokenizer for LeWiDi datamodules."""

    tokenizer: PreTrainedTokenizerBase
    preprocessor: ArabertPreprocessor
    tokenize_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, text: str) -> BatchEncoding:
        """Tokenize arabic text."""
        preprocessed_text = self.preprocessor.preprocess(text)
        return self.tokenizer(preprocessed_text, **self.tokenize_kwargs).convert_to_tensors(tensor_type="pt")


@dataclass
class TFIDFTokenizer(Tokenizer):
    """TFIDF tokenizer for LeWiDi datamodules."""

    def __call__(self, text: str) -> BatchEncoding:
        """Tokenize text using TF-IDF."""
        raise NotImplementedError
