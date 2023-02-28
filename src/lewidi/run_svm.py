from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from lewidi.data.datamodules import DataLoaderOptions, DatasetSplitPaths, LeWiDiDataModule
from lewidi.data.datasets import DatasetName
from lewidi.data.tokenizers import TFIDFTokenizer
from lewidi.models import LeWiDiSVMModel


def main(dataset_name: DatasetName) -> None:
    """Train and test the data on an SVM model."""
    vectorizer = TfidfVectorizer(
        analyzer="word", min_df=0.00007, max_df=0.2, max_features=300  # pyright: ignore
    )
    clf = LogisticRegression(warm_start=True, verbose=1, solver="liblinear", C=10, max_iter=1000)

    raw_data_root = Path("storage/data/raw/").joinpath(dataset_name.value)
    data_paths = DatasetSplitPaths(
        train=raw_data_root.joinpath(f"{dataset_name.value}_train.json"),
        val=raw_data_root.joinpath(f"{dataset_name.value}_dev.json"),
        test=raw_data_root.joinpath(f"{dataset_name.value}_test.json"),
    )
    processed_data_root = Path("storage/data/processed/").joinpath(dataset_name.value)

    datamodule = LeWiDiDataModule(
        dataset_name,
        data_paths,
        TFIDFTokenizer(),
        processed_data_root,
        DataLoaderOptions(batch_size=1, shuffle=False, num_workers=0, drop_last=False),
    )

    model = LeWiDiSVMModel(vectorizer, clf, datamodule)

    model.train()
    model.test()


if __name__ == "__main__":
    main(DatasetName.conv_abuse)
