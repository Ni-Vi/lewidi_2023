from enum import Enum


class DatasetLanguage(Enum):
    """Language for the dataset."""

    english = "en"
    arabic = "ar"


class DatasetName(Enum):
    """Dataset names used in modelling."""

    conv_abuse = "ConvAbuse"
    hs_brexit = "HS-Brexit"
    armis = "ArMIS"
    md_agreement = "MD-Agreement"
