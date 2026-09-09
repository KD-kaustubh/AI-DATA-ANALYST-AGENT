"""AI Data Analyst Agent package."""

from analyst.errors import (
    DatasetError,
    DatasetNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
)
from analyst.loader import load_dataset
from analyst.profiling import (
    CategoricalStats,
    ColumnProfile,
    DatasetProfile,
    DatetimeColumn,
    NumericStats,
    profile_dataset,
)
from analyst.validation import SUPPORTED_EXTENSIONS

__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "CategoricalStats",
    "ColumnProfile",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetProfile",
    "DatasetReadError",
    "DatetimeColumn",
    "EmptyDatasetError",
    "NumericStats",
    "UnsupportedFileTypeError",
    "load_dataset",
    "profile_dataset",
]
