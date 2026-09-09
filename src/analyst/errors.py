"""Errors raised while loading and validating datasets."""


class DatasetError(Exception):
    """Base class for every dataset loading or validation failure."""


class DatasetNotFoundError(DatasetError):
    """The given path does not point to an existing file."""


class UnsupportedFileTypeError(DatasetError):
    """The file extension is not one we know how to read."""


class EmptyDatasetError(DatasetError):
    """The file exists but holds no usable rows."""


class DatasetReadError(DatasetError):
    """The file could not be parsed, usually because it is corrupted."""


class AnalysisError(Exception):
    """Base class for analysis and chart failures."""


class ColumnNotFoundError(AnalysisError):
    """A requested column is not in the dataset."""


class InvalidOperationError(AnalysisError):
    """The requested operation does not apply to this data."""
