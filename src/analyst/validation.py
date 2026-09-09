"""Checks on dataset files and on the frames and columns we analyse."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from pandas.api import types as pdtypes

from analyst.errors import (
    ColumnNotFoundError,
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidOperationError,
    UnsupportedFileTypeError,
)

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


def validate_source(path: str | os.PathLike[str]) -> Path:
    """Check that `path` is a readable dataset file and return it as a Path.

    Raises DatasetNotFoundError if the path is missing or is not a file,
    UnsupportedFileTypeError for extensions we cannot read, and
    EmptyDatasetError for zero-byte files.
    """
    source = Path(path)

    if not source.exists():
        raise DatasetNotFoundError(f"No such file: {source}")
    if not source.is_file():
        raise DatasetNotFoundError(f"Not a file: {source}")

    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        found = source.suffix or source.name
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{found}'. Supported types: {supported}"
        )

    if source.stat().st_size == 0:
        raise EmptyDatasetError(f"File is empty: {source}")

    return source


def validate_frame(frame: pd.DataFrame, source: Path) -> None:
    """Reject a file that parsed correctly but carries no data.

    A header-only sheet counts as empty: there is nothing to analyse.
    """
    if frame.columns.empty or frame.empty:
        raise EmptyDatasetError(f"No data rows found in: {source}")


def require_columns(frame: pd.DataFrame, columns: Sequence[Any]) -> None:
    """Raise ColumnNotFoundError if any of `columns` is not in the frame."""
    missing = [str(name) for name in columns if name not in frame.columns]
    if missing:
        available = ", ".join(str(name) for name in frame.columns)
        raise ColumnNotFoundError(
            f"Column(s) not found: {', '.join(missing)}. Available: {available}"
        )


def require_numeric(frame: pd.DataFrame, columns: Sequence[Any]) -> None:
    """Raise InvalidOperationError for columns that are not real numbers.

    Booleans are rejected: they are counted as categorical everywhere else.
    """
    wrong = [
        f"{name} ({frame[name].dtype})"
        for name in columns
        if not pdtypes.is_numeric_dtype(frame[name])
        or pdtypes.is_bool_dtype(frame[name])
    ]
    if wrong:
        raise InvalidOperationError(f"Numeric column(s) required, got: {', '.join(wrong)}")
