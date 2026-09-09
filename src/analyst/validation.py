"""Checks that run before and after a dataset file is read."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from analyst.errors import (
    DatasetNotFoundError,
    EmptyDatasetError,
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
