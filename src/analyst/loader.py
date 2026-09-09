"""Read CSV and Excel files into pandas DataFrames."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from analyst.errors import DatasetReadError, EmptyDatasetError
from analyst.validation import validate_frame, validate_source


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a CSV or XLSX file into a DataFrame.

    The file is validated before and after reading, so any failure surfaces
    as a DatasetError subclass rather than a raw pandas or openpyxl error.
    """
    source = validate_source(path)
    frame = _read(source)
    validate_frame(frame, source)
    return frame


def _read(source: Path) -> pd.DataFrame:
    """Dispatch to the reader for this file type, normalising failures."""
    try:
        if source.suffix.lower() == ".csv":
            return pd.read_csv(source)
        return pd.read_excel(source, engine="openpyxl")
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError(f"File is empty: {source}") from exc
    except Exception as exc:
        # pandas and openpyxl raise a wide range of parser, encoding and zip
        # errors for damaged files; callers only care that the read failed.
        raise DatasetReadError(f"Could not read {source}: {exc}") from exc
