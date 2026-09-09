"""Describe a DataFrame without changing it.

The profile is plain data (dataclasses that convert to dicts) so that later
components can consume it programmatically instead of parsing prose.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes

# Share of non-null values that must parse before a text column is reported
# as a likely datetime or as a mistyped numeric column.
DETECTION_THRESHOLD = 0.9


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column facts that apply to every kind of column."""

    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int


@dataclass(frozen=True)
class NumericStats:
    """Summary statistics for one numeric column."""

    name: str
    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None
    median: float | None


@dataclass(frozen=True)
class CategoricalStats:
    """Summary of one categorical (text, boolean or category) column."""

    name: str
    unique_count: int
    most_frequent: Any
    most_frequent_count: int


@dataclass(frozen=True)
class DatetimeColumn:
    """A column that holds dates, either by dtype or by content."""

    name: str
    is_datetime_dtype: bool
    parsed_ratio: float


@dataclass(frozen=True)
class DatasetProfile:
    """Structured description of a dataset."""

    row_count: int
    column_count: int
    column_names: list[str]
    dtypes: dict[str, str]
    duplicate_row_count: int
    columns: list[ColumnProfile]
    numeric_columns: list[NumericStats]
    categorical_columns: list[CategoricalStats]
    datetime_columns: list[DatetimeColumn]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as nested dicts and lists, ready to serialise."""
        return asdict(self)


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    """Build a profile of `frame`.

    The frame is only read: no values are cleaned, cast or dropped, and
    columns that merely look like dates are reported as metadata instead of
    being converted.
    """
    row_count = len(frame)
    columns_data = [frame.iloc[:, position] for position in range(frame.shape[1])]

    columns = [_profile_column(series, row_count) for series in columns_data]
    duplicate_row_count = int(frame.duplicated().sum())
    datetime_columns = _detect_datetime_columns(columns_data)
    numeric_like = _numeric_like_text_columns(
        columns_data, {column.name for column in datetime_columns}
    )

    return DatasetProfile(
        row_count=row_count,
        column_count=frame.shape[1],
        column_names=[str(name) for name in frame.columns],
        dtypes={str(series.name): str(series.dtype) for series in columns_data},
        duplicate_row_count=duplicate_row_count,
        columns=columns,
        numeric_columns=[
            _numeric_stats(series) for series in columns_data if _is_numeric(series)
        ],
        categorical_columns=[
            _categorical_stats(series)
            for series in columns_data
            if _is_categorical(series)
        ],
        datetime_columns=datetime_columns,
        warnings=_collect_warnings(
            columns, duplicate_row_count, datetime_columns, numeric_like, row_count
        ),
    )


def _is_numeric(series: pd.Series) -> bool:
    """True for real numbers; booleans are treated as categorical instead."""
    return pdtypes.is_numeric_dtype(series) and not pdtypes.is_bool_dtype(series)


def _is_categorical(series: pd.Series) -> bool:
    """True for text, boolean and category columns."""
    if pdtypes.is_bool_dtype(series):
        return True
    if pdtypes.is_numeric_dtype(series):
        return False
    return not pdtypes.is_datetime64_any_dtype(series)


def _is_text(series: pd.Series) -> bool:
    return pdtypes.is_object_dtype(series) or pdtypes.is_string_dtype(series)


def _profile_column(series: pd.Series, row_count: int) -> ColumnProfile:
    missing = int(series.isna().sum())
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        non_null_count=row_count - missing,
        missing_count=missing,
        missing_percentage=round(missing / row_count * 100, 2) if row_count else 0.0,
        unique_count=int(series.nunique(dropna=True)),
    )


def _numeric_stats(series: pd.Series) -> NumericStats:
    values = series.dropna()
    return NumericStats(
        name=str(series.name),
        count=int(values.count()),
        mean=_as_float(values.mean()),
        std=_as_float(values.std()),
        minimum=_as_float(values.min()),
        maximum=_as_float(values.max()),
        median=_as_float(values.median()),
    )


def _categorical_stats(series: pd.Series) -> CategoricalStats:
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return CategoricalStats(
            name=str(series.name),
            unique_count=0,
            most_frequent=None,
            most_frequent_count=0,
        )
    return CategoricalStats(
        name=str(series.name),
        unique_count=int(series.nunique(dropna=True)),
        most_frequent=_as_python(counts.index[0]),
        most_frequent_count=int(counts.iloc[0]),
    )


def _detect_datetime_columns(columns_data: list[pd.Series]) -> list[DatetimeColumn]:
    """Find columns holding dates, by dtype or by parsing their text.

    A text column is only reported when nearly every value parses as a date
    and the column does not also look numeric, which keeps bare years and
    numeric identifiers from being mistaken for dates.
    """
    detected: list[DatetimeColumn] = []
    for series in columns_data:
        if pdtypes.is_datetime64_any_dtype(series):
            detected.append(DatetimeColumn(str(series.name), True, 1.0))
            continue
        if not _is_text(series):
            continue
        date_ratio = _parse_ratio(series, _coerce_datetime)
        if date_ratio < DETECTION_THRESHOLD:
            continue
        if _parse_ratio(series, _coerce_numeric) >= DETECTION_THRESHOLD:
            continue
        detected.append(DatetimeColumn(str(series.name), False, round(date_ratio, 2)))
    return detected


def _numeric_like_text_columns(
    columns_data: list[pd.Series], datetime_names: set[str]
) -> list[str]:
    """Text columns whose values are almost all numbers."""
    return [
        str(series.name)
        for series in columns_data
        if _is_text(series)
        and str(series.name) not in datetime_names
        and _parse_ratio(series, _coerce_numeric) >= DETECTION_THRESHOLD
    ]


def _parse_ratio(
    series: pd.Series, converter: Callable[[pd.Series], pd.Series]
) -> float:
    """Share of non-null values that `converter` can parse."""
    values = series.dropna()
    if values.empty:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            converted = converter(values)
        except (ValueError, TypeError):
            return 0.0
    return int(converted.notna().sum()) / len(values)


def _coerce_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _coerce_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _collect_warnings(
    columns: list[ColumnProfile],
    duplicate_row_count: int,
    datetime_columns: list[DatetimeColumn],
    numeric_like: list[str],
    row_count: int,
) -> list[str]:
    """Describe problems worth a human's attention. Nothing is modified."""
    messages: list[str] = []

    if duplicate_row_count:
        messages.append(f"{duplicate_row_count} duplicate row(s) found.")

    for column in columns:
        if not column.missing_count:
            continue
        if row_count and column.missing_count == row_count:
            messages.append(f"Column {column.name!r} is entirely missing.")
        else:
            messages.append(
                f"Column {column.name!r} has {column.missing_count} missing "
                f"value(s) ({column.missing_percentage}%)."
            )

    for detected in datetime_columns:
        if not detected.is_datetime_dtype:
            messages.append(
                f"Column {detected.name!r} is stored as text but looks like a datetime."
            )

    messages.extend(
        f"Column {name!r} is stored as text but looks numeric." for name in numeric_like
    )
    return messages


def _as_float(value: Any) -> float | None:
    """Convert a numpy or pandas number to a float, mapping NaN to None."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _as_python(value: Any) -> Any:
    """Convert a numpy or pandas scalar to a plain Python value."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
