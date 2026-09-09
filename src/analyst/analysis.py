"""Deterministic analysis operations over a DataFrame.

Each function validates its inputs, does the work in pandas, and returns an
AnalysisResult describing what was done. Nothing here interprets natural
language: callers choose the operation, pandas computes the answer.

Operations never modify the frame they are given.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from pandas.api import types as pdtypes

from analyst.conversion import to_optional_float, to_python
from analyst.errors import InvalidOperationError
from analyst.validation import require_columns, require_numeric

AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")
# Aggregations that make no sense on text.
NUMERIC_AGGREGATIONS = ("sum", "mean", "median")
COMPARISON_OPERATORS = ("==", "!=", ">", ">=", "<", "<=")
MEMBERSHIP_OPERATORS = ("in", "not_in")
OPERATORS = COMPARISON_OPERATORS + MEMBERSHIP_OPERATORS
PERIODS = ("year", "month")
CORRELATION_METHODS = ("pearson", "spearman", "kendall")

_COMPARISONS: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@dataclass(frozen=True)
class Condition:
    """One structured filter test, such as `revenue >= 100`."""

    column: str
    operator: str
    value: Any

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Condition":
        """Build a Condition from a plain dict."""
        missing = {"column", "operator", "value"} - set(mapping)
        if missing:
            raise InvalidOperationError(
                f"Condition is missing: {', '.join(sorted(missing))}"
            )
        return cls(mapping["column"], mapping["operator"], mapping["value"])


@dataclass(frozen=True)
class AnalysisResult:
    """What was asked, what it was asked of, and what came back."""

    operation: str
    columns: list[str]
    parameters: dict[str, Any]
    rows: list[dict[str, Any]]
    row_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as nested dicts and lists, ready to serialise."""
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        """Rebuild the result rows as a DataFrame, for charting or chaining."""
        return pd.DataFrame(self.rows)


def filter_rows(
    frame: pd.DataFrame,
    conditions: Sequence[Condition | Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> AnalysisResult:
    """Keep the rows matching every condition.

    Conditions are structured values, never expressions, so nothing here
    evaluates caller-supplied code.
    """
    parsed = [_as_condition(condition) for condition in conditions]
    if not parsed:
        raise InvalidOperationError("At least one condition is required.")
    _check_limit(limit)
    require_columns(frame, [condition.column for condition in parsed])

    mask = pd.Series(True, index=frame.index)
    for condition in parsed:
        mask &= _condition_mask(frame[condition.column], condition)

    matched = frame[mask]
    kept = matched if limit is None else matched.head(limit)
    return AnalysisResult(
        operation="filter",
        columns=[condition.column for condition in parsed],
        parameters={
            "conditions": [to_python(asdict(item)) for item in parsed],
            "limit": limit,
        },
        rows=_records(kept),
        row_count=len(kept),
        metadata={
            "source_rows": len(frame),
            "matched_rows": len(matched),
            "truncated": len(kept) < len(matched),
        },
    )


def sort_rows(
    frame: pd.DataFrame,
    by: str | Sequence[str],
    ascending: bool | Sequence[bool] = True,
    *,
    limit: int | None = None,
) -> AnalysisResult:
    """Sort by one or more columns."""
    columns = _as_list(by)
    if not columns:
        raise InvalidOperationError("At least one sort column is required.")
    _check_limit(limit)
    require_columns(frame, columns)

    orders = (
        list(ascending)
        if isinstance(ascending, (list, tuple))
        else [bool(ascending)] * len(columns)
    )
    if len(orders) != len(columns):
        raise InvalidOperationError(
            f"Got {len(orders)} sort direction(s) for {len(columns)} column(s)."
        )

    # A stable sort keeps ties in their original order, so results repeat.
    ordered = frame.sort_values(by=columns, ascending=orders, kind="stable")
    kept = ordered if limit is None else ordered.head(limit)
    return AnalysisResult(
        operation="sort",
        columns=columns,
        parameters={"by": columns, "ascending": orders, "limit": limit},
        rows=_records(kept),
        row_count=len(kept),
        metadata={"source_rows": len(frame), "truncated": len(kept) < len(ordered)},
    )


def group_aggregate(
    frame: pd.DataFrame,
    group_by: str | Sequence[str],
    aggregations: Mapping[str, str | Sequence[str]],
) -> AnalysisResult:
    """Group by one or more columns and aggregate other columns.

    `aggregations` maps a column to one aggregation or a list of them, for
    example `{"revenue": ["sum", "mean"], "units": "sum"}`.
    """
    group_columns = _as_list(group_by)
    if not group_columns:
        raise InvalidOperationError("At least one grouping column is required.")
    require_columns(frame, group_columns)

    spec = _aggregation_spec(frame, aggregations)
    grouped = frame.groupby(group_columns, dropna=False, sort=True).agg(spec)
    grouped.columns = [f"{column}_{name}" for column, name in grouped.columns]
    grouped = grouped.reset_index()

    return AnalysisResult(
        operation="group_aggregate",
        columns=group_columns + list(spec),
        parameters={"group_by": group_columns, "aggregations": spec},
        rows=_records(grouped),
        row_count=len(grouped),
        metadata={"source_rows": len(frame), "group_count": len(grouped)},
    )


def describe_numeric(
    frame: pd.DataFrame, columns: str | Sequence[str] | None = None
) -> AnalysisResult:
    """Summarise numeric columns; defaults to every numeric column."""
    selected = _as_list(columns) if columns is not None else _numeric_columns(frame)
    if not selected:
        raise InvalidOperationError("No numeric columns to describe.")
    require_columns(frame, selected)
    require_numeric(frame, selected)

    rows = []
    for name in selected:
        values = frame[name].dropna()
        rows.append(
            {
                "column": str(name),
                "count": int(values.count()),
                "mean": to_optional_float(values.mean()),
                "median": to_optional_float(values.median()),
                "std": to_optional_float(values.std()),
                "minimum": to_optional_float(values.min()),
                "maximum": to_optional_float(values.max()),
            }
        )

    return AnalysisResult(
        operation="describe",
        columns=[str(name) for name in selected],
        parameters={"columns": [str(name) for name in selected]},
        rows=rows,
        row_count=len(rows),
        metadata={"source_rows": len(frame)},
    )


def value_counts(
    frame: pd.DataFrame, column: str, *, limit: int | None = None
) -> AnalysisResult:
    """Count how often each value appears in a column."""
    _check_limit(limit)
    require_columns(frame, [column])

    series = frame[column]
    counts = series.value_counts(dropna=True)
    total = int(series.count())
    rows = [
        {
            "value": to_python(value),
            "count": int(count),
            "percentage": round(int(count) / total * 100, 2) if total else 0.0,
        }
        for value, count in counts.items()
    ]
    kept = rows if limit is None else rows[:limit]

    return AnalysisResult(
        operation="value_counts",
        columns=[str(column)],
        parameters={"column": str(column), "limit": limit},
        rows=kept,
        row_count=len(kept),
        metadata={
            "distinct_values": len(rows),
            "non_null_count": total,
            "missing_count": int(series.isna().sum()),
            "truncated": len(kept) < len(rows),
        },
    )


def correlation(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    method: str = "pearson",
) -> AnalysisResult:
    """Correlate numeric columns and return the matrix row by row.

    With exactly two columns the single pairwise value is also reported in
    the metadata.
    """
    if method not in CORRELATION_METHODS:
        raise InvalidOperationError(
            f"Unknown method '{method}'. Supported: {', '.join(CORRELATION_METHODS)}"
        )

    selected = _as_list(columns) if columns is not None else _numeric_columns(frame)
    require_columns(frame, selected)
    require_numeric(frame, selected)
    if len(selected) < 2:
        raise InvalidOperationError("Correlation needs at least two numeric columns.")

    matrix = frame[selected].corr(method=method)
    rows = [
        {"column": str(name)}
        | {str(other): to_optional_float(matrix.at[name, other]) for other in selected}
        for name in selected
    ]

    metadata: dict[str, Any] = {"method": method, "source_rows": len(frame)}
    if len(selected) == 2:
        metadata["correlation"] = to_optional_float(matrix.iat[0, 1])

    return AnalysisResult(
        operation="correlation",
        columns=[str(name) for name in selected],
        parameters={"columns": [str(name) for name in selected], "method": method},
        rows=rows,
        row_count=len(rows),
        metadata=metadata,
    )


def group_by_period(
    frame: pd.DataFrame,
    datetime_column: str,
    period: str,
    aggregations: Mapping[str, str | Sequence[str]] | None = None,
) -> AnalysisResult:
    """Group rows by year or month of an explicitly named datetime column.

    Text dates are parsed on a copy; the caller's frame is left alone. Rows
    whose date cannot be parsed are dropped from the grouping and counted in
    the metadata.
    """
    if period not in PERIODS:
        raise InvalidOperationError(
            f"Unknown period '{period}'. Supported: {', '.join(PERIODS)}"
        )
    require_columns(frame, [datetime_column])

    moments = _as_datetime(frame[datetime_column], datetime_column)
    working = frame.copy()
    working["__period__"] = moments.dt.strftime("%Y" if period == "year" else "%Y-%m")
    usable = working[working["__period__"].notna()]
    if usable.empty:
        raise InvalidOperationError(
            f"Column '{datetime_column}' holds no dates that could be parsed."
        )

    if aggregations:
        spec = _aggregation_spec(frame, aggregations)
        grouped = usable.groupby("__period__", sort=True).agg(spec)
        grouped.columns = [f"{column}_{name}" for column, name in grouped.columns]
    else:
        spec = {}
        grouped = usable.groupby("__period__", sort=True).size().to_frame("row_count")

    grouped = grouped.reset_index().rename(columns={"__period__": "period"})
    return AnalysisResult(
        operation="group_by_period",
        columns=[str(datetime_column)] + list(spec),
        parameters={
            "datetime_column": str(datetime_column),
            "period": period,
            "aggregations": spec,
        },
        rows=_records(grouped),
        row_count=len(grouped),
        metadata={
            "source_rows": len(frame),
            "unparsed_rows": len(working) - len(usable),
            "period_count": len(grouped),
        },
    )


def _as_condition(condition: Condition | Mapping[str, Any]) -> Condition:
    if isinstance(condition, Condition):
        return condition
    if isinstance(condition, Mapping):
        return Condition.from_mapping(condition)
    raise InvalidOperationError(
        f"Expected a Condition or a dict, got {type(condition).__name__}."
    )


def _condition_mask(series: pd.Series, condition: Condition) -> pd.Series:
    """Build the boolean mask for a single condition."""
    if condition.operator in MEMBERSHIP_OPERATORS:
        if not isinstance(condition.value, (list, tuple, set)):
            raise InvalidOperationError(
                f"Operator '{condition.operator}' needs a list of values, "
                f"got {type(condition.value).__name__}."
            )
        inside = series.isin(list(condition.value))
        return inside if condition.operator == "in" else ~inside

    if condition.operator not in _COMPARISONS:
        raise InvalidOperationError(
            f"Unknown operator '{condition.operator}'. Supported: {', '.join(OPERATORS)}"
        )

    try:
        return _COMPARISONS[condition.operator](series, condition.value)
    except (TypeError, ValueError) as exc:
        raise InvalidOperationError(
            f"Cannot compare column '{condition.column}' ({series.dtype}) "
            f"with {condition.value!r}."
        ) from exc


def _aggregation_spec(
    frame: pd.DataFrame, aggregations: Mapping[str, str | Sequence[str]]
) -> dict[str, list[str]]:
    """Validate an aggregation mapping and normalise it to lists."""
    if not aggregations:
        raise InvalidOperationError("At least one aggregation is required.")

    spec: dict[str, list[str]] = {}
    for column, requested in aggregations.items():
        names = _as_list(requested)
        if not names:
            raise InvalidOperationError(f"No aggregation given for '{column}'.")
        unknown = [name for name in names if name not in AGGREGATIONS]
        if unknown:
            raise InvalidOperationError(
                f"Unknown aggregation(s): {', '.join(unknown)}. "
                f"Supported: {', '.join(AGGREGATIONS)}"
            )
        spec[column] = names

    require_columns(frame, list(spec))
    numeric_needed = [
        column
        for column, names in spec.items()
        if any(name in NUMERIC_AGGREGATIONS for name in names)
    ]
    require_numeric(frame, numeric_needed)
    return spec


def _as_datetime(series: pd.Series, column: str) -> pd.Series:
    """Return the column as datetimes without touching the original."""
    if pdtypes.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            converted = pd.to_datetime(series, errors="coerce")
        except (TypeError, ValueError) as exc:
            raise InvalidOperationError(
                f"Column '{column}' ({series.dtype}) does not hold dates."
            ) from exc
    if not converted.notna().any():
        raise InvalidOperationError(
            f"Column '{column}' ({series.dtype}) does not hold dates."
        )
    return converted


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(name)
        for name in frame.columns
        if pdtypes.is_numeric_dtype(frame[name]) and not pdtypes.is_bool_dtype(frame[name])
    ]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert rows to JSON-safe dicts."""
    return [
        {str(name): to_python(value) for name, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _check_limit(limit: int | None) -> None:
    if limit is None:
        return
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise InvalidOperationError(f"limit must be a positive integer, got {limit!r}.")
