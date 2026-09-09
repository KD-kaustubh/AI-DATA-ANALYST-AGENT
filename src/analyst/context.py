"""Turn a DatasetProfile into the schema summary a model is given.

The model needs to know which columns exist and what they hold; it does not
need the data. Only the profile is used here, so the description can never
drift from the real dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from analyst.profiling import DatasetProfile

# A handful of warnings is context; the whole list is noise.
MAX_WARNINGS = 5


@dataclass(frozen=True)
class ColumnSummary:
    """One column as the model sees it."""

    name: str
    dtype: str
    kind: str
    missing_count: int
    unique_count: int
    example_value: Any = None


@dataclass(frozen=True)
class DatasetContext:
    """Everything the model is told about the dataset."""

    row_count: int
    column_count: int
    columns: list[ColumnSummary]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_text(self) -> str:
        """Render the context as the compact block that goes in the prompt."""
        lines = [
            f"Rows: {self.row_count}",
            f"Columns: {self.column_count}",
            "",
            "Columns:",
        ]
        for column in self.columns:
            details = [f"type {column.dtype}", column.kind]
            if column.missing_count:
                details.append(f"{column.missing_count} missing")
            if column.example_value is not None:
                details.append(f"example {column.example_value!r}")
            lines.append(f"- {column.name} ({', '.join(details)})")

        if self.datetime_columns:
            lines.append("")
            lines.append(f"Date columns: {', '.join(self.datetime_columns)}")
        if self.warnings:
            lines.append("")
            lines.append("Data quality notes:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


def build_context(profile: DatasetProfile) -> DatasetContext:
    """Summarise a profile for the model."""
    numeric = [stats.name for stats in profile.numeric_columns]
    categorical = [stats.name for stats in profile.categorical_columns]
    datetimes = [column.name for column in profile.datetime_columns]
    examples = {
        stats.name: stats.most_frequent for stats in profile.categorical_columns
    }

    columns = [
        ColumnSummary(
            name=column.name,
            dtype=column.dtype,
            kind=_kind(column.name, numeric, datetimes),
            missing_count=column.missing_count,
            unique_count=column.unique_count,
            example_value=examples.get(column.name),
        )
        for column in profile.columns
    ]

    return DatasetContext(
        row_count=profile.row_count,
        column_count=profile.column_count,
        columns=columns,
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetimes,
        warnings=profile.warnings[:MAX_WARNINGS],
    )


def _kind(name: str, numeric: list[str], datetimes: list[str]) -> str:
    """Label a column so the model picks compatible tools."""
    if name in datetimes:
        return "date"
    if name in numeric:
        return "numeric"
    return "categorical"
