"""AI Data Analyst Agent package."""

from analyst.analysis import (
    AGGREGATIONS,
    OPERATORS,
    PERIODS,
    AnalysisResult,
    Condition,
    correlation,
    describe_numeric,
    filter_rows,
    group_aggregate,
    group_by_period,
    sort_rows,
    value_counts,
)
from analyst.charts import (
    bar_chart,
    box_plot,
    correlation_heatmap,
    figure_to_png_bytes,
    histogram,
    line_chart,
    scatter_plot,
)
from analyst.errors import (
    AnalysisError,
    ColumnNotFoundError,
    DatasetError,
    DatasetNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    InvalidOperationError,
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
    "AGGREGATIONS",
    "OPERATORS",
    "PERIODS",
    "SUPPORTED_EXTENSIONS",
    "AnalysisError",
    "AnalysisResult",
    "CategoricalStats",
    "ColumnNotFoundError",
    "ColumnProfile",
    "Condition",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetProfile",
    "DatasetReadError",
    "DatetimeColumn",
    "EmptyDatasetError",
    "InvalidOperationError",
    "NumericStats",
    "UnsupportedFileTypeError",
    "bar_chart",
    "box_plot",
    "correlation",
    "correlation_heatmap",
    "describe_numeric",
    "figure_to_png_bytes",
    "filter_rows",
    "group_aggregate",
    "group_by_period",
    "histogram",
    "line_chart",
    "load_dataset",
    "profile_dataset",
    "scatter_plot",
    "sort_rows",
    "value_counts",
]
