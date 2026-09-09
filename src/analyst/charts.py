"""Matplotlib charts for datasets and analysis results.

Figures are built with matplotlib's object-oriented API rather than pyplot,
so nothing is stored in a global figure registry. That keeps the module
headless-safe and equally usable from a script, a notebook, a future API
response or a future Streamlit app. Callers decide whether to display the
Figure, save it, or turn it into PNG bytes.
"""

from __future__ import annotations

import io
from typing import Any, Sequence

import pandas as pd
from matplotlib.figure import Figure

from analyst.analysis import AnalysisResult, correlation
from analyst.errors import InvalidOperationError
from analyst.validation import require_columns, require_numeric

DEFAULT_FIGSIZE = (8.0, 5.0)

ChartData = pd.DataFrame | AnalysisResult


def bar_chart(
    data: ChartData,
    x: str,
    y: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Draw `y` as bars, one per `x` category."""
    frame = _as_frame(data)
    require_columns(frame, [x, y])
    require_numeric(frame, [y])

    figure, axes = _new_figure(figsize)
    axes.bar(frame[x].astype(str), frame[y], color="#4C78A8")
    _label(axes, title or f"{y} by {x}", x, y)
    _rotate_labels(axes, frame[x].astype(str))
    return figure


def line_chart(
    data: ChartData,
    x: str,
    y: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Draw `y` against `x` as a line."""
    frame = _as_frame(data)
    require_columns(frame, [x, y])
    require_numeric(frame, [y])

    figure, axes = _new_figure(figsize)
    axes.plot(frame[x].astype(str), frame[y], marker="o", color="#4C78A8")
    _label(axes, title or f"{y} over {x}", x, y)
    _rotate_labels(axes, frame[x].astype(str))
    return figure


def histogram(
    data: ChartData,
    column: str,
    *,
    bins: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Show how one numeric column is distributed."""
    frame = _as_frame(data)
    require_columns(frame, [column])
    require_numeric(frame, [column])
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise InvalidOperationError(f"bins must be a positive integer, got {bins!r}.")

    values = frame[column].dropna()
    if values.empty:
        raise InvalidOperationError(f"Column '{column}' has no values to plot.")

    figure, axes = _new_figure(figsize)
    axes.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
    _label(axes, title or f"Distribution of {column}", column, "Frequency")
    return figure


def scatter_plot(
    data: ChartData,
    x: str,
    y: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Plot two numeric columns against each other."""
    frame = _as_frame(data)
    require_columns(frame, [x, y])
    require_numeric(frame, [x, y])

    figure, axes = _new_figure(figsize)
    axes.scatter(frame[x], frame[y], color="#4C78A8")
    _label(axes, title or f"{y} vs {x}", x, y)
    return figure


def box_plot(
    data: ChartData,
    columns: str | Sequence[str],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> Figure:
    """Show the spread of one or more numeric columns."""
    frame = _as_frame(data)
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names:
        raise InvalidOperationError("At least one column is required.")
    require_columns(frame, names)
    require_numeric(frame, names)

    series = [frame[name].dropna() for name in names]
    if any(values.empty for values in series):
        raise InvalidOperationError("Every column needs at least one value to plot.")

    figure, axes = _new_figure(figsize)
    axes.boxplot(series, tick_labels=names)
    _label(axes, title or "Value spread", "", "Value")
    return figure


def correlation_heatmap(
    data: ChartData,
    columns: Sequence[str] | None = None,
    *,
    method: str = "pearson",
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> Figure:
    """Draw the correlation matrix of the numeric columns as a heatmap."""
    frame = _as_frame(data)
    result = correlation(frame, columns=columns, method=method)
    names = result.columns
    matrix = [
        [row[name] if row[name] is not None else float("nan") for name in names]
        for row in result.rows
    ]

    figure, axes = _new_figure(figsize)
    image = axes.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes.set_xticks(range(len(names)), labels=names, rotation=45, ha="right")
    axes.set_yticks(range(len(names)), labels=names)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axes.text(
                column_index, row_index, f"{value:.2f}", ha="center", va="center"
            )
    axes.set_title(title or f"{method.title()} correlation")
    figure.colorbar(image, ax=axes)
    return figure


def figure_to_png_bytes(figure: Figure, *, dpi: int = 100) -> bytes:
    """Render a Figure to PNG bytes, for an API response or a UI."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()


def _as_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, AnalysisResult):
        frame = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        raise InvalidOperationError(
            f"Expected a DataFrame or AnalysisResult, got {type(data).__name__}."
        )
    if frame.empty:
        raise InvalidOperationError("There is no data to chart.")
    return frame


def _new_figure(figsize: tuple[float, float]) -> tuple[Figure, Any]:
    figure = Figure(figsize=figsize, layout="constrained")
    return figure, figure.add_subplot()


def _label(axes: Any, title: str, x_label: str, y_label: str) -> None:
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)


def _rotate_labels(axes: Any, labels: pd.Series) -> None:
    """Tilt category labels once they stop fitting side by side."""
    if len(labels) > 6 or labels.str.len().max() > 8:
        axes.tick_params(axis="x", labelrotation=45)
