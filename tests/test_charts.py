"""Tests for the chart builders.

Figures are built and inspected in memory. Nothing is displayed, and the one
test that writes a file writes it to pytest's temporary directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from analyst import (
    ColumnNotFoundError,
    InvalidOperationError,
    bar_chart,
    box_plot,
    correlation_heatmap,
    figure_to_png_bytes,
    group_aggregate,
    histogram,
    line_chart,
    scatter_plot,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_bar_chart_draws_one_bar_per_row(analysis_frame: pd.DataFrame):
    figure = bar_chart(analysis_frame, "product", "revenue")

    assert isinstance(figure, Figure)
    assert len(figure.axes[0].patches) == 6


def test_line_chart_draws_a_line(analysis_frame: pd.DataFrame):
    figure = line_chart(analysis_frame, "product", "units")

    assert len(figure.axes[0].lines) == 1


def test_histogram_uses_the_requested_bins(analysis_frame: pd.DataFrame):
    figure = histogram(analysis_frame, "revenue", bins=4)

    assert len(figure.axes[0].patches) == 4


def test_scatter_plot_draws_the_points(analysis_frame: pd.DataFrame):
    figure = scatter_plot(analysis_frame, "units", "revenue")

    assert len(figure.axes[0].collections) == 1


def test_box_plot_handles_several_columns(analysis_frame: pd.DataFrame):
    figure = box_plot(analysis_frame, ["units", "revenue"])

    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    assert labels == ["units", "revenue"]


def test_box_plot_accepts_a_single_column(analysis_frame: pd.DataFrame):
    assert isinstance(box_plot(analysis_frame, "units"), Figure)


def test_correlation_heatmap_labels_every_column(analysis_frame: pd.DataFrame):
    figure = correlation_heatmap(analysis_frame)

    labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
    assert labels == ["units", "revenue"]


def test_titles_can_be_overridden(analysis_frame: pd.DataFrame):
    figure = bar_chart(analysis_frame, "region", "revenue", title="Sales by region")

    assert figure.axes[0].get_title() == "Sales by region"


def test_charts_accept_an_analysis_result(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"revenue": "sum"})

    figure = bar_chart(result, "region", "revenue_sum")

    assert len(figure.axes[0].patches) == 3


def test_building_a_chart_opens_no_interactive_window(analysis_frame: pd.DataFrame):
    before = len(plt.get_fignums())

    bar_chart(analysis_frame, "region", "revenue")
    histogram(analysis_frame, "units")

    assert len(plt.get_fignums()) == before


def test_figure_converts_to_png_bytes(analysis_frame: pd.DataFrame):
    data = figure_to_png_bytes(scatter_plot(analysis_frame, "units", "revenue"))

    assert data.startswith(PNG_MAGIC)
    assert len(data) > 1000


def test_figure_can_be_saved_to_a_temporary_path(
    tmp_path: Path, analysis_frame: pd.DataFrame
):
    target = tmp_path / "chart.png"

    bar_chart(analysis_frame, "region", "revenue").savefig(target)

    assert target.exists()
    assert target.read_bytes().startswith(PNG_MAGIC)


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: bar_chart(frame, "nope", "revenue"),
        lambda frame: line_chart(frame, "region", "nope"),
        lambda frame: histogram(frame, "nope"),
        lambda frame: scatter_plot(frame, "units", "nope"),
        lambda frame: box_plot(frame, ["nope"]),
    ],
)
def test_charts_report_an_unknown_column(analysis_frame: pd.DataFrame, build):
    with pytest.raises(ColumnNotFoundError):
        build(analysis_frame)


@pytest.mark.parametrize(
    "build",
    [
        lambda frame: bar_chart(frame, "region", "product"),
        lambda frame: histogram(frame, "region"),
        lambda frame: scatter_plot(frame, "region", "revenue"),
        lambda frame: box_plot(frame, ["region"]),
    ],
)
def test_charts_refuse_non_numeric_values(analysis_frame: pd.DataFrame, build):
    with pytest.raises(InvalidOperationError, match="Numeric column"):
        build(analysis_frame)


def test_charts_refuse_an_empty_dataset():
    with pytest.raises(InvalidOperationError, match="no data"):
        bar_chart(pd.DataFrame({"a": [], "b": []}), "a", "b")


def test_charts_refuse_something_that_is_not_a_dataset():
    with pytest.raises(InvalidOperationError, match="Expected a DataFrame"):
        bar_chart([1, 2, 3], "a", "b")


def test_histogram_rejects_a_bad_bin_count(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="positive integer"):
        histogram(analysis_frame, "units", bins=0)


def test_box_plot_needs_at_least_one_column(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError):
        box_plot(analysis_frame, [])


def test_histogram_rejects_a_numeric_column_that_is_all_missing():
    frame = pd.DataFrame({"value": pd.Series([None, None], dtype="float64")})

    with pytest.raises(InvalidOperationError, match="no values"):
        histogram(frame, "value")


def test_histogram_rejects_an_untyped_all_missing_column():
    frame = pd.DataFrame({"value": [None, None], "other": [1, 2]})

    with pytest.raises(InvalidOperationError, match="Numeric column"):
        histogram(frame, "value")
