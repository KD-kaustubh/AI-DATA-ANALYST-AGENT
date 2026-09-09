"""Tests for the analysis operations.

Expected numbers are worked out by hand or with the standard library, so the
assertions do not simply echo whatever pandas produced.
"""

import json
import statistics

import pandas as pd
import pytest

from analyst import (
    AnalysisError,
    ColumnNotFoundError,
    Condition,
    InvalidOperationError,
    correlation,
    describe_numeric,
    filter_rows,
    group_aggregate,
    group_by_period,
    sort_rows,
    value_counts,
)

UNITS = [10, 5, 8, 12, 7, 3]


def values(result, key):
    """Pull one field out of every result row."""
    return [row[key] for row in result.rows]


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def test_filters_on_equality(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("region", "==", "North")])

    assert result.row_count == 3
    assert set(values(result, "region")) == {"North"}
    assert result.metadata["source_rows"] == 6
    assert result.metadata["matched_rows"] == 3


def test_filters_on_numeric_comparison(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("revenue", ">", 70.0)])

    assert sorted(values(result, "revenue")) == [80.0, 100.0, 120.0]


def test_filters_on_inequality(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("units", "!=", 10)])

    assert result.row_count == 5
    assert 10 not in values(result, "units")


def test_filters_on_membership(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("region", "in", ["South", "West"])])

    assert result.row_count == 3
    assert set(values(result, "region")) == {"South", "West"}


def test_filters_on_non_membership(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("region", "not_in", ["North"])])

    assert set(values(result, "region")) == {"South", "West"}


def test_conditions_combine_with_and(analysis_frame: pd.DataFrame):
    result = filter_rows(
        analysis_frame,
        [Condition("region", "==", "North"), Condition("units", ">=", 8)],
    )

    assert sorted(values(result, "units")) == [8, 10]


def test_conditions_may_be_plain_dicts(analysis_frame: pd.DataFrame):
    result = filter_rows(
        analysis_frame, [{"column": "region", "operator": "==", "value": "West"}]
    )

    assert result.row_count == 1


def test_filter_limit_truncates_and_says_so(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("units", ">", 0)], limit=2)

    assert result.row_count == 2
    assert result.metadata["matched_rows"] == 6
    assert result.metadata["truncated"] is True


def test_filter_reports_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError, match="nope"):
        filter_rows(analysis_frame, [Condition("nope", "==", 1)])


def test_filter_rejects_an_unknown_operator(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Unknown operator"):
        filter_rows(analysis_frame, [Condition("units", "~=", 1)])


def test_filter_rejects_membership_without_a_list(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="needs a list"):
        filter_rows(analysis_frame, [Condition("region", "in", "North")])


def test_filter_rejects_a_malformed_condition(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="missing"):
        filter_rows(analysis_frame, [{"column": "units", "operator": ">"}])


def test_filter_needs_at_least_one_condition(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError):
        filter_rows(analysis_frame, [])


def test_filter_rejects_comparing_text_with_a_number(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Cannot compare"):
        filter_rows(analysis_frame, [Condition("region", ">", 5)])


def test_filter_leaves_the_original_frame_alone(analysis_frame: pd.DataFrame):
    before = analysis_frame.copy(deep=True)

    filter_rows(analysis_frame, [Condition("region", "==", "North")])

    assert analysis_frame.equals(before)


# --------------------------------------------------------------------------
# sorting
# --------------------------------------------------------------------------


def test_sorts_ascending(analysis_frame: pd.DataFrame):
    result = sort_rows(analysis_frame, "revenue")

    assert values(result, "revenue") == [30.0, 50.0, 70.0, 80.0, 100.0, 120.0]


def test_sorts_descending(analysis_frame: pd.DataFrame):
    result = sort_rows(analysis_frame, "revenue", ascending=False)

    assert values(result, "revenue") == [120.0, 100.0, 80.0, 70.0, 50.0, 30.0]


def test_sorts_by_several_columns(analysis_frame: pd.DataFrame):
    result = sort_rows(analysis_frame, ["region", "revenue"], ascending=[True, False])

    assert values(result, "region") == ["North"] * 3 + ["South"] * 2 + ["West"]
    assert values(result, "revenue")[:3] == [100.0, 80.0, 30.0]


def test_sort_limit_returns_the_top_rows(analysis_frame: pd.DataFrame):
    result = sort_rows(analysis_frame, "revenue", ascending=False, limit=2)

    assert values(result, "revenue") == [120.0, 100.0]
    assert result.metadata["truncated"] is True


def test_sort_reports_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        sort_rows(analysis_frame, "nope")


def test_sort_rejects_mismatched_directions(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="direction"):
        sort_rows(analysis_frame, ["region", "revenue"], ascending=[True])


def test_sort_rejects_a_bad_limit(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="positive integer"):
        sort_rows(analysis_frame, "revenue", limit=0)


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def test_groups_and_sums(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"revenue": "sum"})

    totals = dict(zip(values(result, "region"), values(result, "revenue_sum")))
    assert totals == {"North": 210.0, "South": 120.0, "West": 120.0}


def test_groups_and_averages(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"revenue": "mean"})

    averages = dict(zip(values(result, "region"), values(result, "revenue_mean")))
    assert averages["North"] == pytest.approx(70.0)
    assert averages["South"] == pytest.approx(60.0)


def test_groups_and_counts(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"units": "count"})

    counts = dict(zip(values(result, "region"), values(result, "units_count")))
    assert counts == {"North": 3, "South": 2, "West": 1}


def test_groups_by_several_columns(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, ["region", "product"], {"units": "sum"})

    assert result.row_count == 4
    pairs = {
        (region, product): units
        for region, product, units in zip(
            values(result, "region"), values(result, "product"), values(result, "units_sum")
        )
    }
    assert pairs[("North", "A")] == 18
    assert pairs[("North", "B")] == 3


def test_supports_several_aggregations_at_once(analysis_frame: pd.DataFrame):
    result = group_aggregate(
        analysis_frame, "region", {"revenue": ["sum", "max"], "units": "min"}
    )

    row = next(item for item in result.rows if item["region"] == "North")
    assert row["revenue_sum"] == 210.0
    assert row["revenue_max"] == 100.0
    assert row["units_min"] == 3


def test_group_rejects_an_unknown_aggregation(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Unknown aggregation"):
        group_aggregate(analysis_frame, "region", {"revenue": "kurtosis"})


def test_group_rejects_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        group_aggregate(analysis_frame, "nope", {"revenue": "sum"})


def test_group_rejects_summing_text(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Numeric column"):
        group_aggregate(analysis_frame, "region", {"product": "mean"})


# --------------------------------------------------------------------------
# descriptive statistics
# --------------------------------------------------------------------------


def test_describes_a_numeric_column(analysis_frame: pd.DataFrame):
    row = describe_numeric(analysis_frame, "units").rows[0]

    assert row["column"] == "units"
    assert row["count"] == 6
    assert row["mean"] == pytest.approx(statistics.mean(UNITS))
    assert row["median"] == pytest.approx(statistics.median(UNITS))
    assert row["std"] == pytest.approx(statistics.stdev(UNITS))
    assert row["minimum"] == 3.0
    assert row["maximum"] == 12.0


def test_describes_every_numeric_column_by_default(analysis_frame: pd.DataFrame):
    result = describe_numeric(analysis_frame)

    assert values(result, "column") == ["units", "revenue"]


def test_describe_rejects_a_text_column(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Numeric column"):
        describe_numeric(analysis_frame, "region")


def test_describe_rejects_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        describe_numeric(analysis_frame, "nope")


def test_describe_needs_a_numeric_column_somewhere():
    with pytest.raises(InvalidOperationError, match="No numeric columns"):
        describe_numeric(pd.DataFrame({"name": ["a", "b"]}))


# --------------------------------------------------------------------------
# value counts
# --------------------------------------------------------------------------


def test_counts_values(analysis_frame: pd.DataFrame):
    result = value_counts(analysis_frame, "region")

    counts = dict(zip(values(result, "value"), values(result, "count")))
    assert counts == {"North": 3, "South": 2, "West": 1}
    assert result.metadata["distinct_values"] == 3


def test_value_counts_include_percentages(analysis_frame: pd.DataFrame):
    result = value_counts(analysis_frame, "region")

    percentages = dict(zip(values(result, "value"), values(result, "percentage")))
    assert percentages["North"] == 50.0
    assert percentages["South"] == pytest.approx(33.33)


def test_value_counts_skip_missing_values():
    frame = pd.DataFrame({"grade": ["a", "a", None, "b"]})

    result = value_counts(frame, "grade")

    assert dict(zip(values(result, "value"), values(result, "count"))) == {"a": 2, "b": 1}
    assert result.metadata["missing_count"] == 1


def test_value_counts_respect_a_limit(analysis_frame: pd.DataFrame):
    result = value_counts(analysis_frame, "region", limit=1)

    assert result.row_count == 1
    assert result.metadata["truncated"] is True


def test_value_counts_reject_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        value_counts(analysis_frame, "nope")


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------


def test_correlation_of_a_perfect_relationship(analysis_frame: pd.DataFrame):
    result = correlation(analysis_frame, ["units", "revenue"])

    assert result.metadata["correlation"] == pytest.approx(1.0)
    assert result.rows[0]["units"] == pytest.approx(1.0)
    assert result.rows[0]["revenue"] == pytest.approx(1.0)


def test_correlation_of_an_inverse_relationship():
    frame = pd.DataFrame({"up": [1, 2, 3, 4], "down": [4, 3, 2, 1]})

    result = correlation(frame, ["up", "down"])

    assert result.metadata["correlation"] == pytest.approx(-1.0)


def test_correlation_returns_a_full_matrix(analysis_frame: pd.DataFrame):
    result = correlation(analysis_frame)

    assert values(result, "column") == ["units", "revenue"]
    assert result.row_count == 2


def test_correlation_rejects_a_text_column(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Numeric column"):
        correlation(analysis_frame, ["units", "region"])


def test_correlation_needs_two_columns(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="at least two"):
        correlation(analysis_frame, ["units"])


def test_correlation_rejects_an_unknown_method(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Unknown method"):
        correlation(analysis_frame, ["units", "revenue"], method="magic")


# --------------------------------------------------------------------------
# datetime grouping
# --------------------------------------------------------------------------


def test_groups_by_year(analysis_frame: pd.DataFrame):
    result = group_by_period(analysis_frame, "sold_at", "year")

    counts = dict(zip(values(result, "period"), values(result, "row_count")))
    assert counts == {"2024": 5, "2025": 1}


def test_groups_by_month(analysis_frame: pd.DataFrame):
    result = group_by_period(analysis_frame, "sold_at", "month")

    counts = dict(zip(values(result, "period"), values(result, "row_count")))
    assert counts == {"2024-01": 2, "2024-02": 2, "2024-03": 1, "2025-01": 1}


def test_groups_by_period_with_an_aggregation(analysis_frame: pd.DataFrame):
    result = group_by_period(analysis_frame, "sold_at", "year", {"revenue": "sum"})

    totals = dict(zip(values(result, "period"), values(result, "revenue_sum")))
    assert totals == {"2024": 420.0, "2025": 30.0}


def test_groups_dates_stored_as_text_without_changing_the_frame():
    frame = pd.DataFrame({"day": ["2024-01-05", "2024-02-06"], "n": [1, 2]})
    before = frame.copy(deep=True)

    result = group_by_period(frame, "day", "month")

    assert values(result, "period") == ["2024-01", "2024-02"]
    assert frame.equals(before)
    assert frame["day"].dtype == before["day"].dtype


def test_period_grouping_rejects_an_unknown_period(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="Unknown period"):
        group_by_period(analysis_frame, "sold_at", "fortnight")


def test_period_grouping_rejects_a_column_without_dates(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="does not hold dates"):
        group_by_period(analysis_frame, "region", "year")


def test_period_grouping_rejects_an_unknown_column(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        group_by_period(analysis_frame, "nope", "year")


# --------------------------------------------------------------------------
# result shape
# --------------------------------------------------------------------------


def test_results_describe_the_operation(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"revenue": "sum"})

    assert result.operation == "group_aggregate"
    assert result.columns == ["region", "revenue"]
    assert result.parameters["aggregations"] == {"revenue": ["sum"]}


def test_results_are_json_serialisable(analysis_frame: pd.DataFrame):
    for result in (
        filter_rows(analysis_frame, [Condition("region", "==", "North")]),
        sort_rows(analysis_frame, "revenue"),
        group_aggregate(analysis_frame, "region", {"revenue": "sum"}),
        describe_numeric(analysis_frame),
        value_counts(analysis_frame, "region"),
        correlation(analysis_frame),
        group_by_period(analysis_frame, "sold_at", "month"),
    ):
        restored = json.loads(json.dumps(result.to_dict()))
        assert restored["operation"] == result.operation
        assert restored["row_count"] == result.row_count


def test_timestamps_leave_as_iso_strings(analysis_frame: pd.DataFrame):
    result = filter_rows(analysis_frame, [Condition("region", "==", "West")])

    assert result.rows[0]["sold_at"] == "2024-02-28T00:00:00"


def test_results_convert_back_to_a_frame(analysis_frame: pd.DataFrame):
    result = group_aggregate(analysis_frame, "region", {"revenue": "sum"})

    frame = result.to_frame()
    assert list(frame.columns) == ["region", "revenue_sum"]
    assert len(frame) == 3


def test_column_and_operation_errors_share_a_base_class(analysis_frame: pd.DataFrame):
    with pytest.raises(AnalysisError):
        sort_rows(analysis_frame, "nope")
    with pytest.raises(AnalysisError):
        correlation(analysis_frame, ["units"])
