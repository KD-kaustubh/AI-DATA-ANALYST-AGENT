"""Tests for dataset profiling.

Expected statistics are computed with the standard library so the assertions
do not simply restate what pandas returned.
"""

import json
import statistics

import pandas as pd
import pytest

from analyst import profile_dataset

REVENUE = [100.0, 250.5, 75.25, 310.0, 310.0]


def find(items, name):
    """Pick the entry describing `name` out of a profile section."""
    return next(item for item in items if item.name == name)


def test_counts_rows_and_columns(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    assert profile.row_count == 6
    assert profile.column_count == 5


def test_reports_column_names_and_dtypes(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    assert profile.column_names == ["id", "city", "signup_date", "revenue", "active"]
    assert profile.dtypes["id"].startswith("int")
    assert profile.dtypes["revenue"].startswith("float")
    assert profile.dtypes["active"] == "bool"


def test_counts_missing_values(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    city = find(profile.columns, "city")
    assert city.missing_count == 2
    assert city.non_null_count == 4
    assert city.missing_percentage == 33.33

    revenue = find(profile.columns, "revenue")
    assert revenue.missing_count == 1
    assert revenue.missing_percentage == 16.67

    assert find(profile.columns, "id").missing_count == 0


def test_counts_duplicate_rows(sample_frame: pd.DataFrame):
    assert profile_dataset(sample_frame).duplicate_row_count == 1


def test_counts_unique_values_ignoring_missing(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    assert find(profile.columns, "id").unique_count == 5
    assert find(profile.columns, "city").unique_count == 2
    assert find(profile.columns, "signup_date").unique_count == 5


def test_numeric_statistics(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    assert [stats.name for stats in profile.numeric_columns] == ["id", "revenue"]

    revenue = find(profile.numeric_columns, "revenue")
    assert revenue.count == 5
    assert revenue.mean == pytest.approx(statistics.mean(REVENUE))
    assert revenue.std == pytest.approx(statistics.stdev(REVENUE))
    assert revenue.minimum == pytest.approx(75.25)
    assert revenue.maximum == pytest.approx(310.0)
    assert revenue.median == pytest.approx(statistics.median(REVENUE))


def test_booleans_are_categorical_not_numeric(sample_frame: pd.DataFrame):
    profile = profile_dataset(sample_frame)

    assert "active" not in [stats.name for stats in profile.numeric_columns]
    active = find(profile.categorical_columns, "active")
    assert active.most_frequent is True
    assert active.most_frequent_count == 5


def test_categorical_statistics(sample_frame: pd.DataFrame):
    city = find(profile_dataset(sample_frame).categorical_columns, "city")

    assert city.unique_count == 2
    assert city.most_frequent == "Delhi"
    assert city.most_frequent_count == 3


def test_detects_dates_stored_as_text(sample_frame: pd.DataFrame):
    detected = profile_dataset(sample_frame).datetime_columns

    assert [column.name for column in detected] == ["signup_date"]
    assert detected[0].is_datetime_dtype is False
    assert detected[0].parsed_ratio == 1.0


def test_detects_real_datetime_columns():
    frame = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-06-30"])})

    detected = profile_dataset(frame).datetime_columns

    assert len(detected) == 1
    assert detected[0].is_datetime_dtype is True


def test_does_not_treat_plain_numbers_or_words_as_dates():
    frame = pd.DataFrame(
        {
            "year": ["2020", "2021", "2022"],
            "city": ["Delhi", "Mumbai", "Pune"],
            "count": [1, 2, 3],
        }
    )

    assert profile_dataset(frame).datetime_columns == []


def test_profiling_leaves_the_frame_untouched(sample_frame: pd.DataFrame):
    before = sample_frame.copy(deep=True)
    dtypes_before = sample_frame.dtypes.to_dict()

    profile_dataset(sample_frame)

    assert sample_frame.equals(before)
    assert sample_frame.dtypes.to_dict() == dtypes_before


def test_warns_about_duplicates_missing_values_and_suspicious_types(
    sample_frame: pd.DataFrame,
):
    messages = " ".join(profile_dataset(sample_frame).warnings)

    assert "1 duplicate row(s) found." in messages
    assert "'city' has 2 missing value(s)" in messages
    assert "'signup_date' is stored as text but looks like a datetime" in messages


def test_warns_about_numbers_stored_as_text():
    frame = pd.DataFrame({"price": ["10", "20", "30"]})

    messages = profile_dataset(frame).warnings

    assert any("looks numeric" in message for message in messages)


def test_warns_about_a_column_with_no_values():
    frame = pd.DataFrame({"note": [None, None], "id": [1, 2]})

    messages = profile_dataset(frame).warnings

    assert any("entirely missing" in message for message in messages)


def test_handles_a_frame_with_no_rows():
    profile = profile_dataset(pd.DataFrame({"id": [], "city": []}))

    assert profile.row_count == 0
    assert profile.column_count == 2
    assert find(profile.columns, "id").missing_percentage == 0.0


def test_to_dict_is_json_serialisable(sample_frame: pd.DataFrame):
    as_dict = profile_dataset(sample_frame).to_dict()

    assert json.loads(json.dumps(as_dict))["row_count"] == 6
    assert as_dict["columns"][0]["name"] == "id"
