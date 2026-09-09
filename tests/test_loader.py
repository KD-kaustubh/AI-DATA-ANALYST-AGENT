"""Tests for dataset loading and file validation."""

from pathlib import Path

import pandas as pd
import pytest

from analyst import (
    DatasetError,
    DatasetNotFoundError,
    DatasetReadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
    load_dataset,
)


def test_loads_csv(csv_file: Path):
    frame = load_dataset(csv_file)

    assert frame.shape == (6, 5)
    assert list(frame.columns) == ["id", "city", "signup_date", "revenue", "active"]


def test_loads_xlsx(xlsx_file: Path):
    frame = load_dataset(xlsx_file)

    assert frame.shape == (6, 5)
    assert list(frame.columns) == ["id", "city", "signup_date", "revenue", "active"]


def test_csv_and_xlsx_agree_on_values(csv_file: Path, xlsx_file: Path):
    from_csv = load_dataset(csv_file)
    from_xlsx = load_dataset(xlsx_file)

    pd.testing.assert_frame_equal(from_csv, from_xlsx, check_dtype=False)
    assert from_csv["id"].tolist() == [1, 2, 3, 4, 5, 5]


def test_accepts_a_path_given_as_a_string(csv_file: Path):
    assert not load_dataset(str(csv_file)).empty


def test_extension_check_ignores_case(tmp_path: Path, sample_frame: pd.DataFrame):
    path = tmp_path / "upper.CSV"
    sample_frame.to_csv(path, index=False)

    assert load_dataset(path).shape == (6, 5)


def test_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_directory_is_not_a_dataset(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        load_dataset(tmp_path)


def test_unsupported_extension_is_rejected(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("id,value\n1,2\n", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_dataset(path)


def test_zero_byte_file_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDatasetError):
        load_dataset(path)


def test_header_only_csv_is_rejected(tmp_path: Path):
    path = tmp_path / "headers.csv"
    path.write_text("id,city\n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="No data rows"):
        load_dataset(path)


def test_corrupted_xlsx_is_reported(tmp_path: Path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"this is not a real spreadsheet")

    with pytest.raises(DatasetReadError):
        load_dataset(path)


def test_every_failure_shares_one_base_class(tmp_path: Path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.csv")
