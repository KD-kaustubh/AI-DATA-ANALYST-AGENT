"""Shared fixtures.

The sample frame is deliberately small and deterministic, and carries every
trait the profiler needs to report: missing values, a duplicated row, numeric
and categorical columns, and dates stored as text.
"""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Six rows where the last one duplicates the fifth."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 5],
            "city": ["Delhi", "Mumbai", "Delhi", "Delhi", None, None],
            "signup_date": [
                "2024-01-05",
                "2024-02-11",
                "2024-03-02",
                "2024-04-19",
                "2024-05-30",
                "2024-05-30",
            ],
            "revenue": [100.0, 250.5, None, 75.25, 310.0, 310.0],
            "active": [True, False, True, True, True, True],
        }
    )


@pytest.fixture
def csv_file(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    path = tmp_path / "sample.csv"
    sample_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def xlsx_file(tmp_path: Path, sample_frame: pd.DataFrame) -> Path:
    path = tmp_path / "sample.xlsx"
    sample_frame.to_excel(path, index=False)
    return path
