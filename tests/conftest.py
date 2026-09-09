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


@pytest.fixture
def analysis_frame() -> pd.DataFrame:
    """Six sales rows. Revenue is exactly units * 10, so correlation is 1.0."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "West", "South", "North"],
            "product": ["A", "B", "A", "C", "B", "B"],
            "units": [10, 5, 8, 12, 7, 3],
            "revenue": [100.0, 50.0, 80.0, 120.0, 70.0, 30.0],
            "sold_at": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-01-20",
                    "2024-02-10",
                    "2024-02-28",
                    "2024-03-05",
                    "2025-01-02",
                ]
            ),
        }
    )


class FakeLLM:
    """A scripted stand-in for a real model.

    Returns the queued replies in order and records every call, so tests can
    check what the model was shown without touching a network or an API key.
    """

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate(
        self, prompt: str, *, system: str | None = None, json_output: bool = False
    ) -> str:
        self.calls.append(
            {"prompt": prompt, "system": system, "json_output": json_output}
        )
        if not self.replies:
            raise AssertionError("The model was called more times than expected.")
        return self.replies.pop(0)


class BrokenLLM:
    """A model that always fails, standing in for a provider outage."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("connection reset")

    def generate(self, prompt: str, *, system=None, json_output: bool = False) -> str:
        raise self.error


@pytest.fixture
def fake_llm():
    return FakeLLM
