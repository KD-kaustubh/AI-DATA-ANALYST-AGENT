"""Request and response models for the HTTP API.

These are the only shapes that leave the process. DataFrames, profiles,
prompts and clients stay behind them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from analyst.agent import Answer
from analyst.api.store import StoredDataset, StoredSession

MAX_QUESTION_LENGTH = 1000


class HealthResponse(BaseModel):
    """Liveness check.

    `provider` and `model` reflect the active configuration and are None
    when no provider is configured. Never a key or other credential.
    """

    status: str
    version: str
    provider: str | None = None
    model: str | None = None


class ColumnInfo(BaseModel):
    """One column, as described by the dataset profile."""

    name: str
    dtype: str
    kind: str
    missing_count: int
    unique_count: int


class DatasetSummary(BaseModel):
    """What a caller learns about an uploaded dataset. Never its rows."""

    dataset_id: str
    filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    warnings: list[str]
    created_at: datetime


class SessionRequest(BaseModel):
    """Open a conversation over a dataset."""

    dataset_id: str = Field(min_length=1)


class SessionResponse(BaseModel):
    """An open conversation."""

    session_id: str
    dataset_id: str
    created_at: datetime
    turn_count: int


class QuestionRequest(BaseModel):
    """A natural-language question."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class AnalyzeRequest(QuestionRequest):
    """A one-off question against a dataset, without a conversation."""

    dataset_id: str = Field(min_length=1)


class StepInfo(BaseModel):
    """One tool call the agent made, without its rows."""

    number: int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    row_count: int | None = None
    error: str | None = None


class GroundingInfo(BaseModel):
    """Which figures in the answer were traced back to the evidence."""

    is_grounded: bool
    checked: list[str]
    unsupported: list[str]


class AnswerResponse(BaseModel):
    """The analyst's reply and how it was reached."""

    question: str
    answer: str
    kind: str
    dataset_id: str
    session_id: str | None = None
    tools_used: list[str]
    steps: list[StepInfo]
    result: dict[str, Any] | None = None
    grounding: GroundingInfo | None = None


class ErrorResponse(BaseModel):
    """Every failure the API reports."""

    detail: str
    code: str


def dataset_summary(stored: StoredDataset) -> DatasetSummary:
    """Describe a stored dataset from its profile."""
    context = stored.context
    return DatasetSummary(
        dataset_id=stored.dataset_id,
        filename=stored.filename,
        row_count=context.row_count,
        column_count=context.column_count,
        columns=[
            ColumnInfo(
                name=column.name,
                dtype=column.dtype,
                kind=column.kind,
                missing_count=column.missing_count,
                unique_count=column.unique_count,
            )
            for column in context.columns
        ],
        numeric_columns=context.numeric_columns,
        categorical_columns=context.categorical_columns,
        datetime_columns=context.datetime_columns,
        warnings=context.warnings,
        created_at=stored.created_at,
    )


def session_response(session: StoredSession) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        dataset_id=session.dataset_id,
        created_at=session.created_at,
        turn_count=len(session.conversation.turns),
    )


def answer_response(
    answer: Answer, dataset_id: str, session_id: str | None = None
) -> AnswerResponse:
    """Flatten an agent Answer into the response shape.

    Only the last verified result carries rows; the earlier steps are
    reported by name, arguments and size.
    """
    return AnswerResponse(
        question=answer.question,
        answer=answer.text,
        kind=answer.kind,
        dataset_id=dataset_id,
        session_id=session_id,
        tools_used=[step.tool for step in answer.steps if step.succeeded],
        steps=[
            StepInfo(
                number=step.number,
                tool=step.tool,
                arguments=step.arguments,
                row_count=(step.result or {}).get("row_count"),
                error=step.error,
            )
            for step in answer.steps
        ],
        result=answer.evidence,
        grounding=(
            GroundingInfo(**answer.grounding) if answer.grounding else None
        ),
    )
