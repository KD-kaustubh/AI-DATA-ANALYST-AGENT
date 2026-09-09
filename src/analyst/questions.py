"""Answer one question about one dataset.

The flow is fixed and runs exactly once: the model picks a tool, pandas runs
it, and the model puts the verified result into words. There is no loop and
no planning; the model never sees the data except through a tool result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from analyst.context import DatasetContext, build_context
from analyst.errors import InvalidOperationError, LLMError, LLMProviderError
from analyst.llm import LLMClient
from analyst.profiling import DatasetProfile, profile_dataset
from analyst.prompts import (
    ANSWER_RULES,
    TOOL_SELECTION_RULES,
    build_answer_prompt,
    build_tool_prompt,
)
from analyst.tools import CALL_TOOL, ToolRequest, describe_tools, run_tool


@dataclass(frozen=True)
class Answer:
    """The reply, plus the verified result it was written from."""

    question: str
    text: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_question(
    frame: pd.DataFrame,
    question: str,
    client: LLMClient,
    *,
    profile: DatasetProfile | None = None,
) -> Answer:
    """Answer `question` about `frame` using one tool call.

    `profile` may be supplied to avoid re-profiling a dataset that has
    already been profiled.
    """
    if not question or not question.strip():
        raise InvalidOperationError("A question is required.")
    if frame.empty or frame.columns.empty:
        raise InvalidOperationError("There is no data to analyse.")

    context = build_context(profile or profile_dataset(frame))
    request = _select_tool(question, context, client)

    if request.action != CALL_TOOL:
        # The model decided no tool fits; its message is the answer.
        return Answer(question=question, text=request.message or "")

    result = run_tool(frame, request)
    evidence = result.to_dict()
    text = _write_answer(question, evidence, client)

    return Answer(
        question=question,
        text=text,
        tool=request.tool,
        arguments=request.arguments,
        evidence=evidence,
    )


def _select_tool(
    question: str, context: DatasetContext, client: LLMClient
) -> ToolRequest:
    reply = _generate(
        client,
        build_tool_prompt(question, context, describe_tools()),
        system=TOOL_SELECTION_RULES,
        json_output=True,
    )
    return ToolRequest.from_text(reply)


def _write_answer(question: str, evidence: dict[str, Any], client: LLMClient) -> str:
    reply = _generate(
        client,
        build_answer_prompt(question, evidence),
        system=ANSWER_RULES,
    )
    return reply.strip()


def _generate(
    client: LLMClient, prompt: str, *, system: str, json_output: bool = False
) -> str:
    """Call the client, keeping provider details out of the error."""
    try:
        return client.generate(prompt, system=system, json_output=json_output)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMProviderError(
            f"The language model request failed ({type(exc).__name__})."
        ) from exc
