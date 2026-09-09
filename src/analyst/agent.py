"""A bounded agent loop over the registered analysis tools.

The model chooses one action at a time and is shown the verified result of
each step. It never touches the data directly: every action goes through the
tool dispatcher, which only runs registered functions with declared
arguments. The loop always terminates, because each step consumes one of a
fixed number of iterations.

Failures inside a step are handed back to the model rather than raised, so it
can correct a bad argument or explain the problem, still within the step
budget. Failures the model cannot fix, such as the provider being down, are
raised.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from analyst.context import DatasetContext, build_context
from analyst.errors import (
    AnalysisError,
    InvalidOperationError,
    LLMError,
    LLMProviderError,
    LLMResponseError,
    ToolError,
)
from analyst.grounding import check_grounding
from analyst.llm import LLMClient
from analyst.profiling import DatasetProfile, profile_dataset
from analyst.prompts import (
    AGENT_RULES,
    ANSWER_RULES,
    build_agent_prompt,
    build_answer_prompt,
    trim_result,
)
from analyst.tools import CALL_TOOL, CLARIFY, ToolRequest, describe_tools, run_tool

# Tool calls allowed for one question. Enough for a two or three step
# analysis plus a correction, low enough to bound cost and latency.
DEFAULT_MAX_STEPS = 5

ANSWERED = "answer"
CLARIFICATION = "clarification"
INCOMPLETE = "incomplete"

INCOMPLETE_MESSAGE = (
    "I could not finish this analysis within the step limit. "
    "Try asking for one specific figure at a time."
)


@dataclass(frozen=True)
class EvidenceStep:
    """One attempt: what was asked for, and what came back."""

    number: int
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.result is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    """The reply, plus the verified evidence it was written from."""

    question: str
    text: str
    kind: str = ANSWERED
    steps: list[EvidenceStep] = field(default_factory=list)
    # The last successful step, repeated here so a caller with one tool call
    # can read it without walking the list.
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None

    @property
    def results(self) -> list[dict[str, Any]]:
        """Every verified result, oldest first."""
        return [step.result for step in self.steps if step.result is not None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_question(
    frame: pd.DataFrame,
    question: str,
    client: LLMClient,
    *,
    profile: DatasetProfile | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    history: list[dict[str, Any]] | None = None,
) -> Answer:
    """Answer `question` about `frame`, running up to `max_steps` tools.

    `profile` avoids re-profiling a dataset already profiled. `history` is
    the bounded conversation context supplied by a Conversation.
    """
    if not question or not question.strip():
        raise InvalidOperationError("A question is required.")
    if frame.empty or frame.columns.empty:
        raise InvalidOperationError("There is no data to analyse.")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise InvalidOperationError(
            f"max_steps must be a positive integer, got {max_steps!r}."
        )

    question = question.strip()
    context = build_context(profile or profile_dataset(frame))
    steps: list[EvidenceStep] = []

    for number in range(1, max_steps + 1):
        request = _next_action(question, context, steps, history, client, number)
        if request is None:
            continue  # The reply was malformed; the model was told so.

        if request.action == CLARIFY:
            return Answer(
                question=question,
                text=request.message or "",
                kind=CLARIFICATION,
                steps=steps,
            )

        if request.action != CALL_TOOL:
            if not any(step.succeeded for step in steps):
                # Nothing ran, so there is no evidence to ground: the
                # model's own message is the answer.
                return Answer(
                    question=question, text=request.message or "", steps=steps
                )
            break

        steps.append(_run_step(frame, request, number))
    else:
        return Answer(
            question=question, text=INCOMPLETE_MESSAGE, kind=INCOMPLETE, steps=steps
        )

    return _final_answer(question, context, steps, history, client)


def _next_action(
    question: str,
    context: DatasetContext,
    steps: list[EvidenceStep],
    history: list[dict[str, Any]] | None,
    client: LLMClient,
    number: int,
) -> ToolRequest | None:
    """Ask for the next action, recording a malformed reply as a failed step."""
    reply = _generate(
        client,
        build_agent_prompt(
            question,
            context,
            describe_tools(),
            [step.to_dict() for step in steps],
            history,
        ),
        system=AGENT_RULES,
        json_output=True,
    )
    try:
        return ToolRequest.from_text(reply)
    except LLMResponseError as exc:
        steps.append(
            EvidenceStep(number=number, tool="(unreadable reply)", error=str(exc))
        )
        return None


def _run_step(frame: pd.DataFrame, request: ToolRequest, number: int) -> EvidenceStep:
    """Dispatch one tool call, turning a refusal into feedback."""
    try:
        result = run_tool(frame, request)
    except (ToolError, AnalysisError) as exc:
        return EvidenceStep(
            number=number,
            tool=request.tool or "(none)",
            arguments=request.arguments,
            error=str(exc),
        )
    return EvidenceStep(
        number=number,
        tool=request.tool or "(none)",
        arguments=request.arguments,
        result=trim_result(result.to_dict()),
    )


def _final_answer(
    question: str,
    context: DatasetContext,
    steps: list[EvidenceStep],
    history: list[dict[str, Any]] | None,
    client: LLMClient,
) -> Answer:
    """Write the grounded answer from the verified evidence."""
    evidence = [step.result for step in steps if step.result is not None]
    text = _generate(
        client,
        build_answer_prompt(question, context, evidence, history),
        system=ANSWER_RULES,
    ).strip()

    last = next(step for step in reversed(steps) if step.succeeded)
    return Answer(
        question=question,
        text=text,
        steps=steps,
        tool=last.tool,
        arguments=last.arguments,
        evidence=last.result,
        grounding=check_grounding(text, evidence).to_dict(),
    )


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
