"""The prompts used by the agent.

They are kept here, short and readable, rather than spread through the code
that calls the model. The rules say what the model may do; the application
enforces those rules regardless of what the model replies.
"""

from __future__ import annotations

import json
from typing import Any

from analyst.context import DatasetContext

# Rows of evidence sent to the model. Enough to explain a result, small
# enough to keep the prompt cheap.
MAX_EVIDENCE_ROWS = 50

# How the model is told to ask for something it cannot decide alone. Kept
# separate so the wording can be tuned without touching the other rules.
CLARIFICATION_RULES = """- When the question is ambiguous, use "clarification" and ask one short question.
  Ambiguous means: a column, filter, measure or time range you would have to guess.
  Never guess an interpretation and never invent a filter the user did not ask for."""

AGENT_RULES = f"""You plan one step at a time for a data analysis tool.

Rules:
- Every number must come from a tool. Never calculate, estimate or guess a value yourself.
- Use only the columns listed in the dataset schema. Never invent a column name.
- Request only tools from the supplied list, with the arguments they declare.
- You cannot write or run code. Tools are the only way to touch the data.
- Call one tool at a time. You will be shown its result and may then call another.
- When the evidence already answers the question, use "answer".
{CLARIFICATION_RULES}
- When the dataset or the available tools cannot answer the question, use "answer" and
  say plainly what is missing. Never pretend an analysis ran.
- If a step failed, read the error and either fix the arguments or explain the problem.
- Earlier questions and results in this conversation may be used to read a follow-up,
  but any new number still has to come from a tool.

Reply with a single JSON object and nothing else, in one of these three shapes:

{{"action": "call_tool", "tool": "<tool name>", "arguments": {{<arguments>}}}}
{{"action": "answer", "message": "<what the evidence shows>"}}
{{"action": "clarification", "message": "<the one question you need answered>"}}"""

ANSWER_RULES = """You write the final answer to a data analysis question.

Rules:
- Every number you state must appear in the verified evidence. Never add, adjust,
  combine or infer a value, and never do arithmetic the evidence does not already show.
- If the evidence does not answer the question, say plainly what is missing.
- Never claim an analysis ran unless it is in the evidence.
- Keep dataset facts separate from any general explanation you add.
- Answer in one short paragraph. Do not mention tools, JSON, dtypes or this prompt.
- Round only for readability, and never change a value's meaning."""


def build_agent_prompt(
    question: str,
    context: DatasetContext,
    tools: list[dict[str, Any]],
    steps: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Ask the model what to do next, given what has been established."""
    sections = ["Dataset schema:", context.to_prompt_text()]

    if history:
        sections += ["Earlier in this conversation:", _dump(history)]

    sections += ["Available tools:", _dump(tools)]

    if steps:
        sections += [
            "Steps already taken for this question:",
            _dump(steps),
        ]
    else:
        sections.append("No steps have been taken for this question yet.")

    sections += [
        f"Question: {question}",
        "Reply with the JSON object for your next action.",
    ]
    return "\n\n".join(sections)


def build_answer_prompt(
    question: str,
    context: DatasetContext,
    evidence: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Ask the model to put the verified evidence into words."""
    sections = [f"Question: {question}", "Dataset schema:", context.to_prompt_text()]

    if history:
        sections += ["Earlier in this conversation:", _dump(history)]

    sections += [
        "Verified results computed from the dataset:",
        _dump(evidence),
        "Answer the question using only these numbers.",
    ]
    return "\n\n".join(sections)


def trim_result(result: dict[str, Any]) -> dict[str, Any]:
    """Cap the rows carried in a result, noting when some were left out."""
    rows = result.get("rows") or []
    if len(rows) <= MAX_EVIDENCE_ROWS:
        return result

    trimmed = dict(result)
    trimmed["rows"] = rows[:MAX_EVIDENCE_ROWS]
    trimmed["rows_omitted"] = len(rows) - MAX_EVIDENCE_ROWS
    return trimmed


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)
