"""The two prompts used in the question-answering flow.

They are kept here, short and readable, rather than spread through the code
that calls the model. The rules say what the model may do; the application
enforces those rules regardless of what the model replies.
"""

from __future__ import annotations

import json
from typing import Any

from analyst.context import DatasetContext

# Rows of evidence sent back for the final answer. Enough to explain a
# result, small enough to keep the prompt cheap.
MAX_EVIDENCE_ROWS = 50

TOOL_SELECTION_RULES = """You choose which analysis tool answers a question about one dataset.

Rules:
- Every number must come from a tool. Never calculate, estimate or guess a value yourself.
- Use only the columns listed in the dataset schema. Never invent a column name.
- Request only tools from the supplied list, with the arguments they declare.
- You cannot write or run code. Tools are the only way to touch the data.
- If the dataset cannot answer the question, or no tool fits, use the "answer" action to say so plainly.

Reply with a single JSON object and nothing else, in one of these two shapes:

{"action": "call_tool", "tool": "<tool name>", "arguments": {<arguments>}}
{"action": "answer", "message": "<why no tool was used>"}"""

ANSWER_RULES = """You explain the result of a data analysis in plain language.

Rules:
- Use only the numbers in the supplied result. Never add, adjust or infer a value.
- If the result does not answer the question, say what it does show instead.
- Answer in one short paragraph. Do not repeat the raw result or mention tools, JSON or column dtypes.
- Round only for readability, and never change a value's meaning."""


def build_tool_prompt(
    question: str, context: DatasetContext, tools: list[dict[str, Any]]
) -> str:
    """Ask the model which tool to run."""
    return "\n\n".join(
        [
            "Dataset schema:",
            context.to_prompt_text(),
            "Available tools:",
            json.dumps(tools, indent=2),
            f"Question: {question}",
            "Reply with the JSON object for the tool that answers it.",
        ]
    )


def build_answer_prompt(question: str, result: dict[str, Any]) -> str:
    """Ask the model to put a verified result into words."""
    return "\n\n".join(
        [
            f"Question: {question}",
            "Verified result computed from the dataset:",
            json.dumps(_trim_rows(result), indent=2, default=str),
            "Answer the question using only these numbers.",
        ]
    )


def _trim_rows(result: dict[str, Any]) -> dict[str, Any]:
    """Cap the rows sent to the model, noting when some were left out."""
    rows = result.get("rows") or []
    if len(rows) <= MAX_EVIDENCE_ROWS:
        return result

    trimmed = dict(result)
    trimmed["rows"] = rows[:MAX_EVIDENCE_ROWS]
    trimmed["rows_omitted"] = len(rows) - MAX_EVIDENCE_ROWS
    return trimmed
