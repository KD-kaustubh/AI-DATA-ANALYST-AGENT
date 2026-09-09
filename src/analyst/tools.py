"""The tools a model is allowed to ask for, and the dispatcher that runs them.

Every tool is a thin description of an existing analysis function. No
analytical logic lives here, and nothing is looked up dynamically: a request
either names a tool in the registry below or it is rejected. There is no
eval, no exec and no import driven by model output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

from analyst.analysis import (
    AGGREGATIONS,
    CORRELATION_METHODS,
    OPERATORS,
    PERIODS,
    AnalysisResult,
    correlation,
    describe_numeric,
    filter_rows,
    group_aggregate,
    group_by_period,
    sort_rows,
    value_counts,
)
from analyst.errors import InvalidToolArgumentsError, LLMResponseError, UnknownToolError

CALL_TOOL = "call_tool"
ANSWER = "answer"
CLARIFY = "clarification"
ACTIONS = (CALL_TOOL, ANSWER, CLARIFY)
# Actions that carry a message to the user instead of running anything.
MESSAGE_ACTIONS = (ANSWER, CLARIFY)

# JSON type names mapped to what they may arrive as.
_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass(frozen=True)
class ToolArgument:
    """One argument a tool accepts."""

    name: str
    types: tuple[str, ...]
    description: str
    required: bool = False
    allowed_values: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        described: dict[str, Any] = {
            "name": self.name,
            "type": " or ".join(self.types),
            "required": self.required,
            "description": self.description,
        }
        if self.allowed_values:
            described["allowed_values"] = list(self.allowed_values)
        return described


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: what it does and which function performs it."""

    name: str
    purpose: str
    arguments: tuple[ToolArgument, ...]
    function: Callable[..., AnalysisResult]

    def to_dict(self) -> dict[str, Any]:
        """Describe the tool for a prompt. The function itself is left out."""
        return {
            "name": self.name,
            "purpose": self.purpose,
            "arguments": [argument.to_dict() for argument in self.arguments],
        }


@dataclass(frozen=True)
class ToolRequest:
    """What the model decided to do.

    `action` is one of:
    - "call_tool", with a tool name and arguments;
    - "answer", carrying a message when the evidence is enough or the
      dataset cannot answer the question;
    - "clarification", carrying a question to put back to the user when the
      request is ambiguous.
    """

    action: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_text(cls, text: str) -> "ToolRequest":
        """Parse a model reply into a request, rejecting anything malformed."""
        payload = _parse_json_object(text)

        action = payload.get("action")
        if action not in ACTIONS:
            raise LLMResponseError(
                f"Expected action to be one of {', '.join(ACTIONS)}, got {action!r}."
            )

        if action in MESSAGE_ACTIONS:
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip():
                raise LLMResponseError(f"A '{action}' action needs a message.")
            return cls(action=action, message=message.strip())

        tool = payload.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise LLMResponseError("A 'call_tool' action needs a tool name.")

        arguments = payload.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise LLMResponseError(
                f"Tool arguments must be an object, got {type(arguments).__name__}."
            )

        return cls(action=CALL_TOOL, tool=tool.strip(), arguments=arguments)


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="filter_rows",
            purpose="Keep only the rows matching one or more conditions.",
            arguments=(
                ToolArgument(
                    name="conditions",
                    types=("array",),
                    description=(
                        "List of objects with 'column', 'operator' and 'value'. "
                        f"Operators: {', '.join(OPERATORS)}. "
                        "'in' and 'not_in' take a list as the value. "
                        "All conditions must hold at once."
                    ),
                    required=True,
                ),
                ToolArgument(
                    name="limit",
                    types=("integer",),
                    description="Maximum rows to return.",
                ),
            ),
            function=filter_rows,
        ),
        ToolSpec(
            name="sort_rows",
            purpose="Order rows by one or more columns, for example to find the largest values.",
            arguments=(
                ToolArgument(
                    name="by",
                    types=("string", "array"),
                    description="Column name, or list of column names.",
                    required=True,
                ),
                ToolArgument(
                    name="ascending",
                    types=("boolean", "array"),
                    description=(
                        "True for smallest first, false for largest first. "
                        "Use a list to give one direction per column."
                    ),
                ),
                ToolArgument(
                    name="limit",
                    types=("integer",),
                    description="Maximum rows to return, e.g. 5 for a top-five question.",
                ),
            ),
            function=sort_rows,
        ),
        ToolSpec(
            name="group_aggregate",
            purpose="Group rows by one or more columns and aggregate other columns.",
            arguments=(
                ToolArgument(
                    name="group_by",
                    types=("string", "array"),
                    description="Column name, or list of column names, to group by.",
                    required=True,
                ),
                ToolArgument(
                    name="aggregations",
                    types=("object",),
                    description=(
                        "Object mapping a column to one aggregation or a list of them. "
                        f"Allowed: {', '.join(AGGREGATIONS)}. "
                        "sum, mean and median need a numeric column."
                    ),
                    required=True,
                ),
            ),
            function=group_aggregate,
        ),
        ToolSpec(
            name="describe_numeric",
            purpose="Summary statistics (count, mean, median, std, min, max) for numeric columns.",
            arguments=(
                ToolArgument(
                    name="columns",
                    types=("string", "array"),
                    description="Column or columns to describe. Omit for every numeric column.",
                ),
            ),
            function=describe_numeric,
        ),
        ToolSpec(
            name="value_counts",
            purpose="Count how often each value appears in one column.",
            arguments=(
                ToolArgument(
                    name="column",
                    types=("string",),
                    description="The column to count values in.",
                    required=True,
                ),
                ToolArgument(
                    name="limit",
                    types=("integer",),
                    description="Maximum distinct values to return.",
                ),
            ),
            function=value_counts,
        ),
        ToolSpec(
            name="correlation",
            purpose="Correlation between numeric columns, as a matrix.",
            arguments=(
                ToolArgument(
                    name="columns",
                    types=("array",),
                    description=(
                        "Numeric columns to correlate, at least two. "
                        "Omit for every numeric column."
                    ),
                ),
                ToolArgument(
                    name="method",
                    types=("string",),
                    description="Correlation method.",
                    allowed_values=CORRELATION_METHODS,
                ),
            ),
            function=correlation,
        ),
        ToolSpec(
            name="group_by_period",
            purpose="Group rows by the year or month of a datetime column.",
            arguments=(
                ToolArgument(
                    name="datetime_column",
                    types=("string",),
                    description="The column holding dates.",
                    required=True,
                ),
                ToolArgument(
                    name="period",
                    types=("string",),
                    description="Size of each bucket.",
                    required=True,
                    allowed_values=PERIODS,
                ),
                ToolArgument(
                    name="aggregations",
                    types=("object",),
                    description=(
                        "Optional, same shape as in group_aggregate. "
                        "Omit to count rows per period."
                    ),
                ),
            ),
            function=group_by_period,
        ),
    )
}


def tool_names() -> list[str]:
    """Every registered tool name."""
    return list(TOOLS)


def describe_tools() -> list[dict[str, Any]]:
    """Describe the whole registry for a prompt."""
    return [spec.to_dict() for spec in TOOLS.values()]


def get_tool(name: str) -> ToolSpec:
    """Look up a tool, refusing anything not registered."""
    spec = TOOLS.get(name)
    if spec is None:
        raise UnknownToolError(
            f"Unknown tool '{name}'. Available: {', '.join(tool_names())}"
        )
    return spec


def validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Check the arguments against the tool's declared signature."""
    if not isinstance(arguments, Mapping):
        raise InvalidToolArgumentsError(
            f"Arguments for '{spec.name}' must be an object."
        )

    declared = {argument.name: argument for argument in spec.arguments}
    unknown = [name for name in arguments if name not in declared]
    if unknown:
        raise InvalidToolArgumentsError(
            f"Unknown argument(s) for '{spec.name}': {', '.join(sorted(unknown))}. "
            f"Accepted: {', '.join(declared)}"
        )

    missing = [
        argument.name
        for argument in spec.arguments
        if argument.required and arguments.get(argument.name) is None
    ]
    if missing:
        raise InvalidToolArgumentsError(
            f"Missing required argument(s) for '{spec.name}': {', '.join(missing)}"
        )

    checked: dict[str, Any] = {}
    for name, value in arguments.items():
        if value is None:
            continue
        _check_type(spec.name, declared[name], value)
        checked[name] = value
    return checked


def run_tool(frame: pd.DataFrame, request: ToolRequest) -> AnalysisResult:
    """Run a validated request against the dataset.

    Only registered tools run, and only with arguments they declare. Analysis
    errors are left to travel as themselves, since they already say what went
    wrong with the data.
    """
    if request.action != CALL_TOOL:
        raise InvalidToolArgumentsError(
            f"Only a '{CALL_TOOL}' request can be run, got '{request.action}'."
        )

    spec = get_tool(request.tool or "")
    arguments = validate_arguments(spec, request.arguments)
    try:
        return spec.function(frame, **arguments)
    except TypeError as exc:
        raise InvalidToolArgumentsError(
            f"Tool '{spec.name}' rejected the given arguments: {exc}"
        ) from exc


def _check_type(tool: str, argument: ToolArgument, value: Any) -> None:
    accepted = tuple(
        allowed for name in argument.types for allowed in _TYPES.get(name, ())
    )
    # bool is a subclass of int, so numeric arguments must exclude it.
    if isinstance(value, bool) and "boolean" not in argument.types:
        accepted = ()
    if not accepted or not isinstance(value, accepted):
        raise InvalidToolArgumentsError(
            f"Argument '{argument.name}' of '{tool}' must be "
            f"{' or '.join(argument.types)}, got {type(value).__name__}."
        )
    if argument.allowed_values and value not in argument.allowed_values:
        raise InvalidToolArgumentsError(
            f"Argument '{argument.name}' of '{tool}' must be one of "
            f"{', '.join(argument.allowed_values)}, got {value!r}."
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    """Read a JSON object out of a model reply, fences and all."""
    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("The model returned an empty response.")

    cleaned = _strip_code_fence(text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = _parse_embedded_object(cleaned)

    if not isinstance(payload, dict):
        raise LLMResponseError(
            f"Expected a JSON object, got {type(payload).__name__}."
        )
    return payload


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    without_open = text.split("\n", 1)[-1]
    closing = without_open.rfind("```")
    return (without_open[:closing] if closing != -1 else without_open).strip()


def _parse_embedded_object(text: str) -> Any:
    """Last resort: take the outermost {...} from a reply wrapped in prose."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMResponseError("The model did not return JSON.")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"The model returned invalid JSON: {exc.msg}") from exc
