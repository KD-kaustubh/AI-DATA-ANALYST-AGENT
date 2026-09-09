"""Tests for tool definitions, request parsing and the dispatcher."""

import json

import pandas as pd
import pytest

from analyst import (
    TOOLS,
    AnalysisResult,
    ColumnNotFoundError,
    InvalidOperationError,
    InvalidToolArgumentsError,
    LLMResponseError,
    ToolRequest,
    UnknownToolError,
    describe_tools,
    get_tool,
    run_tool,
    tool_names,
)

EXPECTED_TOOLS = [
    "filter_rows",
    "sort_rows",
    "group_aggregate",
    "describe_numeric",
    "value_counts",
    "correlation",
    "group_by_period",
]


def call(tool: str, **arguments) -> ToolRequest:
    return ToolRequest(action="call_tool", tool=tool, arguments=arguments)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_every_phase_two_operation_is_exposed():
    assert tool_names() == EXPECTED_TOOLS


def test_each_tool_describes_itself():
    for described in describe_tools():
        assert described["purpose"].strip()
        assert "function" not in described
        for argument in described["arguments"]:
            assert argument["description"].strip()
            assert argument["type"]


def test_tool_descriptions_are_json_serialisable():
    assert json.loads(json.dumps(describe_tools()))


def test_tools_point_at_the_real_analysis_functions():
    from analyst import analysis

    for name, spec in TOOLS.items():
        assert spec.function is getattr(analysis, name)


def test_restrictions_are_published_to_the_model():
    period = next(
        argument
        for argument in get_tool("group_by_period").to_dict()["arguments"]
        if argument["name"] == "period"
    )
    assert period["allowed_values"] == ["year", "month"]
    assert period["required"] is True


def test_unknown_tool_is_rejected():
    with pytest.raises(UnknownToolError, match="drop_table"):
        get_tool("drop_table")


# --------------------------------------------------------------------------
# parsing what the model sent back
# --------------------------------------------------------------------------


def test_parses_a_tool_call():
    request = ToolRequest.from_text(
        '{"action": "call_tool", "tool": "value_counts", '
        '"arguments": {"column": "region"}}'
    )

    assert request.action == "call_tool"
    assert request.tool == "value_counts"
    assert request.arguments == {"column": "region"}


def test_parses_a_plain_answer():
    request = ToolRequest.from_text(
        '{"action": "answer", "message": "The dataset has no price column."}'
    )

    assert request.action == "answer"
    assert request.tool is None
    assert "price" in request.message


def test_parses_json_wrapped_in_a_code_fence():
    request = ToolRequest.from_text(
        '```json\n{"action": "call_tool", "tool": "describe_numeric", '
        '"arguments": {}}\n```'
    )

    assert request.tool == "describe_numeric"


def test_parses_json_surrounded_by_chatter():
    request = ToolRequest.from_text(
        'Sure! Here you go:\n{"action": "call_tool", "tool": "correlation", '
        '"arguments": {}}\nHope that helps.'
    )

    assert request.tool == "correlation"


def test_missing_arguments_default_to_empty():
    assert ToolRequest.from_text(
        '{"action": "call_tool", "tool": "describe_numeric"}'
    ).arguments == {}


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "not json at all",
        "{oops",
        '["a", "list"]',
        '{"action": "delete_everything"}',
        '{"tool": "value_counts"}',
        '{"action": "call_tool"}',
        '{"action": "call_tool", "tool": ""}',
        '{"action": "call_tool", "tool": "value_counts", "arguments": "region"}',
        '{"action": "answer"}',
        '{"action": "answer", "message": "  "}',
    ],
)
def test_malformed_replies_are_rejected(reply):
    with pytest.raises(LLMResponseError):
        ToolRequest.from_text(reply)


def test_a_request_round_trips_through_json():
    request = call("value_counts", column="region")

    assert json.loads(json.dumps(request.to_dict()))["tool"] == "value_counts"


# --------------------------------------------------------------------------
# dispatching
# --------------------------------------------------------------------------


def test_dispatches_to_the_real_operation(analysis_frame: pd.DataFrame):
    result = run_tool(
        analysis_frame, call("group_aggregate", group_by="region", aggregations={"revenue": "sum"})
    )

    assert isinstance(result, AnalysisResult)
    assert result.operation == "group_aggregate"
    totals = {row["region"]: row["revenue_sum"] for row in result.rows}
    assert totals == {"North": 210.0, "South": 120.0, "West": 120.0}


def test_dispatches_a_filter(analysis_frame: pd.DataFrame):
    result = run_tool(
        analysis_frame,
        call(
            "filter_rows",
            conditions=[{"column": "region", "operator": "==", "value": "North"}],
        ),
    )

    assert result.row_count == 3


def test_dispatches_a_period_grouping(analysis_frame: pd.DataFrame):
    result = run_tool(
        analysis_frame, call("group_by_period", datetime_column="sold_at", period="year")
    )

    assert {row["period"] for row in result.rows} == {"2024", "2025"}


def test_unknown_tool_never_runs(analysis_frame: pd.DataFrame):
    with pytest.raises(UnknownToolError):
        run_tool(analysis_frame, call("execute_python", code="import os"))


def test_unknown_argument_is_rejected(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidToolArgumentsError, match="Unknown argument"):
        run_tool(analysis_frame, call("value_counts", column="region", sneaky=True))


def test_missing_required_argument_is_rejected(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidToolArgumentsError, match="Missing required"):
        run_tool(analysis_frame, call("value_counts"))


def test_wrong_argument_type_is_rejected(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidToolArgumentsError, match="must be string"):
        run_tool(analysis_frame, call("value_counts", column=["region"]))


def test_disallowed_value_is_rejected(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidToolArgumentsError, match="must be one of"):
        run_tool(
            analysis_frame,
            call("group_by_period", datetime_column="sold_at", period="fortnight"),
        )


def test_boolean_is_not_accepted_as_a_number(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidToolArgumentsError, match="must be integer"):
        run_tool(analysis_frame, call("value_counts", column="region", limit=True))


def test_analysis_errors_travel_as_themselves(analysis_frame: pd.DataFrame):
    with pytest.raises(ColumnNotFoundError):
        run_tool(analysis_frame, call("value_counts", column="nope"))

    with pytest.raises(InvalidOperationError):
        run_tool(
            analysis_frame,
            call("group_aggregate", group_by="region", aggregations={"revenue": "hack"}),
        )


def test_an_answer_request_cannot_be_dispatched(analysis_frame: pd.DataFrame):
    request = ToolRequest(action="answer", message="no tool needed")

    with pytest.raises(InvalidToolArgumentsError):
        run_tool(analysis_frame, request)
