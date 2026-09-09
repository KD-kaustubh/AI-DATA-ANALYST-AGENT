"""Tests for the bounded agent loop, driven by a scripted model.

Each test runs the real pipeline: the fake model chooses actions, pandas
executes the tools for real, and the fake model is handed verified results.

The loop asks the model once per step, then once more for the final wording,
so a one-tool question uses three replies: the tool call, the decision to
answer, and the answer itself.
"""

import json

import pandas as pd
import pytest

from analyst import (
    InvalidOperationError,
    LLMProviderError,
    answer_question,
    profile_dataset,
)
from conftest import BrokenLLM, FakeLLM


def tool_call(tool: str, **arguments) -> str:
    return json.dumps({"action": "call_tool", "tool": tool, "arguments": arguments})


def done(message: str = "The evidence answers the question.") -> str:
    return json.dumps({"action": "answer", "message": message})


def clarify(message: str) -> str:
    return json.dumps({"action": "clarification", "message": message})


GROUP_CALL = tool_call(
    "group_aggregate", group_by="region", aggregations={"revenue": "sum"}
)


# --------------------------------------------------------------------------
# a single tool, as in the previous phase
# --------------------------------------------------------------------------


def test_answers_a_one_tool_question(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North brought in the most revenue, 210.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    assert answer.kind == "answer"
    assert answer.tool == "group_aggregate"
    assert answer.text == "North brought in the most revenue, 210."
    assert len(answer.steps) == 1


def test_the_result_is_kept_as_evidence(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    totals = {row["region"]: row["revenue_sum"] for row in answer.evidence["rows"]}
    assert totals == {"North": 210.0, "South": 120.0, "West": 120.0}
    assert answer.results == [answer.evidence]


def test_the_numbers_come_from_pandas_not_the_model(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    assert "210.0" in model.calls[-1]["prompt"]
    assert "Verified results" in model.calls[-1]["prompt"]


def test_planning_asks_for_json_and_the_answer_does_not(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    assert [call["json_output"] for call in model.calls] == [True, True, False]
    assert "one step at a time" in model.calls[0]["system"]
    assert "final answer" in model.calls[-1]["system"]


def test_the_model_sees_the_real_schema(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    for name in analysis_frame.columns:
        assert name in model.calls[0]["prompt"]


def test_a_supplied_profile_is_reused(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer = answer_question(
        analysis_frame,
        "Which region earns most?",
        model,
        profile=profile_dataset(analysis_frame),
    )

    assert answer.tool == "group_aggregate"


# --------------------------------------------------------------------------
# several tools for one question
# --------------------------------------------------------------------------


def test_runs_two_tools_for_one_question(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        GROUP_CALL,
        tool_call("describe_numeric", columns="revenue"),
        done(),
        "North leads with 210, and revenue averages 105.",
    )

    answer = answer_question(analysis_frame, "Compare regions and revenue", model)

    assert [step.tool for step in answer.steps] == [
        "group_aggregate",
        "describe_numeric",
    ]
    assert len(answer.results) == 2
    assert answer.tool == "describe_numeric"


def test_runs_three_tools_in_sequence(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        GROUP_CALL,
        tool_call("value_counts", column="region"),
        tool_call("sort_rows", by="revenue", ascending=False, limit=2),
        done(),
        "North leads.",
    )

    answer = answer_question(analysis_frame, "Tell me about regions", model)

    assert [step.number for step in answer.steps] == [1, 2, 3]
    assert len(answer.results) == 3


def test_every_step_is_shown_to_the_model(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        GROUP_CALL, tool_call("value_counts", column="region"), done(), "North leads."
    )

    answer_question(analysis_frame, "Regions?", model)

    assert "No steps have been taken" in model.calls[0]["prompt"]
    assert "group_aggregate" in model.calls[1]["prompt"]
    assert "value_counts" in model.calls[2]["prompt"]


# --------------------------------------------------------------------------
# the step limit
# --------------------------------------------------------------------------


def test_the_loop_stops_at_the_step_limit(analysis_frame: pd.DataFrame):
    model = FakeLLM(*[tool_call("value_counts", column="region")] * 10)

    answer = answer_question(analysis_frame, "Loop forever", model, max_steps=3)

    assert answer.kind == "incomplete"
    assert "step limit" in answer.text
    assert len(answer.steps) == 3
    assert len(model.calls) == 3


def test_the_step_limit_is_configurable(analysis_frame: pd.DataFrame):
    model = FakeLLM(*[tool_call("value_counts", column="region")] * 10)

    answer = answer_question(analysis_frame, "Loop forever", model, max_steps=1)

    assert len(answer.steps) == 1


@pytest.mark.parametrize("limit", [0, -1, 2.5, True])
def test_a_bad_step_limit_is_rejected(analysis_frame: pd.DataFrame, limit):
    with pytest.raises(InvalidOperationError, match="max_steps"):
        answer_question(analysis_frame, "Q?", FakeLLM(done()), max_steps=limit)


# --------------------------------------------------------------------------
# recovering inside the loop
# --------------------------------------------------------------------------


def test_an_unknown_tool_is_refused_and_reported_back(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        tool_call("run_python", code="import os"), GROUP_CALL, done(), "North leads."
    )

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    assert answer.steps[0].error is not None
    assert "Unknown tool" in answer.steps[0].error
    assert answer.steps[1].succeeded
    assert "Unknown tool" in model.calls[1]["prompt"]


def test_bad_arguments_are_refused_and_can_be_corrected(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        tool_call("value_counts", limit=5),
        tool_call("value_counts", column="region"),
        done(),
        "Three sales in the north.",
    )

    answer = answer_question(analysis_frame, "Counts?", model)

    assert "Missing required" in answer.steps[0].error
    assert answer.steps[1].succeeded


def test_a_hallucinated_column_is_refused(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        tool_call("value_counts", column="customer_name"),
        done("The dataset has no customer column."),
    )

    answer = answer_question(analysis_frame, "Which customer?", model)

    assert "customer_name" in answer.steps[0].error
    assert answer.text == "The dataset has no customer column."
    assert answer.evidence is None


def test_a_malformed_reply_is_reported_back(analysis_frame: pd.DataFrame):
    model = FakeLLM("I think it is probably North.", GROUP_CALL, done(), "North leads.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    assert answer.steps[0].tool == "(unreadable reply)"
    assert answer.steps[1].succeeded


def test_repeated_malformed_replies_stop_at_the_limit(analysis_frame: pd.DataFrame):
    model = FakeLLM(*["nonsense"] * 10)

    answer = answer_question(analysis_frame, "Q?", model, max_steps=2)

    assert answer.kind == "incomplete"
    assert all(step.error for step in answer.steps)


def test_an_empty_result_is_still_evidence(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        tool_call(
            "filter_rows",
            conditions=[{"column": "revenue", "operator": ">", "value": 9999}],
        ),
        done(),
        "No sales match that.",
    )

    answer = answer_question(analysis_frame, "Sales above 9999?", model)

    assert answer.evidence["row_count"] == 0
    assert answer.text == "No sales match that."


# --------------------------------------------------------------------------
# clarification and unsupported requests
# --------------------------------------------------------------------------


def test_the_model_can_ask_for_clarification(analysis_frame: pd.DataFrame):
    model = FakeLLM(clarify("Do you mean by revenue or by units?"))

    answer = answer_question(analysis_frame, "Show me the best region.", model)

    assert answer.kind == "clarification"
    assert answer.text == "Do you mean by revenue or by units?"
    assert answer.evidence is None
    assert len(model.calls) == 1


def test_clarification_after_a_tool_still_returns_the_question(
    analysis_frame: pd.DataFrame,
):
    model = FakeLLM(GROUP_CALL, clarify("Best by total or by average?"))

    answer = answer_question(analysis_frame, "Which is best?", model)

    assert answer.kind == "clarification"
    assert len(answer.steps) == 1


def test_an_unsupported_request_is_declined_without_a_tool(
    analysis_frame: pd.DataFrame,
):
    model = FakeLLM(done("This dataset has no weather data, so I cannot answer that."))

    answer = answer_question(analysis_frame, "Was it raining?", model)

    assert answer.kind == "answer"
    assert answer.tool is None
    assert answer.evidence is None
    assert "cannot answer" in answer.text
    assert len(model.calls) == 1


# --------------------------------------------------------------------------
# failures that are not the model's to fix
# --------------------------------------------------------------------------


def test_a_provider_failure_is_wrapped(analysis_frame: pd.DataFrame):
    with pytest.raises(LLMProviderError, match="RuntimeError"):
        answer_question(analysis_frame, "Which region earns most?", BrokenLLM())


def test_a_provider_failure_does_not_leak_its_message(analysis_frame: pd.DataFrame):
    broken = BrokenLLM(RuntimeError("key=secret-value-123 rejected"))

    with pytest.raises(LLMProviderError) as raised:
        answer_question(analysis_frame, "Which region earns most?", broken)

    assert "secret-value-123" not in str(raised.value)


def test_an_empty_dataset_is_rejected():
    model = FakeLLM(GROUP_CALL)

    with pytest.raises(InvalidOperationError, match="no data"):
        answer_question(pd.DataFrame({"a": []}), "Anything?", model)

    assert model.calls == []


@pytest.mark.parametrize("question", ["", "   "])
def test_an_empty_question_is_rejected(analysis_frame: pd.DataFrame, question):
    with pytest.raises(InvalidOperationError, match="question is required"):
        answer_question(analysis_frame, question, FakeLLM(done()))


# --------------------------------------------------------------------------
# the answer object
# --------------------------------------------------------------------------


def test_the_answer_serialises(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, done(), "North leads.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    restored = json.loads(json.dumps(answer.to_dict()))
    assert restored["tool"] == "group_aggregate"
    assert restored["steps"][0]["number"] == 1
