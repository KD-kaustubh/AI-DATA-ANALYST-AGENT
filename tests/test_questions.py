"""End-to-end tests of the question flow with a scripted model.

Each test drives the real pipeline: the fake model picks a tool, pandas runs
it for real, and the fake model is handed the verified result.
"""

import json

import pandas as pd
import pytest

from analyst import (
    ColumnNotFoundError,
    InvalidOperationError,
    InvalidToolArgumentsError,
    LLMProviderError,
    LLMResponseError,
    UnknownToolError,
    answer_question,
    profile_dataset,
)
from conftest import BrokenLLM, FakeLLM

GROUP_CALL = json.dumps(
    {
        "action": "call_tool",
        "tool": "group_aggregate",
        "arguments": {"group_by": "region", "aggregations": {"revenue": "sum"}},
    }
)


def test_answers_a_question_end_to_end(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, "North brought in the most revenue, 210.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    assert answer.tool == "group_aggregate"
    assert answer.text == "North brought in the most revenue, 210."
    assert answer.question == "Which region earns most?"


def test_the_answer_keeps_the_verified_result_as_evidence(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    totals = {row["region"]: row["revenue_sum"] for row in answer.evidence["rows"]}
    assert totals == {"North": 210.0, "South": 120.0, "West": 120.0}
    assert answer.evidence["operation"] == "group_aggregate"


def test_the_numbers_come_from_pandas_not_the_model(analysis_frame: pd.DataFrame):
    """The model is shown real totals it never computed."""
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    evidence_prompt = model.calls[1]["prompt"]
    assert "210.0" in evidence_prompt
    assert "Verified result" in evidence_prompt


def test_the_first_call_asks_for_json_and_the_second_does_not(
    analysis_frame: pd.DataFrame,
):
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    assert model.calls[0]["json_output"] is True
    assert model.calls[1]["json_output"] is False
    assert "Never calculate" in model.calls[0]["system"]


def test_the_model_sees_the_real_schema(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer_question(analysis_frame, "Which region earns most?", model)

    prompt = model.calls[0]["prompt"]
    for name in analysis_frame.columns:
        assert name in prompt


def test_a_supplied_profile_is_reused(analysis_frame: pd.DataFrame):
    profile = profile_dataset(analysis_frame)
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer = answer_question(
        analysis_frame, "Which region earns most?", model, profile=profile
    )

    assert answer.tool == "group_aggregate"


def test_the_model_may_decline_to_use_a_tool(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        json.dumps(
            {"action": "answer", "message": "This dataset has no customer column."}
        )
    )

    answer = answer_question(analysis_frame, "Who is the top customer?", model)

    assert answer.text == "This dataset has no customer column."
    assert answer.tool is None
    assert answer.evidence is None
    assert len(model.calls) == 1


def test_a_filter_question_runs_the_filter(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        json.dumps(
            {
                "action": "call_tool",
                "tool": "filter_rows",
                "arguments": {
                    "conditions": [
                        {"column": "revenue", "operator": ">", "value": 70}
                    ]
                },
            }
        ),
        "Three sales are above 70.",
    )

    answer = answer_question(analysis_frame, "How many sales beat 70?", model)

    assert answer.evidence["row_count"] == 3


def test_an_unknown_tool_is_refused(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        json.dumps({"action": "call_tool", "tool": "run_python", "arguments": {}})
    )

    with pytest.raises(UnknownToolError, match="run_python"):
        answer_question(analysis_frame, "Delete the data", model)


def test_bad_tool_arguments_are_refused(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        json.dumps(
            {"action": "call_tool", "tool": "value_counts", "arguments": {"limit": 5}}
        )
    )

    with pytest.raises(InvalidToolArgumentsError):
        answer_question(analysis_frame, "Counts?", model)


def test_a_hallucinated_column_is_refused(analysis_frame: pd.DataFrame):
    model = FakeLLM(
        json.dumps(
            {
                "action": "call_tool",
                "tool": "value_counts",
                "arguments": {"column": "customer_name"},
            }
        )
    )

    with pytest.raises(ColumnNotFoundError, match="customer_name"):
        answer_question(analysis_frame, "Which customer?", model)


def test_a_malformed_reply_is_reported(analysis_frame: pd.DataFrame):
    model = FakeLLM("I think the answer is probably North.")

    with pytest.raises(LLMResponseError):
        answer_question(analysis_frame, "Which region earns most?", model)


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
        answer_question(analysis_frame, question, FakeLLM(GROUP_CALL))


def test_the_answer_serialises(analysis_frame: pd.DataFrame):
    model = FakeLLM(GROUP_CALL, "North leads.")

    answer = answer_question(analysis_frame, "Which region earns most?", model)

    restored = json.loads(json.dumps(answer.to_dict()))
    assert restored["tool"] == "group_aggregate"
    assert restored["text"] == "North leads."
