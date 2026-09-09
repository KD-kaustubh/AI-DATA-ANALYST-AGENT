"""Tests for conversation state, follow-up questions and grounding."""

import json

import pandas as pd
import pytest

from analyst import (
    Conversation,
    GroqClient,
    InvalidOperationError,
    check_grounding,
)
from conftest import FakeLLM
from test_agent import GROUP_CALL, clarify, done, tool_call


def start(frame: pd.DataFrame, *replies: str, **options) -> Conversation:
    return Conversation(frame, FakeLLM(*replies), **options)


# --------------------------------------------------------------------------
# turns
# --------------------------------------------------------------------------


def test_a_turn_is_recorded(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads with 210.")

    answer = session.ask("Which region earns most?")

    assert len(session.turns) == 1
    assert session.turns[0].question == "Which region earns most?"
    assert session.turns[0].answer == answer.text
    assert session.turns[0].kind == "answer"


def test_the_first_question_has_no_history(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads.")

    assert session.history() == []
    session.ask("Which region earns most?")
    assert "Earlier in this conversation" not in session.client.calls[0]["prompt"]


def test_a_follow_up_sees_the_previous_turn(analysis_frame: pd.DataFrame):
    session = start(
        analysis_frame,
        GROUP_CALL,
        done(),
        "North 210, South 120, West 120.",
        tool_call("sort_rows", by="revenue", ascending=False, limit=1),
        done(),
        "North is highest.",
    )
    session.ask("What is total revenue by region?")

    session.ask("Which one is highest?")

    follow_up_prompt = session.client.calls[3]["prompt"]
    assert "Earlier in this conversation" in follow_up_prompt
    assert "What is total revenue by region?" in follow_up_prompt
    assert "210" in follow_up_prompt


def test_the_dataset_is_profiled_once(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads.")
    profile = session.profile

    session.ask("Which region earns most?")

    assert session.profile is profile


def test_reset_forgets_the_turns(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads.")
    session.ask("Which region earns most?")

    session.reset()

    assert session.turns == []
    assert session.history() == []


# --------------------------------------------------------------------------
# history stays bounded
# --------------------------------------------------------------------------


def test_history_keeps_only_the_recent_turns(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, *([GROUP_CALL, done(), "ok"] * 5), history_turns=2)
    for number in range(5):
        session.ask(f"Question {number}?")

    history = session.history()

    assert len(history) == 2
    assert history[0]["question"] == "Question 3?"
    assert history[1]["question"] == "Question 4?"


def test_only_the_newest_turn_carries_its_rows(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, *([GROUP_CALL, done(), "ok"] * 3), history_turns=3)
    for number in range(3):
        session.ask(f"Question {number}?")

    history = session.history()

    assert "results_summary" in history[0]
    assert "rows" not in json.dumps(history[0])
    assert "results" in history[-1]
    assert history[-1]["results"][0]["rows"]


def test_an_older_summary_keeps_what_was_computed(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, *([GROUP_CALL, done(), "ok"] * 2), history_turns=2)
    session.ask("First?")
    session.ask("Second?")

    summary = session.history()[0]["results_summary"][0]

    assert summary["operation"] == "group_aggregate"
    assert summary["row_count"] == 3
    assert "rows" not in summary


def test_history_can_be_switched_off(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "ok", history_turns=0)
    session.ask("First?")

    assert session.history() == []


def test_a_turn_without_results_is_still_remembered(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, clarify("By revenue or by units?"))

    answer = session.ask("Show me the best region.")

    assert answer.kind == "clarification"
    assert session.history()[0] == {
        "question": "Show me the best region.",
        "answer": "By revenue or by units?",
    }


def test_an_empty_dataset_cannot_start_a_conversation():
    with pytest.raises(InvalidOperationError, match="no data"):
        Conversation(pd.DataFrame({"a": []}), FakeLLM())


def test_negative_history_is_rejected(analysis_frame: pd.DataFrame):
    with pytest.raises(InvalidOperationError, match="history_turns"):
        Conversation(analysis_frame, FakeLLM(), history_turns=-1)


# --------------------------------------------------------------------------
# provider compatibility
# --------------------------------------------------------------------------


def test_a_conversation_runs_on_any_client(analysis_frame: pd.DataFrame, monkeypatch):
    """The agent talks to the LLMClient interface, not to a provider."""
    replies = iter([GROUP_CALL, done(), "North leads."])

    class FakeCompletions:
        def create(self, **settings):
            message = type("Message", (), {"content": next(replies)})()
            return type("Reply", (), {"choices": [type("C", (), {"message": message})()]})()

    class FakeSDK:
        def Groq(self, api_key):  # noqa: N802
            chat = type("Chat", (), {"completions": FakeCompletions()})()
            return type("Client", (), {"chat": chat})()

    monkeypatch.setattr("analyst.groq._import_sdk", lambda: FakeSDK())
    from analyst import LLMConfig

    client = GroqClient(LLMConfig(api_key="placeholder", model="m", provider="groq"))
    answer = Conversation(analysis_frame, client).ask("Which region earns most?")

    assert answer.tool == "group_aggregate"


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------


EVIDENCE = [
    {
        "operation": "group_aggregate",
        "rows": [{"region": "North", "revenue_sum": 210.0}],
        "metadata": {"source_rows": 6},
    }
]


def test_numbers_from_the_evidence_are_grounded():
    report = check_grounding("North earned 210.", EVIDENCE)

    assert report.is_grounded
    assert report.unsupported == []


def test_an_invented_number_is_flagged():
    report = check_grounding("North earned 999.", EVIDENCE)

    assert not report.is_grounded
    assert report.unsupported == ["999"]


def test_rounding_is_allowed():
    evidence = [{"rows": [{"mean": 209.153}]}]

    assert check_grounding("about 209.15", evidence).is_grounded
    assert check_grounding("about 209.2", evidence).is_grounded
    assert check_grounding("about 209", evidence).is_grounded
    assert not check_grounding("about 215", evidence).is_grounded


def test_numbers_inside_strings_count_as_evidence():
    evidence = [{"rows": [{"period": "2024", "row_count": 5}]}]

    assert check_grounding("In 2024 there were 5 sales.", evidence).is_grounded


def test_small_counts_are_not_challenged():
    assert check_grounding("There are 2 clear groups.", EVIDENCE).is_grounded


def test_an_answer_with_no_numbers_is_grounded():
    report = check_grounding("The north performs best overall.", EVIDENCE)

    assert report.is_grounded
    assert report.checked == []


def test_the_answer_carries_its_grounding_report(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads with 210.")

    answer = session.ask("Which region earns most?")

    assert answer.grounding["is_grounded"] is True
    assert "210" in answer.grounding["checked"]


def test_an_ungrounded_answer_is_flagged_not_hidden(analysis_frame: pd.DataFrame):
    session = start(analysis_frame, GROUP_CALL, done(), "North leads with 875.")

    answer = session.ask("Which region earns most?")

    assert answer.grounding["is_grounded"] is False
    assert answer.grounding["unsupported"] == ["875"]
