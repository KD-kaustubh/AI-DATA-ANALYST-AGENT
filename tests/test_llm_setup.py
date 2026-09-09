"""Tests for configuration, dataset context and prompt building.

Nothing here contacts a provider or reads a real API key.
"""

import json

import pandas as pd
import pytest

from analyst import (
    DEFAULT_MODEL,
    LLMConfigurationError,
    build_context,
    describe_tools,
    load_config,
    profile_dataset,
)
from analyst.prompts import (
    ANSWER_RULES,
    MAX_EVIDENCE_ROWS,
    TOOL_SELECTION_RULES,
    build_answer_prompt,
    build_tool_prompt,
)

FAKE_ENV = {"GOOGLE_API_KEY": "test-key-not-real", "MODEL_NAME": "gemini-test"}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_reads_key_and_model_from_the_environment():
    config = load_config(FAKE_ENV)

    assert config.api_key == "test-key-not-real"
    assert config.model == "gemini-test"


def test_model_name_falls_back_to_a_default():
    assert load_config({"GOOGLE_API_KEY": "k"}).model == DEFAULT_MODEL
    assert load_config({"GOOGLE_API_KEY": "k", "MODEL_NAME": ""}).model == DEFAULT_MODEL


@pytest.mark.parametrize("env", [{}, {"GOOGLE_API_KEY": ""}, {"GOOGLE_API_KEY": "   "}])
def test_a_missing_key_is_reported_clearly(env):
    with pytest.raises(LLMConfigurationError, match="GOOGLE_API_KEY"):
        load_config(env)


def test_the_api_key_never_appears_in_a_repr():
    config = load_config(FAKE_ENV)

    assert "test-key-not-real" not in repr(config)
    assert "gemini-test" in repr(config)


# --------------------------------------------------------------------------
# dataset context
# --------------------------------------------------------------------------


def test_context_comes_from_the_profile(analysis_frame: pd.DataFrame):
    context = build_context(profile_dataset(analysis_frame))

    assert context.row_count == 6
    assert context.column_count == 5
    assert [column.name for column in context.columns] == [
        "region",
        "product",
        "units",
        "revenue",
        "sold_at",
    ]


def test_context_separates_column_kinds(analysis_frame: pd.DataFrame):
    context = build_context(profile_dataset(analysis_frame))

    assert context.numeric_columns == ["units", "revenue"]
    assert context.datetime_columns == ["sold_at"]
    assert "region" in context.categorical_columns


def test_context_reports_missing_values_and_warnings(sample_frame: pd.DataFrame):
    context = build_context(profile_dataset(sample_frame))

    city = next(column for column in context.columns if column.name == "city")
    assert city.missing_count == 2
    assert any("duplicate" in warning for warning in context.warnings)


def test_context_is_json_serialisable(analysis_frame: pd.DataFrame):
    context = build_context(profile_dataset(analysis_frame))

    assert json.loads(json.dumps(context.to_dict()))["row_count"] == 6


def test_context_text_names_every_column(analysis_frame: pd.DataFrame):
    text = build_context(profile_dataset(analysis_frame)).to_prompt_text()

    for name in analysis_frame.columns:
        assert name in text
    assert "Rows: 6" in text


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def test_tool_prompt_carries_the_schema_and_the_tools(analysis_frame: pd.DataFrame):
    context = build_context(profile_dataset(analysis_frame))

    prompt = build_tool_prompt("Which region earns most?", context, describe_tools())

    assert "Which region earns most?" in prompt
    assert "revenue" in prompt
    assert "group_aggregate" in prompt


def test_tool_prompt_does_not_ship_the_dataframe(analysis_frame: pd.DataFrame):
    context = build_context(profile_dataset(analysis_frame))

    prompt = build_tool_prompt("How much?", context, describe_tools())

    # Individual cell values must not be in the prompt; only the schema is.
    for value in ["120.0", "2024-02-28", "2025-01-02"]:
        assert value not in prompt
    assert prompt.count("North") <= 1  # at most the one example value


def test_rules_forbid_inventing_numbers_and_running_code():
    assert "Never calculate" in TOOL_SELECTION_RULES
    assert "cannot write or run code" in TOOL_SELECTION_RULES
    assert "Never invent a column name" in TOOL_SELECTION_RULES
    assert "only the numbers in the supplied result" in ANSWER_RULES


def test_answer_prompt_includes_the_verified_result():
    result = {"operation": "value_counts", "rows": [{"value": "North", "count": 3}]}

    prompt = build_answer_prompt("How many northern sales?", prompt_result := result)

    assert "How many northern sales?" in prompt
    assert "North" in prompt
    assert prompt_result["rows"][0]["count"] == 3


def test_answer_prompt_trims_a_long_result():
    rows = [{"value": index} for index in range(MAX_EVIDENCE_ROWS + 20)]

    prompt = build_answer_prompt("Everything?", {"operation": "filter", "rows": rows})

    assert "rows_omitted" in prompt
    assert f'"value": {MAX_EVIDENCE_ROWS + 19}' not in prompt


def test_answer_prompt_leaves_a_short_result_alone():
    result = {"operation": "filter", "rows": [{"value": 1}]}

    assert "rows_omitted" not in build_answer_prompt("Q?", result)
