"""Tests for provider configuration and the Groq client.

Both SDKs are stubbed, so nothing here needs an API key or a network.
"""

import pytest

from analyst import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_MODEL,
    PROVIDERS,
    GeminiClient,
    GroqClient,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMProviderError,
    create_client,
    get_provider_settings,
    load_config,
    resolve_provider,
)
from analyst.errors import LLMResponseError

GEMINI_ENV = {"GOOGLE_API_KEY": "gemini-test-placeholder"}
GROQ_ENV = {"GROQ_API_KEY": "groq-test-placeholder"}
BOTH_ENV = {**GEMINI_ENV, **GROQ_ENV}
GROQ_CONFIG = LLMConfig(
    api_key="groq-test-placeholder", model="groq-test", provider="groq"
)


# --------------------------------------------------------------------------
# stubbed Groq SDK
# --------------------------------------------------------------------------


class FakeCompletions:
    def __init__(self, reply, error):
        self.reply = reply
        self.error = error
        self.calls = []

    def create(self, **settings):
        self.calls.append(settings)
        if self.error:
            raise self.error
        # A list of replies is consumed in order; a single one repeats.
        reply = self.reply.pop(0) if isinstance(self.reply, list) else self.reply
        message = type("Message", (), {"content": reply})()
        choice = type("Choice", (), {"message": message})()
        return type("Completion", (), {"choices": [choice]})()


class FakeGroqSDK:
    def __init__(self, reply="ok", error=None, start_error=None):
        self.reply = reply
        self.error = error
        self.start_error = start_error
        self.built = None

    def Groq(self, api_key):  # noqa: N802 - mirrors the SDK's name
        if self.start_error:
            raise self.start_error
        completions = FakeCompletions(self.reply, self.error)
        chat = type("Chat", (), {"completions": completions})()
        self.built = type("Client", (), {"chat": chat, "api_key": api_key})()
        return self.built


@pytest.fixture
def install_fake_groq(monkeypatch):
    def install(**kwargs):
        fake = FakeGroqSDK(**kwargs)
        monkeypatch.setattr("analyst.groq._import_sdk", lambda: fake)
        return fake

    return install


@pytest.fixture
def install_fake_gemini(monkeypatch):
    """Minimal Gemini stub, only enough for create_client to succeed."""

    class FakeTypes:
        GenerateContentConfig = lambda **kwargs: kwargs  # noqa: E731

    class FakeGenai:
        def Client(self, api_key):  # noqa: N802
            return type("Client", (), {"models": None, "api_key": api_key})()

    monkeypatch.setattr(
        "analyst.gemini._import_sdk", lambda: (FakeGenai(), FakeTypes)
    )


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_both_providers_are_registered():
    assert PROVIDERS == ("gemini", "groq")
    assert get_provider_settings("gemini").key_variable == "GOOGLE_API_KEY"
    assert get_provider_settings("groq").key_variable == "GROQ_API_KEY"
    assert get_provider_settings("groq").model_variable == "GROQ_MODEL_NAME"


def test_gemini_configuration_is_unchanged():
    config = load_config(GEMINI_ENV)

    assert config.provider == "gemini"
    assert config.model == DEFAULT_MODEL
    assert load_config({**GEMINI_ENV, "MODEL_NAME": "custom"}).model == "custom"


def test_groq_configuration_reads_its_own_variables():
    config = load_config(GROQ_ENV, provider="groq")

    assert config.provider == "groq"
    assert config.model == DEFAULT_GROQ_MODEL
    assert (
        load_config({**GROQ_ENV, "GROQ_MODEL_NAME": "custom"}, provider="groq").model
        == "custom"
    )


def test_each_provider_ignores_the_other_model_variable():
    config = load_config({**GROQ_ENV, "MODEL_NAME": "gemini-only"}, provider="groq")

    assert config.model == DEFAULT_GROQ_MODEL


@pytest.mark.parametrize(
    "provider, env, variable",
    [
        ("gemini", {}, "GOOGLE_API_KEY"),
        ("gemini", {"GOOGLE_API_KEY": "  "}, "GOOGLE_API_KEY"),
        ("groq", {}, "GROQ_API_KEY"),
        ("groq", GEMINI_ENV, "GROQ_API_KEY"),
    ],
)
def test_a_missing_key_names_the_right_variable(provider, env, variable):
    with pytest.raises(LLMConfigurationError, match=variable):
        load_config(env, provider=provider)


def test_an_unknown_provider_is_refused():
    with pytest.raises(LLMConfigurationError, match="Unknown provider"):
        load_config(GEMINI_ENV, provider="openai")


# --------------------------------------------------------------------------
# choosing a provider
# --------------------------------------------------------------------------


def test_the_provider_variable_wins():
    assert resolve_provider({**BOTH_ENV, "LLM_PROVIDER": "groq"}) == "groq"
    assert resolve_provider({**BOTH_ENV, "LLM_PROVIDER": "GROQ"}) == "groq"


def test_the_provider_falls_back_to_whichever_key_is_set():
    assert resolve_provider(GEMINI_ENV) == "gemini"
    assert resolve_provider(GROQ_ENV) == "groq"


def test_gemini_wins_when_both_keys_are_present():
    assert resolve_provider(BOTH_ENV) == "gemini"


def test_no_key_at_all_is_reported():
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        resolve_provider({})


def test_an_unknown_provider_variable_is_refused():
    with pytest.raises(LLMConfigurationError, match="Unknown provider"):
        resolve_provider({**BOTH_ENV, "LLM_PROVIDER": "hal9000"})


def test_the_factory_builds_the_right_client(install_fake_gemini, install_fake_groq):
    install_fake_groq()

    assert isinstance(create_client(env=GEMINI_ENV), GeminiClient)
    assert isinstance(create_client(env=GROQ_ENV), GroqClient)
    assert isinstance(create_client("groq", env=BOTH_ENV), GroqClient)


def test_the_factory_passes_the_model_through(install_fake_groq):
    install_fake_groq()

    client = create_client(env={**GROQ_ENV, "GROQ_MODEL_NAME": "picked-model"})

    assert client.config.model == "picked-model"


# --------------------------------------------------------------------------
# the Groq client
# --------------------------------------------------------------------------


def test_groq_returns_the_model_reply(install_fake_groq):
    install_fake_groq(reply="North leads.")

    assert GroqClient(GROQ_CONFIG).generate("Which region?") == "North leads."


def test_groq_sends_system_rules_as_a_system_message(install_fake_groq):
    fake = install_fake_groq()

    GroqClient(GROQ_CONFIG).generate("Question?", system="Be careful.")

    call = fake.built.chat.completions.calls[0]
    assert call["model"] == "groq-test"
    assert call["temperature"] == 0.0
    assert call["messages"] == [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "Question?"},
    ]


def test_groq_omits_the_system_message_when_there_is_none(install_fake_groq):
    fake = install_fake_groq()

    GroqClient(GROQ_CONFIG).generate("Question?")

    assert fake.built.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "Question?"}
    ]


def test_groq_asks_for_json_only_when_requested(install_fake_groq):
    fake = install_fake_groq()
    client = GroqClient(GROQ_CONFIG)

    client.generate("a", json_output=True)
    client.generate("b")

    calls = fake.built.chat.completions.calls
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]


def test_groq_passes_the_key_to_the_sdk(install_fake_groq):
    fake = install_fake_groq()

    GroqClient(GROQ_CONFIG)

    assert fake.built.api_key == "groq-test-placeholder"


def test_a_groq_failure_becomes_a_project_error(install_fake_groq):
    install_fake_groq(error=RuntimeError("503 service unavailable"))

    with pytest.raises(LLMProviderError, match="groq-test"):
        GroqClient(GROQ_CONFIG).generate("Question?")


def test_a_groq_failure_does_not_repeat_its_message(install_fake_groq):
    install_fake_groq(error=RuntimeError("api_key=leaked-value rejected"))

    with pytest.raises(LLMProviderError) as raised:
        GroqClient(GROQ_CONFIG).generate("Question?")

    assert "leaked-value" not in str(raised.value)


def test_a_groq_rate_limit_error_carries_its_status_code(install_fake_groq):
    # groq's APIStatusError (and its RateLimitError subclass) expose the
    # HTTP status as `.status_code`.
    quota_error = type("RateLimitError", (Exception,), {"status_code": 429})(
        "Rate limit reached for requests"
    )
    install_fake_groq(error=quota_error)

    with pytest.raises(LLMProviderError) as raised:
        GroqClient(GROQ_CONFIG).generate("Question?")

    assert raised.value.status_code == 429
    assert "429 RateLimitError" in str(raised.value)


def test_a_groq_rate_limit_error_does_not_leak_its_message(install_fake_groq):
    quota_error = type("RateLimitError", (Exception,), {"status_code": 429})(
        "account-identifying-detail-xyz"
    )
    install_fake_groq(error=quota_error)

    with pytest.raises(LLMProviderError) as raised:
        GroqClient(GROQ_CONFIG).generate("Question?")

    assert "account-identifying-detail-xyz" not in str(raised.value)


def test_a_groq_error_without_a_status_code_has_none(install_fake_groq):
    install_fake_groq(error=RuntimeError("connection reset"))

    with pytest.raises(LLMProviderError) as raised:
        GroqClient(GROQ_CONFIG).generate("Question?")

    assert raised.value.status_code is None
    assert "(RuntimeError)." in str(raised.value)


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_an_empty_groq_reply_is_reported(install_fake_groq, reply):
    install_fake_groq(reply=reply)

    with pytest.raises(LLMResponseError, match="empty"):
        GroqClient(GROQ_CONFIG).generate("Question?")


def test_a_bad_groq_setup_is_reported(install_fake_groq):
    install_fake_groq(start_error=ValueError("bad credentials"))

    with pytest.raises(LLMConfigurationError, match="groq-test"):
        GroqClient(GROQ_CONFIG)


# --------------------------------------------------------------------------
# interchangeability and secrecy
# --------------------------------------------------------------------------


def test_both_clients_satisfy_the_same_protocol(install_fake_gemini, install_fake_groq):
    install_fake_groq()

    for client in (create_client(env=GEMINI_ENV), create_client(env=GROQ_ENV)):
        assert isinstance(client, LLMClient)


def test_the_two_clients_are_interchangeable_in_the_question_flow(
    install_fake_groq, analysis_frame
):
    """A Groq client drives the same flow a Gemini client does."""
    import json

    from analyst import answer_question

    install_fake_groq(
        reply=[
            json.dumps(
                {
                    "action": "call_tool",
                    "tool": "value_counts",
                    "arguments": {"column": "region"},
                }
            ),
            json.dumps({"action": "answer", "message": "Counted."}),
            "North appears three times.",
        ]
    )
    answer = answer_question(analysis_frame, "How many per region?", GroqClient(GROQ_CONFIG))

    assert answer.tool == "value_counts"
    assert answer.evidence["rows"][0]["count"] == 3
    assert answer.text == "North appears three times."


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_no_config_ever_shows_its_key(provider):
    env = GEMINI_ENV if provider == "gemini" else GROQ_ENV
    config = load_config(env, provider=provider)

    assert config.api_key not in repr(config)
    assert config.api_key not in str(config)
    assert config.provider in repr(config)
