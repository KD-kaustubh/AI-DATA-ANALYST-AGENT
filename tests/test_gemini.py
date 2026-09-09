"""Tests for the Gemini client with a stubbed SDK.

The real google-genai package is never called: `_import_sdk` is replaced, so
these tests need no API key and make no network request.
"""

import pytest

from analyst import LLMClient, LLMConfig, LLMConfigurationError, LLMProviderError
from analyst.errors import LLMResponseError
from analyst.gemini import GeminiClient

CONFIG = LLMConfig(api_key="test-key-not-real", model="gemini-test")


class FakeConfigType:
    def __init__(self, **settings):
        self.settings = settings


class FakeTypes:
    GenerateContentConfig = FakeConfigType


class FakeModels:
    def __init__(self, reply, error):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        return type("Response", (), {"text": self.reply})()


class FakeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.models = None


class FakeGenai:
    def __init__(self, reply="ok", error=None, start_error=None):
        self.reply = reply
        self.error = error
        self.start_error = start_error
        self.built = None

    def Client(self, api_key):  # noqa: N802 - mirrors the SDK's name
        if self.start_error:
            raise self.start_error
        client = FakeClient(api_key)
        client.models = FakeModels(self.reply, self.error)
        self.built = client
        return client


@pytest.fixture
def install_fake_sdk(monkeypatch):
    def install(**kwargs):
        fake = FakeGenai(**kwargs)
        monkeypatch.setattr(
            "analyst.gemini._import_sdk", lambda: (fake, FakeTypes)
        )
        return fake

    return install


def test_returns_the_model_reply(install_fake_sdk):
    install_fake_sdk(reply="North leads.")

    assert GeminiClient(CONFIG).generate("Which region?") == "North leads."


def test_sends_the_configured_model_and_system_rules(install_fake_sdk):
    fake = install_fake_sdk()

    GeminiClient(CONFIG).generate("Question?", system="Be careful.")

    call = fake.built.models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "Question?"
    assert call["config"].settings["system_instruction"] == "Be careful."
    assert call["config"].settings["temperature"] == 0.0


def test_asks_for_json_only_when_requested(install_fake_sdk):
    fake = install_fake_sdk()
    client = GeminiClient(CONFIG)

    client.generate("a", json_output=True)
    client.generate("b")

    calls = fake.built.models.calls
    assert calls[0]["config"].settings["response_mime_type"] == "application/json"
    assert calls[1]["config"].settings["response_mime_type"] is None


def test_the_api_key_reaches_the_sdk(install_fake_sdk):
    fake = install_fake_sdk()

    GeminiClient(CONFIG)

    assert fake.built.api_key == "test-key-not-real"


def test_a_provider_failure_becomes_a_project_error(install_fake_sdk):
    install_fake_sdk(error=RuntimeError("503 backend unavailable"))

    with pytest.raises(LLMProviderError, match="gemini-test"):
        GeminiClient(CONFIG).generate("Question?")


def test_a_provider_failure_does_not_repeat_its_message(install_fake_sdk):
    install_fake_sdk(error=RuntimeError("api_key=leaked-value rejected"))

    with pytest.raises(LLMProviderError) as raised:
        GeminiClient(CONFIG).generate("Question?")

    assert "leaked-value" not in str(raised.value)


def test_a_rate_limit_error_carries_its_status_code(install_fake_sdk):
    # google-genai's ClientError exposes the HTTP status as `.code`.
    quota_error = type("ClientError", (Exception,), {"code": 429})(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: ..."
    )
    install_fake_sdk(error=quota_error)

    with pytest.raises(LLMProviderError) as raised:
        GeminiClient(CONFIG).generate("Question?")

    assert raised.value.status_code == 429
    assert "429 ClientError" in str(raised.value)


def test_a_rate_limit_error_does_not_leak_the_quota_message(install_fake_sdk):
    quota_error = type("ClientError", (Exception,), {"code": 429})(
        "429 RESOURCE_EXHAUSTED. account-identifying-detail-xyz"
    )
    install_fake_sdk(error=quota_error)

    with pytest.raises(LLMProviderError) as raised:
        GeminiClient(CONFIG).generate("Question?")

    assert "account-identifying-detail-xyz" not in str(raised.value)


def test_an_error_without_a_numeric_code_has_no_status_code(install_fake_sdk):
    install_fake_sdk(error=RuntimeError("connection reset"))

    with pytest.raises(LLMProviderError) as raised:
        GeminiClient(CONFIG).generate("Question?")

    assert raised.value.status_code is None
    assert "(RuntimeError)." in str(raised.value)


def test_a_non_integer_code_attribute_is_ignored(install_fake_sdk):
    # Some exceptions use `.code` for something other than an HTTP status.
    weird_error = type("Weird", (Exception,), {"code": "invalid_argument"})("bad")
    install_fake_sdk(error=weird_error)

    with pytest.raises(LLMProviderError) as raised:
        GeminiClient(CONFIG).generate("Question?")

    assert raised.value.status_code is None
    assert "(Weird)." in str(raised.value)


@pytest.mark.parametrize("reply", ["", "   ", None])
def test_an_empty_reply_is_reported(install_fake_sdk, reply):
    install_fake_sdk(reply=reply)

    with pytest.raises(LLMResponseError, match="empty"):
        GeminiClient(CONFIG).generate("Question?")


def test_a_bad_client_setup_is_reported(install_fake_sdk):
    install_fake_sdk(start_error=ValueError("bad credentials"))

    with pytest.raises(LLMConfigurationError, match="gemini-test"):
        GeminiClient(CONFIG)


def test_the_client_satisfies_the_protocol(install_fake_sdk):
    install_fake_sdk()

    assert isinstance(GeminiClient(CONFIG), LLMClient)
